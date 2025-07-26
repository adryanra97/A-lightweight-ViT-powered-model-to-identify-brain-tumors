from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn.functional as F
from PIL import Image
import io
import numpy as np
import cv2
import os
import json
import sqlite3
from datetime import datetime
from typing import List, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import base64

# Import custom modules
from src.model import create_model, load_model
from src.dataset import BrainTumorDataset

app = FastAPI(
    title="Brain Tumor Detection API",
    description="API for brain tumor detection in MRI images using Vision Transformer",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
MODEL = None
DEVICE = None
TRANSFORM = None
DATABASE_PATH = "predictions.db"

# Configuration
CONFIG = {
    "model_path": "models/best_brain_tumor_model.pth",
    "image_size": 224,
    "class_names": ["No Tumor", "Tumor"],
    "confidence_threshold": 0.5
}

def initialize_database():
    """Initialize SQLite database for storing predictions"""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            prediction TEXT,
            confidence REAL,
            timestamp DATETIME,
            image_data BLOB,
            user_feedback TEXT,
            correct_label TEXT
        )
    """)
    
    conn.commit()
    conn.close()

def get_transform():
    """Get image preprocessing transform"""
    return A.Compose([
        A.Resize(CONFIG["image_size"], CONFIG["image_size"]),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])

def load_trained_model():
    """Load the trained model"""
    global MODEL, DEVICE, TRANSFORM
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
    
    # Create model
    MODEL = create_model(
        model_name="vit_base_patch16_224",
        num_classes=2,
        pretrained=True
    )
    
    # Load trained weights if available
    if os.path.exists(CONFIG["model_path"]):
        MODEL = load_model(CONFIG["model_path"], MODEL, device=DEVICE)
        print(f"Model loaded from {CONFIG['model_path']}")
    else:
        print("Warning: Trained model not found. Using pretrained weights only.")
        MODEL.to(DEVICE)
    
    MODEL.eval()
    
    # Get transform
    TRANSFORM = get_transform()
    
    print("Model loaded successfully!")

@app.on_event("startup")
async def startup_event():
    """Initialize the API"""
    initialize_database()
    load_trained_model()

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Brain Tumor Detection API",
        "version": "1.0.0",
        "status": "active",
        "model_loaded": MODEL is not None
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "device": str(DEVICE) if DEVICE else None
    }

def preprocess_image(image: Image.Image) -> torch.Tensor:
    """Preprocess image for model input"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert PIL to numpy
    image_np = np.array(image)
    
    # Apply transforms
    transformed = TRANSFORM(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)  # Add batch dimension
    
    return image_tensor

def predict_image(image: Image.Image) -> dict:
    """Make prediction on a single image"""
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Preprocess image
        image_tensor = preprocess_image(image)
        image_tensor = image_tensor.to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            outputs = MODEL(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            predicted_class = predicted.item()
            confidence_score = confidence.item()
            
            # Get class probabilities
            class_probs = probabilities[0].cpu().numpy()
            
        result = {
            "prediction": CONFIG["class_names"][predicted_class],
            "confidence": float(confidence_score),
            "class_probabilities": {
                CONFIG["class_names"][i]: float(prob) 
                for i, prob in enumerate(class_probs)
            },
            "risk_level": get_risk_level(confidence_score, predicted_class)
        }
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

def get_risk_level(confidence: float, predicted_class: int) -> str:
    """Determine risk level based on prediction and confidence"""
    if predicted_class == 0:  # No tumor
        return "Low Risk" if confidence > 0.8 else "Monitor"
    else:  # Tumor detected
        if confidence > 0.9:
            return "High Risk - Immediate Attention Required"
        elif confidence > 0.7:
            return "Moderate Risk - Further Investigation Recommended"
        else:
            return "Low-Moderate Risk - Additional Screening Suggested"

@app.post("/predict")
async def predict_single_image(file: UploadFile = File(...)):
    """Predict brain tumor in a single MRI image"""
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Make prediction
        result = predict_image(image)
        
        # Store prediction in database
        store_prediction(file.filename, result, contents)
        
        return JSONResponse(content={
            "success": True,
            "filename": file.filename,
            "result": result,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch")
async def predict_batch_images(files: List[UploadFile] = File(...)):
    """Predict brain tumor in multiple MRI images"""
    
    if len(files) > 50:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 50 files allowed per batch")
    
    results = []
    
    for file in files:
        if not file.content_type.startswith('image/'):
            results.append({
                "filename": file.filename,
                "success": False,
                "error": "Invalid file type"
            })
            continue
        
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            result = predict_image(image)
            store_prediction(file.filename, result, contents)
            
            results.append({
                "filename": file.filename,
                "success": True,
                "result": result
            })
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    # Calculate batch statistics
    successful_predictions = [r for r in results if r["success"]]
    tumor_detected = sum(1 for r in successful_predictions 
                        if r["result"]["prediction"] == "Tumor")
    
    batch_stats = {
        "total_images": len(files),
        "successful_predictions": len(successful_predictions),
        "failed_predictions": len(files) - len(successful_predictions),
        "tumor_detected": tumor_detected,
        "tumor_detection_rate": tumor_detected / len(successful_predictions) * 100 
                               if successful_predictions else 0
    }
    
    return JSONResponse(content={
        "success": True,
        "batch_stats": batch_stats,
        "results": results,
        "timestamp": datetime.now().isoformat()
    })

def store_prediction(filename: str, result: dict, image_data: bytes):
    """Store prediction in database"""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO predictions (filename, prediction, confidence, timestamp, image_data)
            VALUES (?, ?, ?, ?, ?)
        """, (
            filename,
            result["prediction"],
            result["confidence"],
            datetime.now().isoformat(),
            image_data
        ))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"Database error: {e}")

@app.post("/feedback")
async def submit_feedback(
    prediction_id: int = Form(...),
    correct_label: str = Form(...),
    feedback: Optional[str] = Form(None)
):
    """Submit feedback for a prediction"""
    
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE predictions 
            SET user_feedback = ?, correct_label = ?
            WHERE id = ?
        """, (feedback, correct_label, prediction_id))
        
        conn.commit()
        conn.close()
        
        return JSONResponse(content={
            "success": True,
            "message": "Feedback submitted successfully"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/predictions/history")
async def get_prediction_history(limit: int = 100):
    """Get prediction history"""
    
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, filename, prediction, confidence, timestamp, 
                   user_feedback, correct_label
            FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        history = []
        for row in rows:
            history.append({
                "id": row[0],
                "filename": row[1],
                "prediction": row[2],
                "confidence": row[3],
                "timestamp": row[4],
                "user_feedback": row[5],
                "correct_label": row[6]
            })
        
        return JSONResponse(content={
            "success": True,
            "history": history,
            "total_records": len(history)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/predictions/stats")
async def get_prediction_stats():
    """Get prediction statistics"""
    
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        
        # Predictions by class
        cursor.execute("""
            SELECT prediction, COUNT(*) 
            FROM predictions 
            GROUP BY prediction
        """)
        class_counts = dict(cursor.fetchall())
        
        # Average confidence
        cursor.execute("SELECT AVG(confidence) FROM predictions")
        avg_confidence = cursor.fetchone()[0] or 0
        
        # Feedback statistics
        cursor.execute("""
            SELECT COUNT(*) FROM predictions 
            WHERE user_feedback IS NOT NULL
        """)
        feedback_count = cursor.fetchone()[0]
        
        conn.close()
        
        stats = {
            "total_predictions": total_predictions,
            "class_distribution": class_counts,
            "average_confidence": round(avg_confidence, 3),
            "feedback_received": feedback_count,
            "feedback_rate": round(feedback_count / total_predictions * 100, 1) 
                           if total_predictions > 0 else 0
        }
        
        return JSONResponse(content={
            "success": True,
            "stats": stats
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/model/info")
async def get_model_info():
    """Get model information"""
    
    model_info = {
        "model_name": "Vision Transformer (ViT)",
        "architecture": "vit_base_patch16_224",
        "input_size": f"{CONFIG['image_size']}x{CONFIG['image_size']}",
        "num_classes": len(CONFIG["class_names"]),
        "class_names": CONFIG["class_names"],
        "device": str(DEVICE) if DEVICE else None,
        "model_loaded": MODEL is not None
    }
    
    if MODEL is not None:
        total_params = sum(p.numel() for p in MODEL.parameters())
        trainable_params = sum(p.numel() for p in MODEL.parameters() if p.requires_grad)
        
        model_info.update({
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "model_size_mb": round(total_params * 4 / (1024 * 1024), 2)  # Approximate size
        })
    
    return JSONResponse(content={
        "success": True,
        "model_info": model_info
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
