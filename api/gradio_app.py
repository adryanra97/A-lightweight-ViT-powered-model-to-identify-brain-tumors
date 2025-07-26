import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from datetime import datetime
import json
import zipfile
import tempfile
from typing import List, Tuple, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import custom modules
from src.model import create_model, load_model
from src.dataset import BrainTumorDataset

class BrainTumorGradioApp:
    def __init__(self):
        self.model = None
        self.device = None
        self.transform = None
        self.config = {
            "model_path": "models/best_brain_tumor_model.pth",
            "image_size": 224,
            "class_names": ["No Tumor", "Tumor"],
            "database_path": "gradio_predictions.db"
        }
        self.initialize_app()
    
    def initialize_app(self):
        """Initialize the Gradio application"""
        self.setup_device()
        self.load_model()
        self.setup_transform()
        self.initialize_database()
    
    def setup_device(self):
        """Setup computing device"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def load_model(self):
        """Load the trained model"""
        try:
            # Create model
            self.model = create_model(
                model_name="vit_base_patch16_224",
                num_classes=2,
                pretrained=True
            )
            
            # Load trained weights if available
            if os.path.exists(self.config["model_path"]):
                self.model = load_model(self.config["model_path"], self.model, device=self.device)
                print(f"Model loaded from {self.config['model_path']}")
            else:
                print("Warning: Trained model not found. Using pretrained weights only.")
                self.model.to(self.device)
            
            self.model.eval()
            print("Model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def setup_transform(self):
        """Setup image preprocessing transform"""
        self.transform = A.Compose([
            A.Resize(self.config["image_size"], self.config["image_size"]),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def initialize_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.config["database_path"])
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                prediction TEXT,
                confidence REAL,
                timestamp DATETIME,
                user_feedback TEXT,
                correct_label TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batch_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_name TEXT,
                total_images INTEGER,
                correct_predictions INTEGER,
                accuracy REAL,
                timestamp DATETIME
            )
        """)
        
        conn.commit()
        conn.close()
    
    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess image for model input"""
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_np = np.array(image)
        transformed = self.transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0)
        
        return image_tensor
    
    def predict_single_image(self, image: Image.Image) -> Tuple[str, float, Dict, str]:
        """Make prediction on a single image"""
        if self.model is None:
            return "Model not loaded", 0.0, {}, "Error"
        
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image)
            image_tensor = image_tensor.to(self.device)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                predicted_class = predicted.item()
                confidence_score = confidence.item()
                
                # Get all class probabilities
                class_probs = probabilities[0].cpu().numpy()
                prob_dict = {
                    self.config["class_names"][i]: float(prob) 
                    for i, prob in enumerate(class_probs)
                }
            
            prediction = self.config["class_names"][predicted_class]
            risk_level = self.get_risk_level(confidence_score, predicted_class)
            
            return prediction, confidence_score, prob_dict, risk_level
            
        except Exception as e:
            return f"Error: {str(e)}", 0.0, {}, "Error"
    
    def get_risk_level(self, confidence: float, predicted_class: int) -> str:
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
    
    def gradio_predict(self, image):
        """Gradio interface for single image prediction"""
        if image is None:
            return "Please upload an image", "", {}, ""
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        prediction, confidence, probabilities, risk_level = self.predict_single_image(image)
        
        # Store prediction
        self.store_prediction("gradio_upload", prediction, confidence)
        
        # Format output
        result_text = f"**Prediction:** {prediction}\\n**Confidence:** {confidence:.1%}\\n**Risk Level:** {risk_level}"
        
        # Create probability chart
        prob_df = pd.DataFrame(list(probabilities.items()), columns=['Class', 'Probability'])
        
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(prob_df['Class'], prob_df['Probability'], 
                     color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        ax.set_title('Class Probabilities', fontweight='bold', fontsize=14)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        return result_text, fig, f"Confidence: {confidence:.1%}", risk_level
    
    def process_batch_images(self, files):
        """Process batch of images with labels for accuracy testing"""
        if not files:
            return "No files uploaded", "", "", ""
        
        results = []
        correct_predictions = 0
        total_predictions = 0
        
        for file in files:
            try:
                # Extract true label from filename
                filename = os.path.basename(file.name)
                true_label = self.extract_label_from_filename(filename)
                
                # Load and predict
                image = Image.open(file.name)
                prediction, confidence, _, risk_level = self.predict_single_image(image)
                
                # Check if prediction is correct
                is_correct = (
                    (true_label == "no" and prediction == "No Tumor") or
                    (true_label == "yes" and prediction == "Tumor")
                )
                
                if is_correct:
                    correct_predictions += 1
                
                results.append({
                    'Filename': filename,
                    'True Label': true_label.title(),
                    'Prediction': prediction,
                    'Confidence': f"{confidence:.1%}",
                    'Correct': "✅" if is_correct else "❌",
                    'Risk Level': risk_level
                })
                
                total_predictions += 1
                
            except Exception as e:
                results.append({
                    'Filename': os.path.basename(file.name),
                    'True Label': "Unknown",
                    'Prediction': f"Error: {str(e)}",
                    'Confidence': "N/A",
                    'Correct': "❌",
                    'Risk Level': "Error"
                })
        
        # Calculate accuracy
        accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
        
        # Store batch test results
        self.store_batch_test(f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}", 
                             total_predictions, correct_predictions, accuracy)
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        # Create accuracy visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy pie chart
        correct_count = results_df['Correct'].value_counts()
        colors = ['#4ECDC4', '#FF6B6B']
        labels = ['Correct', 'Incorrect']
        sizes = [correct_count.get('✅', 0), correct_count.get('❌', 0)]
        
        ax1.pie(sizes, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
        ax1.set_title(f'Accuracy: {accuracy:.1%}', fontweight='bold', fontsize=14)
        
        # Prediction distribution
        pred_counts = results_df['Prediction'].value_counts()
        ax2.bar(pred_counts.index, pred_counts.values, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        ax2.set_title('Prediction Distribution', fontweight='bold', fontsize=14)
        ax2.set_ylabel('Count')
        
        plt.tight_layout()
        
        # Summary text
        summary = f"""
        **Batch Processing Results:**
        - Total Images: {total_predictions}
        - Correct Predictions: {correct_predictions}
        - Accuracy: {accuracy:.1%}
        - Tumor Detection Rate: {results_df[results_df['Prediction'] == 'Tumor'].shape[0] / total_predictions:.1%}
        """
        
        return summary, results_df, fig, f"Processed {total_predictions} images with {accuracy:.1%} accuracy"
    
    def extract_label_from_filename(self, filename: str) -> str:
        """Extract true label from filename"""
        filename_lower = filename.lower()
        if any(keyword in filename_lower for keyword in ['no', 'normal', 'healthy']):
            return "no"
        elif any(keyword in filename_lower for keyword in ['yes', 'tumor', 'cancer', 'positive']):
            return "yes"
        else:
            # Default based on common naming patterns
            if filename_lower.startswith('n') or 'no' in filename_lower:
                return "no"
            elif filename_lower.startswith('y') or 'yes' in filename_lower:
                return "yes"
            return "unknown"
    
    def store_prediction(self, filename: str, prediction: str, confidence: float):
        """Store prediction in database"""
        try:
            conn = sqlite3.connect(self.config["database_path"])
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO predictions (filename, prediction, confidence, timestamp)
                VALUES (?, ?, ?, ?)
            """, (filename, prediction, confidence, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database error: {e}")
    
    def store_batch_test(self, batch_name: str, total: int, correct: int, accuracy: float):
        """Store batch test results"""
        try:
            conn = sqlite3.connect(self.config["database_path"])
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO batch_tests (batch_name, total_images, correct_predictions, accuracy, timestamp)
                VALUES (?, ?, ?, ?, ?)
            """, (batch_name, total, correct, accuracy, datetime.now().isoformat()))
            
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"Database error: {e}")
    
    def get_prediction_history(self):
        """Get prediction history for display"""
        try:
            conn = sqlite3.connect(self.config["database_path"])
            df = pd.read_sql_query("""
                SELECT filename, prediction, confidence, timestamp 
                FROM predictions 
                ORDER BY timestamp DESC 
                LIMIT 100
            """, conn)
            conn.close()
            
            if df.empty:
                return "No prediction history available"
            
            return df
            
        except Exception as e:
            return f"Error loading history: {e}"
    
    def get_batch_test_history(self):
        """Get batch test history"""
        try:
            conn = sqlite3.connect(self.config["database_path"])
            df = pd.read_sql_query("""
                SELECT batch_name, total_images, correct_predictions, accuracy, timestamp 
                FROM batch_tests 
                ORDER BY timestamp DESC 
                LIMIT 50
            """, conn)
            conn.close()
            
            if df.empty:
                return "No batch test history available"
            
            return df
            
        except Exception as e:
            return f"Error loading batch history: {e}"
    
    def create_interface(self):
        """Create Gradio interface"""
        
        # Custom CSS
        css = """
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        .gr-button {
            background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
            border: none;
            color: white;
            font-weight: bold;
        }
        .gr-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        }
        """
        
        with gr.Blocks(css=css, title="Brain Tumor Detection", theme=gr.themes.Soft()) as interface:
            
            gr.Markdown("""
            # 🧠 Brain Tumor Detection using Vision Transformer
            
            Upload MRI brain images to detect the presence of tumors using state-of-the-art AI technology.
            This application uses a Vision Transformer (ViT) model trained specifically for medical image analysis.
            
            ⚠️ **Medical Disclaimer:** This tool is for research and educational purposes only. 
            Always consult with qualified medical professionals for proper diagnosis and treatment.
            """)
            
            with gr.Tabs():
                # Single Image Prediction Tab
                with gr.TabItem("🔍 Single Image Analysis"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            image_input = gr.Image(
                                type="pil", 
                                label="Upload MRI Brain Image",
                                height=400
                            )
                            predict_btn = gr.Button("🔬 Analyze Image", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            result_text = gr.Markdown(label="Analysis Results")
                            confidence_text = gr.Textbox(label="Confidence Score", interactive=False)
                            risk_text = gr.Textbox(label="Risk Assessment", interactive=False)
                            prob_plot = gr.Plot(label="Class Probabilities")
                    
                    predict_btn.click(
                        fn=self.gradio_predict,
                        inputs=[image_input],
                        outputs=[result_text, prob_plot, confidence_text, risk_text]
                    )
                
                # Batch Testing Tab
                with gr.TabItem("📊 Batch Testing & Accuracy"):
                    gr.Markdown("""
                    ### Batch Image Testing
                    Upload multiple images with proper naming (include 'no', 'yes', 'tumor', etc. in filenames) 
                    to test model accuracy and get detailed performance metrics.
                    """)
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            batch_files = gr.File(
                                file_count="multiple",
                                label="Upload Multiple MRI Images",
                                file_types=["image"]
                            )
                            batch_process_btn = gr.Button("📈 Process Batch", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            batch_summary = gr.Markdown(label="Batch Results Summary")
                            batch_status = gr.Textbox(label="Processing Status", interactive=False)
                    
                    with gr.Row():
                        batch_results = gr.Dataframe(label="Detailed Results")
                        batch_plot = gr.Plot(label="Performance Metrics")
                    
                    batch_process_btn.click(
                        fn=self.process_batch_images,
                        inputs=[batch_files],
                        outputs=[batch_summary, batch_results, batch_plot, batch_status]
                    )
                
                # History and Analytics Tab
                with gr.TabItem("📋 History & Analytics"):
                    gr.Markdown("### Prediction History and Performance Analytics")
                    
                    with gr.Row():
                        history_btn = gr.Button("🔄 Load Prediction History")
                        batch_history_btn = gr.Button("📊 Load Batch Test History")
                    
                    with gr.Row():
                        with gr.Column():
                            prediction_history = gr.Dataframe(label="Recent Predictions")
                        with gr.Column():
                            batch_history = gr.Dataframe(label="Batch Test Results")
                    
                    history_btn.click(
                        fn=self.get_prediction_history,
                        outputs=[prediction_history]
                    )
                    
                    batch_history_btn.click(
                        fn=self.get_batch_test_history,
                        outputs=[batch_history]
                    )
                
                # Information Tab
                with gr.TabItem("ℹ️ About & Instructions"):
                    gr.Markdown("""
                    ## About This Application
                    
                    This brain tumor detection system uses a **Vision Transformer (ViT)** model, which represents 
                    the state-of-the-art in computer vision for medical image analysis.
                    
                    ### Model Information
                    - **Architecture:** Vision Transformer (ViT-Base-Patch16-224)
                    - **Training Data:** Brain MRI dataset with tumor/no-tumor classifications
                    - **Input Size:** 224x224 pixels
                    - **Classes:** No Tumor, Tumor
                    
                    ### How to Use
                    
                    #### Single Image Analysis:
                    1. Upload an MRI brain image (JPG, PNG, etc.)
                    2. Click "Analyze Image"
                    3. Review the prediction, confidence score, and risk assessment
                    
                    #### Batch Testing:
                    1. Prepare images with descriptive filenames (include 'no', 'yes', 'tumor', etc.)
                    2. Upload multiple images
                    3. Click "Process Batch" to get accuracy metrics
                    4. Review detailed results and performance visualization
                    
                    #### Filename Conventions for Batch Testing:
                    - **No Tumor:** Include words like 'no', 'normal', 'healthy' (e.g., 'no_tumor_001.jpg')
                    - **Tumor:** Include words like 'yes', 'tumor', 'positive' (e.g., 'yes_tumor_001.jpg')
                    
                    ### Risk Level Interpretation
                    - **Low Risk:** High confidence prediction of no tumor
                    - **Monitor:** Lower confidence no tumor prediction - routine monitoring recommended
                    - **Low-Moderate Risk:** Lower confidence tumor detection - additional screening suggested
                    - **Moderate Risk:** Moderate confidence tumor detection - further investigation recommended  
                    - **High Risk:** High confidence tumor detection - immediate medical attention required
                    
                    ### Technical Details
                    - **Preprocessing:** Images are resized to 224x224 and normalized using ImageNet statistics
                    - **Augmentation:** Training includes medical-appropriate data augmentation
                    - **Evaluation:** Model performance measured using AUC-ROC, precision, recall, and F1-score
                    
                    ### Important Notes
                    ⚠️ **Medical Disclaimer:** This application is for research and educational purposes only. 
                    It should not be used as a substitute for professional medical diagnosis. Always consult 
                    with qualified healthcare professionals for proper medical evaluation and treatment.
                    
                    🔒 **Privacy:** No uploaded images are permanently stored. Predictions are logged for 
                    performance tracking but without image data.
                    """)
            
            gr.Markdown("""
            ---
            **Developed with ❤️ using Vision Transformer technology for medical image analysis**
            """)
        
        return interface

def main():
    """Main function to run the Gradio app"""
    # Create app instance
    app = BrainTumorGradioApp()
    
    # Create and launch interface
    interface = app.create_interface()
    
    # Launch the app
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True to create public link
        debug=False,
        show_error=True
    )

if __name__ == "__main__":
    main()
