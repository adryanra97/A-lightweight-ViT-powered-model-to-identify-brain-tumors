"""
Updated Inference Module for Brain Tumor Detection
Compatible with the latest notebook fixes
Author: Adryan R A
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import os
from typing import Dict, Tuple, List, Union, Optional, Any
import logging

# Import the updated model
from .model import BrainTumorViT, get_val_transforms

class BrainTumorPredictor:
    """
    Inference class for brain tumor detection with all notebook fixes applied
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        """
        Initialize the predictor
        
        Args:
            model_path: Path to the trained model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[BrainTumorViT] = None
        self.transform = get_val_transforms(image_size=224)
        self.class_names: List[str] = ['No Tumor', 'Tumor']
        
        # Load model
        self.load_model(model_path)
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_model(self, model_path: str) -> None:
        """Load the trained model with proper error handling"""
        try:
            # Create model instance
            self.model = BrainTumorViT(
                model_name='vit_base_patch16_224',
                num_classes=2,
                pretrained=False,  # We're loading our own weights
                dropout=0.1
            )
            
            # Load weights with proper device mapping
            if os.path.exists(model_path):
                # Try loading the checkpoint
                checkpoint = torch.load(
                    model_path, 
                    map_location=self.device,
                    weights_only=False  # Allow loading of full model state
                )
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['state_dict'])
                else:
                    # Assume the checkpoint is just the state dict
                    self.model.load_state_dict(checkpoint)
                
                self.logger.info(f"Model loaded successfully from {model_path}")
            else:
                self.logger.warning(f"Model file not found: {model_path}. Using pretrained weights.")
                # Fall back to pretrained model if file doesn't exist
                self.model = BrainTumorViT(
                    model_name='vit_base_patch16_224',
                    num_classes=2,
                    pretrained=True,
                    dropout=0.1
                )
            
            # Move model to device and set to eval mode
            self.model.to(self.device)
            self.model.eval()
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            # Create a pretrained model as fallback
            self.model = BrainTumorViT(
                model_name='vit_base_patch16_224',
                num_classes=2,
                pretrained=True,
                dropout=0.1
            )
            self.model.to(self.device)
            self.model.eval()
            self.logger.warning("Using pretrained model as fallback")
    
    def preprocess_image(self, image_input) -> torch.Tensor:
        """
        Preprocess image for inference
        
        Args:
            image_input: Can be file path, PIL Image, or numpy array
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        try:
            # Handle different input types
            if isinstance(image_input, str):
                # File path
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Could not load image from {image_input}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_input, Image.Image):
                # PIL Image
                image = np.array(image_input.convert('RGB'))
            elif isinstance(image_input, np.ndarray):
                # Numpy array
                if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                    image = image_input
                elif len(image_input.shape) == 3 and image_input.shape[2] == 4:
                    # RGBA to RGB
                    image = image_input[:, :, :3]
                else:
                    raise ValueError(f"Unsupported image shape: {image_input.shape}")
            else:
                raise ValueError(f"Unsupported image input type: {type(image_input)}")
            
            # Apply transforms
            transformed = self.transform(image=image)
            image_tensor = transformed['image']
            
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)
            
            return image_tensor.to(self.device)
            
        except Exception as e:
            self.logger.error(f"Error preprocessing image: {e}")
            raise
    
    def predict(self, image_input, return_probabilities: bool = True) -> Dict:
        """
        Make prediction on an image
        
        Args:
            image_input: Image to predict on (path, PIL Image, or numpy array)
            return_probabilities: Whether to return class probabilities
            
        Returns:
            Dict containing prediction results
        """
        try:
            # Preprocess image
            image_tensor = self.preprocess_image(image_input)
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = F.softmax(outputs, dim=1)
                
                # Get predicted class
                _, predicted_class = torch.max(outputs, 1)
                predicted_class = predicted_class.item()
                
                # Get confidence score
                confidence = probabilities[0][predicted_class].item()
                
                # Prepare results
                result = {
                    'predicted_class': predicted_class,
                    'predicted_label': self.class_names[predicted_class],
                    'confidence': confidence
                }
                
                if return_probabilities:
                    result['probabilities'] = {
                        self.class_names[i]: probabilities[0][i].item() 
                        for i in range(len(self.class_names))
                    }
                
                return result
                
        except Exception as e:
            self.logger.error(f"Error during prediction: {e}")
            return {
                'error': str(e),
                'predicted_class': None,
                'predicted_label': None,
                'confidence': 0.0
            }
    
    def predict_batch(self, image_list: List, batch_size: int = 8) -> List[Dict]:
        """
        Make predictions on multiple images
        
        Args:
            image_list: List of images to predict on
            batch_size: Batch size for processing
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for i in range(0, len(image_list), batch_size):
            batch = image_list[i:i + batch_size]
            
            # Process batch
            try:
                batch_tensors = []
                for image in batch:
                    tensor = self.preprocess_image(image)
                    batch_tensors.append(tensor.squeeze(0))  # Remove batch dim
                
                # Stack into batch
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                # Make predictions
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    probabilities = F.softmax(outputs, dim=1)
                    
                    _, predicted_classes = torch.max(outputs, 1)
                    
                    # Process results
                    for j, pred_class in enumerate(predicted_classes):
                        pred_class = pred_class.item()
                        confidence = probabilities[j][pred_class].item()
                        
                        result = {
                            'predicted_class': pred_class,
                            'predicted_label': self.class_names[pred_class],
                            'confidence': confidence,
                            'probabilities': {
                                self.class_names[k]: probabilities[j][k].item() 
                                for k in range(len(self.class_names))
                            }
                        }
                        results.append(result)
                        
            except Exception as e:
                # Add error results for failed batch
                for _ in batch:
                    results.append({
                        'error': str(e),
                        'predicted_class': None,
                        'predicted_label': None,
                        'confidence': 0.0
                    })
        
        return results
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model"""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': 'BrainTumorViT',
            'architecture': 'vit_base_patch16_224',
            'device': str(self.device),
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'class_names': self.class_names
        }
