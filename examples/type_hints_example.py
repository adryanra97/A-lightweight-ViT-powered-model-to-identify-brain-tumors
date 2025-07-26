"""
Type Hints Examples for Brain Tumor Detection Project
Author: Adryan R A
Year: 2025

This file demonstrates proper type hints usage in Python for the brain tumor detection project.
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import torch
import numpy as np
from PIL import Image

# Basic type hints
def process_image_path(image_path: str) -> Image.Image:
    """
    Load an image from a file path
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image object
    """
    return Image.open(image_path)

# Multiple return types with Tuple
def get_image_dimensions(image: Image.Image) -> Tuple[int, int]:
    """
    Get image width and height
    
    Args:
        image: PIL Image object
        
    Returns:
        Tuple of (width, height)
    """
    return image.size

# Optional parameters
def resize_image(image: Image.Image, size: int = 224, maintain_aspect: bool = True) -> Image.Image:
    """
    Resize image with optional parameters
    
    Args:
        image: Input image
        size: Target size (default: 224)
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    if maintain_aspect:
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize((size, size))

# Dict type hints
def create_prediction_result(
    prediction: str, 
    confidence: float, 
    probabilities: List[float]
) -> Dict[str, Union[str, float, List[float]]]:
    """
    Create a prediction result dictionary
    
    Args:
        prediction: Predicted class name
        confidence: Confidence score (0-1)
        probabilities: List of class probabilities
        
    Returns:
        Dictionary containing prediction results
    """
    return {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": probabilities,
        "timestamp": "2025-07-26T12:00:00"
    }

# Class with type hints
class ModelPredictor:
    """Example predictor class with type hints"""
    
    def __init__(self, model_path: str, device: Optional[str] = None) -> None:
        """
        Initialize predictor
        
        Args:
            model_path: Path to model file
            device: Device to use ('cpu' or 'cuda')
        """
        self.model_path: str = model_path
        self.device: str = device or 'cpu'
        self.model: Optional[torch.nn.Module] = None
        self.class_names: List[str] = ['No Tumor', 'Tumor']
    
    def load_model(self) -> None:
        """Load the model"""
        # Model loading logic here
        pass
    
    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """
        Make prediction on image
        
        Args:
            image: Input image
            
        Returns:
            Prediction results
        """
        # Prediction logic here
        return {
            "prediction": "No Tumor",
            "confidence": 0.95
        }

# Generic type hints for ML functions
def calculate_metrics(
    predictions: List[int], 
    targets: List[int]
) -> Dict[str, float]:
    """
    Calculate classification metrics
    
    Args:
        predictions: Predicted class indices
        targets: True class indices
        
    Returns:
        Dictionary of metrics
    """
    accuracy = sum(p == t for p, t in zip(predictions, targets)) / len(predictions)
    return {"accuracy": accuracy}

# Union types for flexible inputs
def process_input(
    data: Union[str, Image.Image, np.ndarray]
) -> torch.Tensor:
    """
    Process different types of input data
    
    Args:
        data: Can be file path, PIL Image, or numpy array
        
    Returns:
        Processed tensor
    """
    if isinstance(data, str):
        # Handle file path
        image = Image.open(data)
    elif isinstance(data, Image.Image):
        # Handle PIL Image
        image = data
    elif isinstance(data, np.ndarray):
        # Handle numpy array
        image = Image.fromarray(data)
    else:
        raise ValueError("Unsupported data type")
    
    # Convert to tensor (simplified)
    return torch.tensor([1.0])

# Example usage in main
if __name__ == "__main__":
    # These examples show how type hints help with IDE autocomplete
    # and catching type errors early
    
    predictor: ModelPredictor = ModelPredictor("model.pth")
    result: Dict[str, Any] = predictor.predict(Image.new('RGB', (224, 224)))
    
    print(f"Prediction: {result['prediction']}")
    print("Type hints help catch errors and improve code readability!")
