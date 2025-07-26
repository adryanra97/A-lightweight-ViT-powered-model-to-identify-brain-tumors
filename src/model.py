"""
Vision Transformer Model for Brain Tumor Detection
Author: Adryan R A

This module implements a Vision Transformer (ViT) model architecture
specifically designed for brain tumor detection in MRI images.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional, Dict, Tuple, Any

class BrainTumorViT(nn.Module):
    """
    Vision Transformer for Brain Tumor Classification
    
    This class implements a Vision Transformer model using the timm library
    for efficient ViT implementations, specifically adapted for medical image
    classification tasks.
    
    Args:
        model_name (str): Name of the ViT model from timm library
        num_classes (int): Number of output classes (default: 2)
        pretrained (bool): Whether to use pretrained weights (default: True)
        dropout (float): Dropout rate for regularization (default: 0.1)
    
    Attributes:
        backbone: The ViT backbone model from timm
        feature_dim: Dimension of the feature vector from backbone
        classifier: Custom classification head
    """
    
    def __init__(
        self,
        model_name: str = "vit_base_patch16_224",
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.1
    ):
        super(BrainTumorViT, self).__init__()
        
        # Load pretrained ViT model from timm
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier head
            global_pool=""  # Remove global pooling
        )
        
        # Get feature dimension
        self.feature_dim = self.backbone.num_features
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self) -> None:
        """Initialize classifier layer weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # Extract features using ViT backbone
        features = self.backbone(x)
        
        # Apply classifier
        output = self.classifier(features)
        
        return output
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features without classification
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Feature tensor of shape (batch_size, feature_dim)
        """
        with torch.no_grad():
            features = self.backbone(x)
            pooled_features = nn.AdaptiveAvgPool2d(1)(features)
            return pooled_features.flatten(1)


def create_model(
    model_name: str = "vit_base_patch16_224",
    num_classes: int = 2,
    pretrained: bool = True,
    dropout: float = 0.1
) -> BrainTumorViT:
    """
    Factory function to create brain tumor ViT model
    
    Args:
        model_name: Name of the ViT model from timm
        num_classes: Number of output classes
        pretrained: Whether to use pretrained weights
        dropout: Dropout rate
    
    Returns:
        BrainTumorViT model instance
    """
    return BrainTumorViT(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout
    )


# Available ViT models for different computational requirements
AVAILABLE_MODELS: Dict[str, str] = {
    "vit_tiny_patch16_224": "Tiny ViT - Fastest, Lower Accuracy",
    "vit_small_patch16_224": "Small ViT - Balanced Speed/Accuracy",
    "vit_base_patch16_224": "Base ViT - Good Accuracy, Moderate Speed",
    "vit_large_patch16_224": "Large ViT - Best Accuracy, Slower",
    "vit_base_patch32_224": "Base ViT (32x32 patches) - Faster",
    "deit_base_patch16_224": "DeiT Base - Efficient Training",
    "swin_base_patch4_window7_224": "Swin Transformer - Hierarchical"
}


if __name__ == "__main__":
    # Test model creation
    model = create_model()
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Test forward pass
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
