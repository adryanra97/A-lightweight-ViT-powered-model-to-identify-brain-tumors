"""
Author: Adryan R A

Brain Tumor Dataset Module

This module provides data loading and preprocessing functionality for brain tumor 
classification using medical image datasets. It includes custom dataset classes,
data augmentation pipelines, and utility functions for preparing training data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import os
import pandas as pd
import numpy as np
from typing import Tuple, List, Optional, Dict
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class BrainTumorDataset(Dataset):
    """
    Custom Dataset class for Brain Tumor Classification
    
    This dataset class handles loading and preprocessing of brain tumor images
    from a directory structure where images are organized by class labels.
    
    Args:
        data_dir (str): Root directory containing class subdirectories
        transform (A.Compose, optional): Albumentations transform pipeline
        image_size (tuple): Target image size (height, width)
    
    Attributes:
        data_dir: Path to the dataset directory
        transform: Image transformation pipeline
        image_size: Target dimensions for image resizing
        images: List of tuples (image_path, label)
        classes: List of class names
        class_to_idx: Dictionary mapping class names to indices
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        transform: Optional[A.Compose] = None,
        image_size: int = 224
    ):
        self.data_dir = data_dir
        self.split = split
        self.image_size = image_size
        self.transform = transform
        
        # Load data
        self.data = self._load_data()
        
        # Default transform if none provided
        if self.transform is None:
            self.transform = self._get_default_transforms()
    
    def _load_data(self) -> List[Dict]:
        """Load image paths and labels"""
        data = []
        
        # Class mapping
        class_to_idx = {"no": 0, "yes": 1}
        
        for class_name in ["no", "yes"]:
            class_dir = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            for filename in os.listdir(class_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_path = os.path.join(class_dir, filename)
                    data.append({
                        "image_path": image_path,
                        "label": class_to_idx[class_name],
                        "class_name": class_name
                    })
        
        return data
    
    def _get_default_transforms(self) -> A.Compose:
        """Get default transforms based on split"""
        if self.split == "train":
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.RandomRotate90(p=0.5),
                A.HorizontalFlip(p=0.3),
                A.VerticalFlip(p=0.3),
                A.RandomBrightnessContrast(
                    brightness_limit=0.2,
                    contrast_limit=0.2,
                    p=0.5
                ),
                A.GaussianBlur(blur_limit=3, p=0.3),
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
        else:
            return A.Compose([
                A.Resize(self.image_size, self.image_size),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
                ToTensorV2()
            ])
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.data[idx]
        
        # Load image
        image = cv2.imread(item["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        return image, item["label"]
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Get distribution of classes"""
        distribution = {}
        for item in self.data:
            class_name = item["class_name"]
            distribution[class_name] = distribution.get(class_name, 0) + 1
        return distribution


def create_data_loaders(
    data_dir: str,
    batch_size: int = 32,
    image_size: int = 224,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 4,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test data loaders
    
    Args:
        data_dir: Path to dataset directory
        batch_size: Batch size for data loaders
        image_size: Size to resize images to
        train_split: Proportion of data for training
        val_split: Proportion of data for validation
        test_split: Proportion of data for testing
        num_workers: Number of workers for data loading
        random_seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    
    # Set random seed
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    
    # Load all data
    full_dataset = BrainTumorDataset(data_dir, split="train")
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    # Create separate datasets with appropriate transforms
    train_data = BrainTumorDataset(
        data_dir,
        split="train",
        image_size=image_size
    )
    
    val_data = BrainTumorDataset(
        data_dir,
        split="val",
        image_size=image_size
    )
    
    test_data = BrainTumorDataset(
        data_dir,
        split="test",
        image_size=image_size
    )
    
    # Update datasets with split indices
    train_indices = train_dataset.indices
    val_indices = val_dataset.indices
    test_indices = test_dataset.indices
    
    train_data.data = [full_dataset.data[i] for i in train_indices]
    val_data.data = [full_dataset.data[i] for i in val_indices]
    test_data.data = [full_dataset.data[i] for i in test_indices]
    
    # Create data loaders
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader, test_loader


def get_class_weights(data_dir: str) -> torch.Tensor:
    """
    Calculate class weights for imbalanced dataset
    
    Args:
        data_dir: Path to dataset directory
    
    Returns:
        Tensor of class weights
    """
    dataset = BrainTumorDataset(data_dir)
    distribution = dataset.get_class_distribution()
    
    total_samples = sum(distribution.values())
    num_classes = len(distribution)
    
    weights = []
    for class_name in ["no", "yes"]:
        if class_name in distribution:
            weight = total_samples / (num_classes * distribution[class_name])
            weights.append(weight)
        else:
            weights.append(1.0)
    
    return torch.FloatTensor(weights)


if __name__ == "__main__":
    # Test dataset
    data_dir = "../brain_tumor_dataset"
    
    if os.path.exists(data_dir):
        dataset = BrainTumorDataset(data_dir)
        print(f"Dataset size: {len(dataset)}")
        print(f"Class distribution: {dataset.get_class_distribution()}")
        
        # Test data loading
        sample_image, sample_label = dataset[0]
        print(f"Sample image shape: {sample_image.shape}")
        print(f"Sample label: {sample_label}")
        
        # Test class weights
        weights = get_class_weights(data_dir)
        print(f"Class weights: {weights}")
    else:
        print(f"Dataset directory {data_dir} not found")
