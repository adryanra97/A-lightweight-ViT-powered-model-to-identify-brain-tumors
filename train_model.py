#!/usr/bin/env python3
"""
Brain Tumor Detection Training Script
=====================================

This script trains a Vision Transformer (ViT) model for brain tumor detection
using transfer learning and state-of-the-art techniques.

Usage:
    python train_model.py [options]

Example:
    python train_model.py --epochs 50 --batch_size 32 --lr 1e-4
"""

import argparse
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.model import create_model
from src.dataset import create_data_loaders, get_class_weights
from src.trainer import Trainer

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Brain Tumor Detection Model')
    
    # Data arguments
    parser.add_argument('--data_dir', type=str, default='brain_tumor_dataset',
                        help='Path to dataset directory')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224',
                        help='ViT model variant to use')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Number of output classes')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained weights')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adamw', 'adam', 'sgd'],
                        help='Optimizer to use')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau'],
                        help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Number of warmup epochs')
    parser.add_argument('--freeze_epochs', type=int, default=10,
                        help='Number of epochs to freeze backbone')
    
    # Data split arguments
    parser.add_argument('--train_split', type=float, default=0.7,
                        help='Training data split ratio')
    parser.add_argument('--val_split', type=float, default=0.15,
                        help='Validation data split ratio')
    parser.add_argument('--test_split', type=float, default=0.15,
                        help='Test data split ratio')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Output directory for saving models')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name for organizing outputs')
    parser.add_argument('--save_best_only', action='store_true', default=True,
                        help='Save only the best model')
    
    # Hardware arguments
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cpu', 'cuda'],
                        help='Device to use for training')
    parser.add_argument('--mixed_precision', action='store_true', default=False,
                        help='Use mixed precision training')
    
    # Monitoring arguments
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--save_frequency', type=int, default=10,
                        help='Save model every N epochs')
    parser.add_argument('--log_frequency', type=int, default=1,
                        help='Log metrics every N epochs')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    
    return parser.parse_args()

def set_random_seeds(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_experiment_directory(args):
    """Setup experiment directory structure"""
    if args.experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.experiment_name = f"brain_tumor_vit_{timestamp}"
    
    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / "checkpoints").mkdir(exist_ok=True)
    (experiment_dir / "plots").mkdir(exist_ok=True)
    (experiment_dir / "logs").mkdir(exist_ok=True)
    
    return experiment_dir

def save_config(args, experiment_dir):
    """Save training configuration"""
    config_dict = vars(args).copy()
    config_dict['timestamp'] = datetime.now().isoformat()
    config_dict['experiment_dir'] = str(experiment_dir)
    
    config_path = experiment_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    
    print(f"Configuration saved to: {config_path}")
    return config_path

def create_model_and_trainer(args, train_loader, val_loader, test_loader, experiment_dir):
    """Create model and trainer"""
    
    # Create model
    print(f"Creating {args.model_name} model...")
    model = create_model(
        model_name=args.model_name,
        num_classes=args.num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout
    )
    
    # Get class weights for imbalanced data
    class_weights = get_class_weights(args.data_dir)
    print(f"Class weights: {class_weights}")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=args.device,
        class_weights=class_weights
    )
    
    # Setup optimizer
    trainer.setup_optimizer(
        optimizer_name=args.optimizer,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Setup scheduler
    scheduler_kwargs = {}
    if args.scheduler == 'cosine':
        scheduler_kwargs['T_max'] = args.epochs
        scheduler_kwargs['eta_min'] = args.lr * 0.01
    elif args.scheduler == 'step':
        scheduler_kwargs['step_size'] = args.epochs // 3
        scheduler_kwargs['gamma'] = 0.1
    elif args.scheduler == 'plateau':
        scheduler_kwargs['patience'] = args.early_stopping_patience // 2
        scheduler_kwargs['factor'] = 0.5
    
    trainer.setup_scheduler(args.scheduler, **scheduler_kwargs)
    
    return trainer

def train_model(trainer, args, experiment_dir):
    """Train the model"""
    
    print("Starting training...")
    print(f"Experiment directory: {experiment_dir}")
    print(f"Training for {args.epochs} epochs")
    print(f"Device: {trainer.device}")
    
    # Model save path
    model_save_path = experiment_dir / "checkpoints" / "best_model.pth"
    
    # Train model
    history = trainer.train(
        epochs=args.epochs,
        save_best=True,
        save_path=str(model_save_path),
        early_stopping_patience=args.early_stopping_patience
    )
    
    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(experiment_dir / "logs" / "training_history.csv", index=False)
    
    # Plot training history
    trainer.plot_training_history(save_path=experiment_dir / "plots" / "training_history.png")
    
    return history

def evaluate_model(trainer, args, experiment_dir):
    """Evaluate the trained model"""
    
    print("Evaluating model on test set...")
    
    # Evaluate on test set
    metrics, targets, predictions, probabilities = trainer.evaluate()
    
    # Save detailed evaluation results
    results_df = pd.DataFrame({
        'true_label': targets,
        'predicted_label': predictions,
        'probability_tumor': probabilities
    })
    results_df.to_csv(experiment_dir / "logs" / "test_predictions.csv", index=False)
    
    # Create confusion matrix plot
    trainer.plot_confusion_matrix(
        targets, predictions, 
        save_path=experiment_dir / "plots" / "confusion_matrix.png"
    )
    
    # Generate classification report
    class_names = ['No Tumor', 'Tumor']
    report = classification_report(targets, predictions, target_names=class_names, output_dict=True)
    
    # Save classification report
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(experiment_dir / "logs" / "classification_report.csv")
    
    # Print summary
    print("\\nEvaluation Results:")
    print("-" * 50)
    for metric, value in metrics.items():
        print(f"{metric.upper()}: {value:.4f}")
    
    return metrics, report

def create_model_summary(trainer, args, experiment_dir, metrics):
    """Create comprehensive model summary"""
    
    summary = {
        'experiment_name': args.experiment_name,
        'model_architecture': args.model_name,
        'dataset_size': {
            'train': len(trainer.train_loader.dataset),
            'val': len(trainer.val_loader.dataset),
            'test': len(trainer.test_loader.dataset)
        },
        'training_config': {
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'learning_rate': args.lr,
            'optimizer': args.optimizer,
            'scheduler': args.scheduler
        },
        'final_metrics': metrics,
        'model_parameters': {
            'total': sum(p.numel() for p in trainer.model.parameters()),
            'trainable': sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        },
        'training_time': None,  # Will be filled by main function
        'best_epoch': len(trainer.history['val_auc']) if hasattr(trainer, 'history') else None,
        'best_auc': max(trainer.history['val_auc']) if hasattr(trainer, 'history') else None
    }
    
    # Save summary
    with open(experiment_dir / "model_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    return summary

def main():
    """Main training function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Set random seeds
    set_random_seeds(args.seed)
    
    # Setup experiment directory
    experiment_dir = setup_experiment_directory(args)
    print(f"Experiment directory: {experiment_dir}")
    
    # Save configuration
    save_config(args, experiment_dir)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
        random_seed=args.seed
    )
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_loader.dataset):,}")
    print(f"  Validation: {len(val_loader.dataset):,}")
    print(f"  Test: {len(test_loader.dataset):,}")
    
    # Create model and trainer
    trainer = create_model_and_trainer(args, train_loader, val_loader, test_loader, experiment_dir)
    
    # Record start time
    start_time = time.time()
    
    # Train model
    history = train_model(trainer, args, experiment_dir)
    
    # Calculate training time
    training_time = time.time() - start_time
    
    # Evaluate model
    metrics, report = evaluate_model(trainer, args, experiment_dir)
    
    # Create model summary
    summary = create_model_summary(trainer, args, experiment_dir, metrics)
    summary['training_time'] = training_time
    
    # Save updated summary
    with open(experiment_dir / "model_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)
    
    # Final summary
    print("\\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Experiment: {args.experiment_name}")
    print(f"Training time: {training_time/3600:.2f} hours")
    print(f"Best AUC: {summary.get('best_auc', 'N/A'):.4f}")
    print(f"Final Test AUC: {metrics['auc']:.4f}")
    print(f"Model saved to: {experiment_dir / 'checkpoints' / 'best_model.pth'}")
    print(f"Results saved to: {experiment_dir}")
    print("="*60)

if __name__ == "__main__":
    main()
