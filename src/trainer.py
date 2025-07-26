import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from typing import Dict, List, Tuple, Optional
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import os

class Trainer:
    """
    Trainer class for brain tumor classification model
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: str = "auto",
        class_weights: Optional[torch.Tensor] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        self.model.to(self.device)
        
        # Set up loss function with class weights
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": [],
            "learning_rates": []
        }
    
    def setup_optimizer(
        self,
        optimizer_name: str = "adamw",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
        **kwargs
    ):
        """Setup optimizer"""
        if optimizer_name.lower() == "adamw":
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_name.lower() == "adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                **kwargs
            )
        elif optimizer_name.lower() == "sgd":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
                momentum=kwargs.get("momentum", 0.9),
                **kwargs
            )
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    def setup_scheduler(
        self,
        scheduler_name: str = "cosine",
        **kwargs
    ):
        """Setup learning rate scheduler"""
        if scheduler_name.lower() == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=kwargs.get("T_max", 50),
                eta_min=kwargs.get("eta_min", 1e-6)
            )
        elif scheduler_name.lower() == "step":
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=kwargs.get("step_size", 10),
                gamma=kwargs.get("gamma", 0.1)
            )
        elif scheduler_name.lower() == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="max",
                factor=kwargs.get("factor", 0.5),
                patience=kwargs.get("patience", 5),
                verbose=True
            )
        else:
            self.scheduler = None
    
    def train_epoch(self) -> Tuple[float, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "Acc": f"{100. * correct / total:.2f}%"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def validate(self) -> Tuple[float, float, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                # Store for AUC calculation
                probabilities = torch.softmax(output, dim=1)
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())  # Probability of positive class
                all_targets.extend(target.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        
        # Calculate AUC
        try:
            auc = roc_auc_score(all_targets, all_probabilities)
        except:
            auc = 0.0
        
        return avg_loss, accuracy, auc
    
    def train(
        self,
        epochs: int = 50,
        save_best: bool = True,
        save_path: str = "best_model.pth",
        early_stopping_patience: int = 10
    ):
        """Train the model"""
        best_auc = 0.0
        patience_counter = 0
        
        print(f"Starting training for {epochs} epochs...")
        print(f"Model will be saved to: {save_path}")
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 60)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_auc = self.validate()
            
            # Update scheduler
            if hasattr(self, 'scheduler') and self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_auc)
                else:
                    self.scheduler.step()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Store history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_auc"].append(val_auc)
            self.history["learning_rates"].append(current_lr)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")
            print(f"Learning Rate: {current_lr:.2e}")
            
            # Save best model
            if save_best and val_auc > best_auc:
                best_auc = val_auc
                patience_counter = 0
                print(f"New best AUC: {best_auc:.4f}. Saving model...")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_auc': best_auc,
                    'history': self.history
                }, save_path)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
                break
        
        print(f"\nTraining completed! Best AUC: {best_auc:.4f}")
        return self.history
    
    def evaluate(self, loader: DataLoader = None) -> Dict[str, float]:
        """Evaluate the model on test set"""
        if loader is None:
            loader = self.test_loader
        
        self.model.eval()
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(loader, desc="Evaluating"):
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.model(data)
                probabilities = torch.softmax(output, dim=1)
                _, predicted = output.max(1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_probabilities.extend(probabilities[:, 1].cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='binary')
        recall = recall_score(all_targets, all_predictions, average='binary')
        f1 = f1_score(all_targets, all_predictions, average='binary')
        auc = roc_auc_score(all_targets, all_probabilities)
        
        metrics = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "auc": auc
        }
        
        print("\nEvaluation Results:")
        print("-" * 40)
        for metric, value in metrics.items():
            print(f"{metric.upper()}: {value:.4f}")
        
        return metrics, all_targets, all_predictions, all_probabilities
    
    def plot_training_history(self, save_path: str = None):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history["train_loss"], label="Train Loss")
        axes[0, 0].plot(self.history["val_loss"], label="Val Loss")
        axes[0, 0].set_title("Training and Validation Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history["train_acc"], label="Train Accuracy")
        axes[0, 1].plot(self.history["val_acc"], label="Val Accuracy")
        axes[0, 1].set_title("Training and Validation Accuracy")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Accuracy (%)")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC plot
        axes[1, 0].plot(self.history["val_auc"], label="Val AUC", color="green")
        axes[1, 0].set_title("Validation AUC")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("AUC")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate plot
        axes[1, 1].plot(self.history["learning_rates"], label="Learning Rate", color="red")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_yscale("log")
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Training history plot saved to: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, targets, predictions, save_path: str = None):
        """Plot confusion matrix"""
        cm = confusion_matrix(targets, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No Tumor", "Tumor"],
            yticklabels=["No Tumor", "Tumor"]
        )
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()


def load_model(model_path: str, model: nn.Module, device: str = "auto") -> nn.Module:
    """Load trained model"""
    if device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    print(f"Model loaded from {model_path}")
    print(f"Best AUC during training: {checkpoint.get('best_auc', 'N/A')}")
    
    return model
