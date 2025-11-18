"""
Training and evaluation utilities for trajectory prediction models.
"""
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class Trainer:
    """Handles model training, validation, and testing."""
    
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: optim.Optimizer, device: str = "cuda" if torch.cuda.is_available() else "cpu", x_max: float = 120.0, y_max: float = 53.3):

        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.x_max = x_max
        self.y_max = y_max
        
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for xb, yb in train_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            pred = self.model(xb)
            loss = self.criterion(pred, yb)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(train_loader)
        return avg_loss
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(self.device)
                yb = yb.to(self.device)
                
                pred = self.model(xb)
                loss = self.criterion(pred, yb)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        return avg_loss
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, epochs: int, checkpoint_dir: Optional[Path] = None, early_stopping_patience: int = 5) -> Dict[str, Any]:
        patience_counter = 0
        
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        epoch: int = -1

        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            self.val_losses.append(val_loss)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.5f}, Val Loss: {val_loss:.5f}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                if checkpoint_dir:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        self.model.state_dict(),
                        checkpoint_dir / "cnn_rnn_best_model.pt"
                    )
                    print(f"Saved best model (val_loss: {val_loss:.5f})")
            else:
                patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"\nEarly stopping triggered after {epoch+1} epochs")
                    break
        
        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "epochs_trained": epoch + 1 
        }
    
    def test(self, test_loader: DataLoader) -> Dict[str, Any]:
        """
        Evaluate model on test set (unseen data).
        
        Returns:
            Dictionary with test metrics and predictions
        """
        self.model.eval()
        
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(self.device)
                pred = self.model(xb)
                
                all_preds.append(pred.cpu().numpy())
                all_targets.append(yb.numpy())
        
        # Concatenate all batches
        preds = np.concatenate(all_preds, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        # Denormalize to yards
        preds_yards = self._denormalize(preds)
        targets_yards = self._denormalize(targets)
        
        # Calculate RMSE (using Kaggle formula)
        rmse = self._calculate_rmse(targets_yards, preds_yards)
        
        print(f"\n{'='*50}")
        print(f"TEST SET RESULTS (Unseen Data)")
        print(f"{'='*50}")
        print(f"Average 2D RMSE: {rmse:.3f} yards")
        print(f"Test samples: {len(preds)}")
        
        return {
            "rmse": rmse,
            "predictions": preds_yards,
            "targets": targets_yards,
            "predictions_norm": preds,
            "targets_norm": targets
        }
    
    def _denormalize(self, coords: np.ndarray) -> np.ndarray:
        """Denormalize coordinates back to yards."""
        coords_yards = coords.copy()
        coords_yards[..., 0] *= self.x_max
        coords_yards[..., 1] *= self.y_max
        return coords_yards
    
    def _calculate_rmse(self, targets: np.ndarray, predictions: np.ndarray) -> float:
        """Calculate RMSE using Kaggle formula."""
        return np.sqrt(np.mean((targets - predictions) ** 2) / 2.0)
    
    def load_best_model(self, checkpoint_path: Path):
        """Load the best model from checkpoint."""
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        print(f"Loaded model from {checkpoint_path}")
    
    def plot_training_history(self, save_path: Optional[Path] = None):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved training history plot to {save_path}")
        plt.show()
    
    def visualize_predictions(self, predictions: np.ndarray, targets: np.ndarray, num_examples: int = 3, save_path: Optional[Path] = None):
        """Visualize sample trajectory predictions."""
        fig, axes = plt.subplots(1, num_examples, figsize=(5 * num_examples, 4))
        
        if num_examples == 1:
            axes = [axes]
        
        for idx, ax in enumerate(axes):
            if idx >= len(predictions):
                break
            
            true_traj = targets[idx]
            pred_traj = predictions[idx]
            
            # Calculate RMSE for this example
            rmse = self._calculate_rmse(
                true_traj.reshape(1, -1, 2),
                pred_traj.reshape(1, -1, 2)
            )
            
            ax.plot(true_traj[:, 0], true_traj[:, 1], 
                   label='True', color='green', linewidth=2, marker='o', markersize=3)
            ax.plot(pred_traj[:, 0], pred_traj[:, 1], 
                   label='Predicted', color='red', linewidth=2, 
                   linestyle='--', marker='x', markersize=3)
            
            ax.set_xlabel('X (yards)')
            ax.set_ylabel('Y (yards)')
            ax.set_title(f'Example {idx+1}\nRMSE: {rmse:.2f} yards')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved predictions visualization to {save_path}")
        plt.show()