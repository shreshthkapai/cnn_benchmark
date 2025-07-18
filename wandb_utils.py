import wandb
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from utils.data_loader import get_cifar10_info

class WandbLogger:
    """Minimal yet powerful W&B integration for FAANG-level ML projects."""
    
    def __init__(self, project: str = "cifar10-benchmark", entity: Optional[str] = None):
        self.project = project
        self.entity = entity
        self.run = None
    
    def init_experiment(self, config: Dict[str, Any], model: nn.Module, model_name: str):
        """Initialize W&B run with model architecture tracking."""
        # Auto-detect model stats for config
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        enhanced_config = {
            **config,
            'model_name': model_name,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 ** 2),
            'architecture': str(model.__class__.__name__)
        }
        
        self.run = wandb.init(
            project=self.project,
            entity=self.entity,
            config=enhanced_config,
            name=f"{model_name}-{wandb.util.generate_id()}"
        )
        
        # Log model architecture
        wandb.watch(model, log_freq=100, log_graph=True)
        return self.run
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log training metrics with automatic prefixing."""
        wandb.log(metrics, step=step)
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, epoch: int):
        """Log confusion matrix as W&B image."""
        cifar10_info = get_cifar10_info()
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=cifar10_info['class_names'],
                   yticklabels=cifar10_info['class_names'])
        plt.title(f'Confusion Matrix - Epoch {epoch}')
        plt.tight_layout()
        
        wandb.log({
            "confusion_matrix": wandb.Image(plt),
            "epoch": epoch
        })
        plt.close()
    
    def log_model_checkpoint(self, model: nn.Module, optimizer, epoch: int, 
                           metrics: Dict[str, float], is_best: bool = False):
        """Log model checkpoint with metadata."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            **metrics
        }
        
        filename = f"model_epoch_{epoch}.pth"
        torch.save(checkpoint, filename)
        
        artifact = wandb.Artifact(
            name=f"model-{self.run.id}",
            type="model",
            metadata={"epoch": epoch, "is_best": is_best, **metrics}
        )
        artifact.add_file(filename)
        wandb.log_artifact(artifact)
    
    def finish(self):
        """Cleanup W&B run."""
        if self.run:
            wandb.finish()

def create_hyperparameter_sweep():
    """FAANG-level hyperparameter sweep configuration."""
    return {
        'method': 'bayes',
        'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
        'parameters': {
            'learning_rate': {'distribution': 'log_uniform', 'min': 1e-5, 'max': 1e-2},
            'batch_size': {'values': [32, 64, 128]},
            'weight_decay': {'distribution': 'log_uniform', 'min': 1e-6, 'max': 1e-3},
            'optimizer': {'values': ['adamw', 'sgd']},
            'scheduler': {'values': ['cosine', 'step']},
            'dropout_rate': {'distribution': 'uniform', 'min': 0.1, 'max': 0.5}
        }
    }

def run_hyperparameter_sweep(train_fn, sweep_config: Dict[str, Any], count: int = 20):
    """Execute hyperparameter sweep with W&B."""
    sweep_id = wandb.sweep(sweep_config, project="cifar10-benchmark")
    wandb.agent(sweep_id, train_fn, count=count)

# Integration with existing training loop
def enhanced_train_step(model, train_loader, val_loader, optimizer, criterion, 
                       scheduler, num_epochs, device, logger: WandbLogger):
    """Enhanced training with W&B logging."""
    model.to(device)
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss, train_acc = 0.0, 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == targets).float().mean().item()
        
        # Validation
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        all_preds, all_targets = [], []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == targets).float().mean().item()
                
                all_preds.extend(outputs.argmax(1).cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        # Normalize metrics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        scheduler.step()
        
        # Log to W&B
        logger.log_metrics({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc * 100,
            'val_loss': val_loss,
            'val_accuracy': val_acc * 100,
            'learning_rate': optimizer.param_groups[0]['lr']
        }, step=epoch)
        
        # Log confusion matrix every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.log_confusion_matrix(all_targets, all_preds, epoch)
        
        # Save best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            logger.log_model_checkpoint(
                model, optimizer, epoch, 
                {'val_accuracy': val_acc, 'val_loss': val_loss}, 
                is_best=True
            )
        
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train: {train_loss:.4f}/{train_acc:.3f} | "
              f"Val: {val_loss:.4f}/{val_acc:.3f}")
    
    return best_val_acc