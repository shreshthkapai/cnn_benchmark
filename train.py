import torch
import torch.nn as nn
import time
from collections import defaultdict
from wandb_utils import WandbLogger
import argparse

def train(model, train_loader, val_loader, optimizer, criterion, scheduler, 
          num_epochs, device='cuda', use_wandb=True, model_name="CustomCNN"):
    """
    Production-grade training pipeline with optional W&B integration.
    
    Args:
        model: PyTorch model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer (AdamW/SGD)
        criterion: Loss function (CrossEntropyLoss)
        scheduler: Learning rate scheduler
        num_epochs: Number of training epochs
        device: Device to train on
        use_wandb: Whether to use W&B logging
        model_name: Name of the model for logging
        
    Returns:
        dict: Training history with losses and accuracies
    """
    model.to(device)
    best_val_acc = 0.0
    history = defaultdict(list)
    
    # Initialize W&B if enabled
    logger = None
    if use_wandb:
        logger = WandbLogger()
        config = {
            'epochs': num_epochs,
            'device': str(device),
            'optimizer': optimizer.__class__.__name__,
            'scheduler': scheduler.__class__.__name__,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        logger.init_experiment(config, model, model_name)
    
    print(f"🚀 Training {model_name} on {device} for {num_epochs} epochs...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        train_loss, train_acc = _train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validation phase
        model.eval()
        val_loss, val_acc, val_preds, val_targets = _validate_epoch(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        metrics = {
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr']
        }
        
        for key, value in metrics.items():
            history[key].append(value)
        
        # W&B logging
        if logger:
            logger.log_metrics({
                'epoch': epoch,
                'train_loss': train_loss,
                'train_accuracy': train_acc,
                'val_loss': val_loss,
                'val_accuracy': val_acc,
                'learning_rate': optimizer.param_groups[0]['lr']
            }, step=epoch)
            
            # Log confusion matrix every 20 epochs
            if (epoch + 1) % 20 == 0:
                logger.log_confusion_matrix(val_targets, val_preds, epoch)
        
        # Save best model
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss
            }
            torch.save(checkpoint, f'best_model_{model_name.lower()}.pth')
            
            # Only log best model checkpoint to W&B
            if logger:
                logger.log_model_checkpoint(model, optimizer, epoch, 
                                          {'val_accuracy': val_acc, 'val_loss': val_loss}, 
                                          is_best=True)
        
        # Print progress
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch+1:3d}/{num_epochs} | "
              f"Train: {train_loss:.4f}/{train_acc:.2f}% | "
              f"Val: {val_loss:.4f}/{val_acc:.2f}% | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {epoch_time:.1f}s")
    
    if logger:
        logger.finish()
    
    print(f"\n🎯 Best validation accuracy: {best_val_acc:.2f}%")
    return dict(history)

def _train_epoch(model, train_loader, optimizer, criterion, device):
    """Single training epoch with metrics."""
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    
    return running_loss / len(train_loader), 100.0 * correct / total

def _validate_epoch(model, val_loader, criterion, device):
    """Single validation epoch with predictions."""
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    return (running_loss / len(val_loader), 100.0 * correct / total, 
            all_preds, all_targets)

def create_optimizer(model, opt_type='adamw', lr=0.001, weight_decay=1e-4):
    """Create optimizer with best practices."""
    if opt_type.lower() == 'adamw':
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type.lower() == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_type}")

def create_scheduler(optimizer, scheduler_type='cosine', num_epochs=50):  # Changed default from 100 to 50
    """Create learning rate scheduler."""
    if scheduler_type.lower() == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type.lower() == 'step':
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")

# CLI interface for hyperparameter sweeps
def main():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 models with W&B')
    parser.add_argument('--model', choices=['custom', 'resnet18'], default='custom')
    parser.add_argument('--epochs', type=int, default=50)  # Changed from 100 to 50
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--sweep', action='store_true', help='Run hyperparameter sweep')
    
    args = parser.parse_args()
    
    # Import models
    if args.model == 'custom':
        from models.custom_cnn import create_custom_cnn
        model = create_custom_cnn()
        model_name = "CustomCNN"
    else:
        from models.resnet18 import load_resnet18
        model = load_resnet18()
        model_name = "ResNet18"
    
    # Load data
    from utils.data_loader import get_cifar10_loaders
    train_loader, val_loader, test_loader = get_cifar10_loaders(batch_size=args.batch_size)
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, lr=args.lr)
    scheduler = create_scheduler(optimizer, num_epochs=args.epochs)
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    if args.sweep:
        from wandb_utils import create_hyperparameter_sweep, run_hyperparameter_sweep
        sweep_config = create_hyperparameter_sweep()
        
        def train_fn():
            # W&B will set hyperparameters
            train(model, train_loader, val_loader, optimizer, criterion, 
                  scheduler, args.epochs, model_name=model_name)
        
        run_hyperparameter_sweep(train_fn, sweep_config)
    else:
        train(model, train_loader, val_loader, optimizer, criterion, scheduler, 
              args.epochs, use_wandb=not args.no_wandb, model_name=model_name)

if __name__ == "__main__":
    main()