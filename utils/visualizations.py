import matplotlib.pyplot as plt
import numpy as np
import os

def plot_training_curves(train_acc, val_acc, train_loss, val_loss, model_name, save_dir='plots'):
    """
    Plot and save training/validation curves.
    
    Args:
        train_acc: List of training accuracies
        val_acc: List of validation accuracies
        train_loss: List of training losses
        val_loss: List of validation losses
        model_name: Name of the model for plot titles and filenames
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(train_acc) + 1)
    
    # Set up the plotting style
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot accuracy curves
    ax1.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    ax1.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax1.set_title(f'{model_name} - Accuracy Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Add best accuracy annotations
    best_train_acc = max(train_acc)
    best_val_acc = max(val_acc)
    best_train_epoch = train_acc.index(best_train_acc) + 1
    best_val_epoch = val_acc.index(best_val_acc) + 1
    
    ax1.annotate(f'Best: {best_val_acc:.2f}%', 
                xy=(best_val_epoch, best_val_acc), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot loss curves
    ax2.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    ax2.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax2.set_title(f'{model_name} - Loss Curves', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add minimum loss annotation
    min_val_loss = min(val_loss)
    min_loss_epoch = val_loss.index(min_val_loss) + 1
    ax2.annotate(f'Min: {min_val_loss:.4f}', 
                xy=(min_loss_epoch, min_val_loss), 
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f'{model_name.lower()}_training_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Training curves saved to: {save_path}")
    return fig

def plot_model_comparison(models_history, save_dir='plots'):
    """
    Compare training curves of multiple models.
    
    Args:
        models_history: Dict with model names as keys and history dicts as values
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    plt.style.use('seaborn-v0_8')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (model_name, history) in enumerate(models_history.items()):
        epochs = range(1, len(history['val_acc']) + 1)
        color = colors[i % len(colors)]
        
        # Plot validation accuracy
        ax1.plot(epochs, history['val_acc'], color=color, 
                label=f'{model_name}', linewidth=2, marker='o', markersize=3)
        
        # Plot validation loss
        ax2.plot(epochs, history['val_loss'], color=color, 
                label=f'{model_name}', linewidth=2, marker='o', markersize=3)
    
    # Configure accuracy plot
    ax1.set_title('Model Comparison - Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 100])
    
    # Configure loss plot
    ax2.set_title('Model Comparison - Validation Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save comparison plot
    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 Model comparison saved to: {save_path}")
    return fig

def plot_learning_rate_schedule(lr_history, model_name, save_dir='plots'):
    """
    Plot learning rate schedule over epochs.
    
    Args:
        lr_history: List of learning rates per epoch
        model_name: Name of the model
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    epochs = range(1, len(lr_history) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lr_history, 'g-', linewidth=2)
    plt.title(f'{model_name} - Learning Rate Schedule', fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Learning Rate', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Save plot
    save_path = os.path.join(save_dir, f'{model_name.lower()}_lr_schedule.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"📊 LR schedule saved to: {save_path}")
    return plt.gcf()