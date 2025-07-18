import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import confusion_matrix, classification_report
from utils.data_loader import get_cifar10_info

def evaluate(model, test_loader, device='cuda'):
    """
    Evaluate model on test set with comprehensive metrics.
    
    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        dict: Evaluation results including accuracy, loss, predictions
    """
    model.to(device)
    model.eval()
    
    test_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    test_acc = 100.0 * correct / total
    test_loss /= len(test_loader)
    
    return {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'predictions': np.array(all_predictions),
        'targets': np.array(all_targets)
    }

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save plot (optional)
    """
    cifar10_info = get_cifar10_info()
    class_names = cifar10_info['class_names']
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    return cm

def benchmark_model(model, test_loader, device='cuda', num_samples=100):
    """
    Benchmark model performance metrics.
    
    Args:
        model: PyTorch model to benchmark
        test_loader: Test data loader
        device: Device to run benchmark on
        num_samples: Number of samples for inference timing
        
    Returns:
        dict: Benchmark results
    """
    model.to(device)
    model.eval()
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size_mb = total_params * 4 / (1024 * 1024)  # Float32
    
    # Inference timing
    sample_inputs = []
    sample_count = 0
    
    for inputs, _ in test_loader:
        if sample_count >= num_samples:
            break
        batch_size = min(num_samples - sample_count, inputs.size(0))
        sample_inputs.append(inputs[:batch_size])
        sample_count += batch_size
    
    test_input = torch.cat(sample_inputs, dim=0)[:num_samples].to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_input)
    
    # Timing runs
    torch.cuda.synchronize() if device == 'cuda' else None
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(10):
            _ = model(test_input)
    
    torch.cuda.synchronize() if device == 'cuda' else None
    end_time = time.time()
    
    avg_inference_time = (end_time - start_time) / 10 / num_samples * 1000  # ms per sample
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb,
        'inference_time_ms': avg_inference_time,
        'throughput_samples_per_sec': 1000 / avg_inference_time
    }

def print_evaluation_report(eval_results, benchmark_results, model_name):
    """Print comprehensive evaluation report."""
    print(f"\n{'='*50}")
    print(f"EVALUATION REPORT: {model_name}")
    print(f"{'='*50}")
    
    print(f"📊 Test Performance:")
    print(f"  Accuracy: {eval_results['test_accuracy']:.2f}%")
    print(f"  Loss: {eval_results['test_loss']:.4f}")
    
    print(f"\n🔧 Model Statistics:")
    print(f"  Parameters: {benchmark_results['total_params']:,}")
    print(f"  Model Size: {benchmark_results['model_size_mb']:.2f} MB")
    print(f"  Inference Time: {benchmark_results['inference_time_ms']:.2f} ms/sample")
    print(f"  Throughput: {benchmark_results['throughput_samples_per_sec']:.0f} samples/sec")
    
    # Classification report
    cifar10_info = get_cifar10_info()
    class_names = cifar10_info['class_names']
    
    print(f"\n📈 Per-Class Performance:")
    print(classification_report(eval_results['targets'], eval_results['predictions'], 
                              target_names=class_names, digits=3))