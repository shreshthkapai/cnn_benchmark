#!/usr/bin/env python3
"""
CIFAR-10 CNN Benchmark - Production Pipeline
FAANG-level deep learning experiment orchestration
"""

import torch
import torch.nn as nn
from pathlib import Path
import argparse
import json
from datetime import datetime

# Local imports
from models.custom_cnn import create_custom_cnn
from models.resnet18 import load_resnet18
from utils.data_loader import get_cifar10_loaders
from train import train, create_optimizer, create_scheduler
from eval import evaluate, benchmark_model, print_evaluation_report, plot_confusion_matrix
from wandb_utils import WandbLogger, create_hyperparameter_sweep, run_hyperparameter_sweep

def run_single_experiment(model_name: str, config: dict, use_wandb: bool = True):
    """Run a single training experiment with comprehensive evaluation."""
    print(f"\n🚀 Starting {model_name} experiment...")
    
    # Initialize model
    if model_name.lower() == 'custom':
        model = create_custom_cnn()
        model_display_name = "custom"
    elif model_name.lower() == 'resnet18':
        model = load_resnet18()
        model_display_name = "ResNet18"
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Load data
    train_loader, val_loader, test_loader = get_cifar10_loaders(
        batch_size=config['batch_size'],
        num_workers=config.get('num_workers', 4)
    )
    
    # Setup training components
    optimizer = create_optimizer(
        model, 
        opt_type=config['optimizer'],
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    scheduler = create_scheduler(
        optimizer, 
        scheduler_type=config['scheduler'],
        num_epochs=config['epochs']
    )
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    history = train(
        model, train_loader, val_loader, optimizer, criterion, scheduler,
        num_epochs=config['epochs'], device=device,
        use_wandb=use_wandb, model_name=model_display_name
    )
    
    # Comprehensive evaluation
    print(f"\n📊 Evaluating {model_display_name}...")
    
    # Load best model
    checkpoint = torch.load(f'best_model_{model_name.lower()}.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    eval_results = evaluate(model, test_loader, device)
    benchmark_results = benchmark_model(model, test_loader, device)
    
    # Print comprehensive report
    print_evaluation_report(eval_results, benchmark_results, model_display_name)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        eval_results['targets'], 
        eval_results['predictions'],
        save_path=f'confusion_matrix_{model_name.lower()}.png'
    )
    
    # Save experiment results
    results = {
        'model': model_display_name,
        'config': config,
        'timestamp': datetime.now().isoformat(),
        'evaluation': {
            'test_accuracy': eval_results['test_accuracy'],
            'test_loss': eval_results['test_loss']
        },
        'benchmark': benchmark_results,
        'training_history': history
    }
    
    with open(f'results_{model_name.lower()}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    return results

def run_model_comparison(config: dict, use_wandb: bool = True):
    """Run comparative analysis between CustomCNN and ResNet18."""
    print("\n🔬 Starting Model Comparison Analysis...")
    
    models = ['custom', 'resnet18']
    all_results = {}
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"Training {model_name.upper()}")
        print(f"{'='*60}")
        
        results = run_single_experiment(model_name, config, use_wandb)
        all_results[model_name] = results
    
    # Generate comparison report
    print(f"\n{'='*60}")
    print("📈 MODEL COMPARISON SUMMARY")
    print(f"{'='*60}")
    
    for model_name, results in all_results.items():
        eval_results = results['evaluation']
        benchmark_results = results['benchmark']
        
        print(f"\n{results['model']}:")
        print(f"  Test Accuracy: {eval_results['test_accuracy']:.2f}%")
        print(f"  Parameters: {benchmark_results['total_params']:,}")
        print(f"  Model Size: {benchmark_results['model_size_mb']:.2f} MB")
        print(f"  Inference Time: {benchmark_results['inference_time_ms']:.2f} ms")
        print(f"  Throughput: {benchmark_results['throughput_samples_per_sec']:.0f} samples/sec")
    
    # Determine winner
    custom_acc = all_results['custom']['evaluation']['test_accuracy']
    resnet_acc = all_results['resnet18']['evaluation']['test_accuracy']
    
    if custom_acc > resnet_acc:
        print(f"\n🏆 Winner: CustomCNN ({custom_acc:.2f}% vs {resnet_acc:.2f}%)")
    else:
        print(f"\n🏆 Winner: ResNet18 ({resnet_acc:.2f}% vs {custom_acc:.2f}%)")
    
    # Save comparison results
    with open('model_comparison.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    return all_results

def run_hyperparameter_sweep(model_name: str, count: int = 20):
    """Run automated hyperparameter optimization."""
    print(f"\n🎯 Starting Hyperparameter Sweep for {model_name}...")
    
    sweep_config = create_hyperparameter_sweep()
    
    def train_fn():
        import wandb
        config = wandb.config
        
        # Convert wandb config to dict
        config_dict = {
            'learning_rate': config.learning_rate,
            'batch_size': config.batch_size,
            'weight_decay': config.weight_decay,
            'optimizer': config.optimizer,
            'scheduler': config.scheduler,
            'epochs': 50  # Changed from 50 to 50
        }
        
        run_single_experiment(model_name, config_dict, use_wandb=True)
    
    run_hyperparameter_sweep(train_fn, sweep_config, count=count)

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN Benchmark Pipeline')
    parser.add_argument('--mode', choices=['single', 'compare', 'sweep'], default='compare',
                       help='Experiment mode')
    parser.add_argument('--model', choices=['custom', 'resnet18'], default='custom',
                       help='Model for single experiment')
    parser.add_argument('--no-wandb', action='store_true', help='Disable W&B logging')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')  # Changed from 100 to 50
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--sweep_count', type=int, default=20, help='Number of sweep runs')
    
    args = parser.parse_args()
    
    # Default configuration
    config = {
        'epochs': args.epochs,
        'learning_rate': args.lr,
        'batch_size': args.batch_size,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'scheduler': 'cosine',
        'num_workers': 4
    }
    
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    # Run experiment based on mode
    if args.mode == 'single':
        run_single_experiment(args.model, config, use_wandb=not args.no_wandb)
    elif args.mode == 'compare':
        run_model_comparison(config, use_wandb=not args.no_wandb)
    elif args.mode == 'sweep':
        run_hyperparameter_sweep(args.model, count=args.sweep_count)
    
    print("\n✅ Experiment completed! Check outputs/ directory for results.")

if __name__ == "__main__":
    main()