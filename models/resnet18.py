import torch
import torch.nn as nn
from torchvision import models

def load_resnet18(num_classes=10, pretrained=False):
    """
    Load ResNet18 modified for CIFAR-10 classification.
    
    Args:
        num_classes: Number of output classes (default: 10 for CIFAR-10)
        pretrained: Whether to use ImageNet pretrained weights (default: False for fair comparison)
        
    Returns:
        Modified ResNet18 model
    """
    # Load ResNet18 without pretrained weights for fair comparison
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    
    # Replace final layer for CIFAR-10 (10 classes)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # Initialize the new classifier layer properly
    nn.init.normal_(model.fc.weight, 0, 0.01)
    nn.init.constant_(model.fc.bias, 0)
    
    return model

def get_resnet18_info():
    """Return ResNet18 model information."""
    model = load_resnet18()
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
        'architecture': 'ResNet18 with modified classifier',
        'original_fc_features': 512,
        'modified_fc_classes': 10
    }

def freeze_backbone(model, freeze=True):
    """
    Freeze/unfreeze ResNet18 backbone for transfer learning experiments.
    
    Args:
        model: ResNet18 model
        freeze: Whether to freeze backbone parameters
    """
    for name, param in model.named_parameters():
        if 'fc' not in name:  # Don't freeze the final classifier
            param.requires_grad = not freeze
    
    return model