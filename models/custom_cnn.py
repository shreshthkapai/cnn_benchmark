import torch
import torch.nn as nn
import torch.nn.functional as F

class CustomCNN(nn.Module):
    """
    Custom CNN optimized for CIFAR-10 with ~3-4M parameters.
    Designed to compete with ResNet18 while being lightweight.
    """
    
    def __init__(self, num_classes=10, dropout_rate=0.4):
        super(CustomCNN, self).__init__()
        
        # Feature extractor with efficient blocks
        self.features = nn.Sequential(
            # Block 1: 32x32 -> 32x32
            self._conv_block(3, 64, stride=1),
            self._conv_block(64, 64, stride=1),
            
            # Block 2: 32x32 -> 16x16
            self._conv_block(64, 128, stride=2),
            self._conv_block(128, 128, stride=1),
            
            # Block 3: 16x16 -> 8x8
            self._conv_block(128, 256, stride=2),
            self._conv_block(256, 256, stride=1),
            
            # Block 4: 8x8 -> 4x4 (deep feature extraction)
            self._conv_block(256, 512, stride=2),
        )
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(512, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _conv_block(self, in_channels, out_channels, stride=1):
        """Efficient conv block with BatchNorm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def _initialize_weights(self):
        """Initialize weights using He initialization for ReLU."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x
    
    def get_model_info(self):
        """Return model architecture info."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming float32
            'architecture': '7-layer CNN with BatchNorm and Dropout'
        }

# Factory function for easy instantiation
def create_custom_cnn(num_classes=10, dropout_rate=0.4):
    """Create and return a CustomCNN instance."""
    return CustomCNN(num_classes=num_classes, dropout_rate=dropout_rate)
