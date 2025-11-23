"""
CNN Architecture for CIFAR-10 Classification
"""

import torch
import torch.nn as nn


class CIFAR10CNN(nn.Module):
    """
    Convolutional Neural Network for CIFAR-10
    
    Architecture:
        Input: [batch, 3, 32, 32]
        
        Conv Block 1: 3 → 32 channels
        Conv Block 2: 32 → 64 channels  
        Conv Block 3: 64 → 128 channels
        
        FC Layer 1: 2048 → 256
        FC Layer 2: 256 → 10
        
        Output: [batch, 10]
    """
    
    def __init__(self, num_classes=10):
        super(CIFAR10CNN, self).__init__()
        
        # ============================================
        # CONVOLUTIONAL LAYERS
        # ============================================
        
        # Block 1: Input channels=3 (RGB), Output channels=32
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        # Block 2: 32 → 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Block 3: 64 → 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # ============================================
        # POOLING & ACTIVATION
        # ============================================
        
        self.pool = nn.MaxPool2d(2, 2)  # Reduces size by half
        self.relu = nn.ReLU()
        
        # ============================================
        # FULLY CONNECTED LAYERS
        # ============================================
        
        # After 3 pooling: 32→16→8→4, so 128*4*4=2048
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor [batch, 3, 32, 32]
            
        Returns:
            Output tensor [batch, 10]
        """
        
        # Conv Block 1
        x = self.conv1(x)       # [batch, 32, 32, 32]
        x = self.bn1(x)         # Batch norm
        x = self.relu(x)        # Activation
        x = self.pool(x)        # [batch, 32, 16, 16]
        
        # Conv Block 2
        x = self.conv2(x)       # [batch, 64, 16, 16]
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)        # [batch, 64, 8, 8]
        
        # Conv Block 3
        x = self.conv3(x)       # [batch, 128, 8, 8]
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool(x)        # [batch, 128, 4, 4]
        
        # Flatten
        x = x.view(x.size(0), -1)  # [batch, 2048]
        
        # Fully connected layers
        x = self.fc1(x)         # [batch, 256]
        x = self.relu(x)
        x = self.dropout(x)     # Dropout
        x = self.fc2(x)         # [batch, 10]
        
        return x
    
    def count_parameters(self):
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# Test the model
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Testing CNN Model")
    print("="*50 + "\n")
    
    # Create model
    model = CIFAR10CNN()
    
    print(f" Model Architecture:")
    print(model)
    
    print(f"\n Total Parameters: {model.count_parameters():,}")
    
    # Test forward pass
    print(f"\n Testing forward pass...")
    dummy_input = torch.randn(4, 3, 32, 32)  # Batch of 4 images
    output = model(dummy_input)
    
    print(f"   Input shape:  {dummy_input.shape}")
    print(f"   Output shape: {output.shape}")
    
    print("\n Model test passed!\n")