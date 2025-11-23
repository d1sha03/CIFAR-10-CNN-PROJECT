"""
Training utilities
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import random
import numpy as np


def set_seed(seed=42):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f" Random seed set to {seed}")


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train model for one epoch
    
    Args:
        model: Neural network
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device (cpu/cuda)
        
    Returns:
        avg_loss, accuracy
    """
    
    model.train()  # Set to training mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Progress bar
    pbar = tqdm(train_loader, desc='Training', ncols=100)
    
    for inputs, labels in pbar:
        # Move to device
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


# Test training function
if __name__ == "__main__":
    from model import CIFAR10CNN
    from data_loader import get_cifar10_loaders
    
    print("\n" + "="*50)
    print("Testing Training Function")
    print("="*50 + "\n")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")
    
    set_seed(42)
    
    # Load data (small batch for quick test)
    train_loader, _ = get_cifar10_loaders(batch_size=128)
    
    # Create model
    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    # Train one epoch
    print(f"\n Training one epoch...\n")
    loss, acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    
    print(f"\n Training test passed!")
    print(f"   Loss: {loss:.4f}")
    print(f"   Accuracy: {acc:.2f}%\n")