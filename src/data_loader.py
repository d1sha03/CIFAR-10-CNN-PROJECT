"""
Data loading and preprocessing for CIFAR-10
"""

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms


def get_cifar10_loaders(batch_size=128, num_workers=2):
    """
    Load CIFAR-10 dataset with preprocessing
    
    Args:
        batch_size: Number of samples per batch
        num_workers: Number of subprocesses for data loading
        
    Returns:
        train_loader, test_loader
    """
    
    print(" Loading CIFAR-10 dataset...")
    
    # CIFAR-10 normalization values (pre-calculated)
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]
    
    # Training data transformations (with augmentation)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),      # Flip 50% of images
        transforms.RandomCrop(32, padding=4),   # Random crop
        transforms.ToTensor(),                  # Convert to tensor [0, 1]
        transforms.Normalize(mean, std)         # Normalize
    ])
    
    # Test data transformations (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Download and load training data
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )
    
    # Download and load test data
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f" Training samples: {len(train_dataset)}")
    print(f" Test samples: {len(test_dataset)}")
    print(f" Batch size: {batch_size}")
    
    return train_loader, test_loader


def get_class_names():
    """Return CIFAR-10 class names"""
    return ['airplane', 'automobile', 'bird', 'cat', 'deer', 
            'dog', 'frog', 'horse', 'ship', 'truck']


# Test the data loader
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Testing Data Loader")
    print("="*50 + "\n")
    
    train_loader, test_loader = get_cifar10_loaders(batch_size=4)
    
    # Get one batch
    images, labels = next(iter(train_loader))
    
    print(f"\n Batch Information:")
    print(f"   Images shape: {images.shape}")  # [4, 3, 32, 32]
    print(f"   Labels shape: {labels.shape}")  # [4]
    print(f"   Labels: {labels}")
    
    print("\n Data loader test passed!\n")