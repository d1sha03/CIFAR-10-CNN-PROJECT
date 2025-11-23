"""
Evaluation utilities
"""

import torch
from tqdm import tqdm


def evaluate(model, test_loader, criterion, device):
    """
    Evaluate model on test set
    
    Args:
        model: Neural network
        test_loader: Test data loader
        criterion: Loss function
        device: Device (cpu/cuda)
        
    Returns:
        avg_loss, accuracy, predictions, labels
    """
    
    model.eval()  # Set to evaluation mode
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    all_predictions = []
    all_labels = []
    
    # Disable gradient computation
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating', ncols=100)
        
        for inputs, labels in pbar:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Store for confusion matrix
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.3f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
    
    avg_loss = running_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy, all_predictions, all_labels


# Test evaluation
if __name__ == "__main__":
    from model import CIFAR10CNN
    from data_loader import get_cifar10_loaders
    import torch.nn as nn
    
    print("\n" + "="*50)
    print("Testing Evaluation Function")
    print("="*50 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}\n")
    
    # Load test data
    _, test_loader = get_cifar10_loaders(batch_size=128)
    
    # Create model
    model = CIFAR10CNN().to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Evaluate
    loss, acc, preds, labels = evaluate(model, test_loader, criterion, device)
    
    print(f"\n Evaluation test passed!")
    print(f"   Loss: {loss:.4f}")
    print(f"   Accuracy: {acc:.2f}%")
    print(f"   Predictions: {len(preds)}\n")