"""
CIFAR-10 CNN Training with Multiple Optimizers
Main training script
"""

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import json
import os

# Import our custom modules
from src.data_loader import get_cifar10_loaders
from src.model import CIFAR10CNN
from src.train import train_one_epoch, set_seed
from src.evaluate import evaluate
from src.visualize import plot_learning_curves, plot_confusion_matrix, plot_comparison_table


def train_with_optimizer(model, train_loader, test_loader, optimizer, 
                         criterion, device, num_epochs, optimizer_name):
    """
    Complete training loop for one optimizer
    
    Args:
        model: CNN model
        train_loader: Training data loader
        test_loader: Test data loader
        optimizer: Optimizer instance
        criterion: Loss function
        device: Device (cpu/cuda)
        num_epochs: Number of epochs
        optimizer_name: Name for logging
        
    Returns:
        Dictionary with training history and final metrics
    """
    
    print(f"\n{'='*60}")
    print(f"🚀 Training with {optimizer_name}")
    print(f"{'='*60}\n")
    
    # History tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_acc = 0.0
    best_epoch = 0
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        print(f"\n📅 Epoch {epoch}/{num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Evaluate
        val_loss, val_acc, predictions, labels = evaluate(
            model, test_loader, criterion, device
        )
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch} Summary:")
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            
            # Save model
            save_path = f'models/best_{optimizer_name}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy': val_acc,
                'loss': val_loss,
            }, save_path)
            
            print(f"   ⭐ New best model saved! (Acc: {val_acc:.2f}%)")
    
    print(f"\n✅ Training completed for {optimizer_name}")
    print(f"   Best Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
    
    # Return final predictions for confusion matrix
    return history, predictions, labels, best_acc


def main():
    """Main training pipeline"""
    
    print("\n" + "="*60)
    print(" CIFAR-10 CNN Training Pipeline")
    print("="*60)
    
    # ====================================
    # 1. SETUP
    # ====================================
    
    print("\n Configuration:")
    
    # Set random seed for reproducibility
    set_seed(42)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   Device: {device}")
    
    # Hyperparameters
    BATCH_SIZE = 128
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.01
    
    print(f"   Batch Size: {BATCH_SIZE}")
    print(f"   Epochs: {NUM_EPOCHS}")
    print(f"   Learning Rate: {LEARNING_RATE}")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # ====================================
    # 2. LOAD DATA
    # ====================================
    
    print("\n" + "-"*60)
    train_loader, test_loader = get_cifar10_loaders(
        batch_size=BATCH_SIZE,
        num_workers=2
    )
    
    # ====================================
    # 3. DEFINE OPTIMIZERS TO COMPARE
    # ====================================
    
    print("\n" + "-"*60)
    print("\n Optimizers to Compare:")
    
    optimizers_config = {
        'SGD': lambda params: optim.SGD(params, lr=LEARNING_RATE, momentum=0.9, weight_decay=5e-4),
        'Adam': lambda params: optim.Adam(params, lr=LEARNING_RATE, weight_decay=5e-4),
        'RMSprop': lambda params: optim.RMSprop(params, lr=LEARNING_RATE, weight_decay=5e-4),
        'AdamW': lambda params: optim.AdamW(params, lr=LEARNING_RATE, weight_decay=5e-4)
    }
    
    for i, name in enumerate(optimizers_config.keys(), 1):
        print(f"   {i}. {name}")
    
    # ====================================
    # 4. TRAIN WITH EACH OPTIMIZER
    # ====================================
    
    all_histories = {}
    all_results = {}
    
    criterion = nn.CrossEntropyLoss()
    
    for opt_name, opt_func in optimizers_config.items():
        
        # Create fresh model for each optimizer
        model = CIFAR10CNN().to(device)
        
        print(f"\n Model Parameters: {model.count_parameters():,}")
        
        # Create optimizer
        optimizer = opt_func(model.parameters())
        
        # Train
        history, predictions, labels, best_acc = train_with_optimizer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            num_epochs=NUM_EPOCHS,
            optimizer_name=opt_name
        )
        
        # Store results
        all_histories[opt_name] = history
        all_results[opt_name] = {
            'final_acc': history['val_acc'][-1],
            'best_acc': best_acc,
            'best_train_acc': max(history['train_acc']),
            'final_loss': history['val_loss'][-1],
            'predictions': predictions,
            'labels': labels
        }
    
    # ====================================
    # 5. CALCULATE IMPROVEMENTS
    # ====================================
    
    print("\n" + "="*60)
    print(" Calculating Improvements")
    print("="*60 + "\n")
    
    # Use SGD as baseline
    baseline_acc = all_results['SGD']['final_acc']
    
    for opt_name in all_results.keys():
        improvement = all_results[opt_name]['final_acc'] - baseline_acc
        all_results[opt_name]['improvement'] = improvement
        
        print(f"{opt_name:12} | "
              f"Final Acc: {all_results[opt_name]['final_acc']:6.2f}% | "
              f"Best Acc: {all_results[opt_name]['best_acc']:6.2f}% | "
              f"Improvement: {improvement:+.2f}%")
    
    # ====================================
    # 6. GENERATE VISUALIZATIONS
    # ====================================
    
    print("\n" + "="*60)
    print(" Generating Visualizations")
    print("="*60 + "\n")
    
    # Learning curves
    plot_learning_curves(
        histories=[all_histories[name] for name in optimizers_config.keys()],
        optimizer_names=list(optimizers_config.keys()),
        save_path='results/learning_curves.png'
    )
    
    # Confusion matrices
    for opt_name in optimizers_config.keys():
        plot_confusion_matrix(
            true_labels=all_results[opt_name]['labels'],
            predictions=all_results[opt_name]['predictions'],
            optimizer_name=opt_name,
            save_path=f'results/confusion_matrix_{opt_name}.png'
        )
    
    # Comparison table
    plot_comparison_table(
        results=all_results,
        save_path='results/comparison_table.png'
    )
    
    # ====================================
    # 7. SAVE RESULTS TO JSON
    # ====================================
    
    print("\n" + "="*60)
    print(" Saving Results")
    print("="*60 + "\n")
    
    # Prepare results for JSON (remove numpy arrays)
    json_results = {}
    for opt_name, metrics in all_results.items():
        json_results[opt_name] = {
            'final_accuracy': metrics['final_acc'],
            'best_accuracy': metrics['best_acc'],
            'best_train_accuracy': metrics['best_train_acc'],
            'final_loss': metrics['final_loss'],
            'improvement_over_sgd': metrics['improvement']
        }
    
    # Add metadata
    results_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'configuration': {
            'batch_size': BATCH_SIZE,
            'epochs': NUM_EPOCHS,
            'learning_rate': LEARNING_RATE,
            'device': str(device)
        },
        'results': json_results
    }
    
    # Save to JSON
    with open('results/training_results.json', 'w') as f:
        json.dump(results_data, f, indent=4)
    
    print("✓ Results saved to results/training_results.json")
    
    # ====================================
    # 8. FINAL SUMMARY
    # ====================================
    
    print("\n" + "="*60)
    print(" TRAINING COMPLETE!")
    print("="*60)
    
    print("\n Generated Files:")
    print("   Models:")
    for opt_name in optimizers_config.keys():
        print(f"      - models/best_{opt_name}.pth")
    
    print("\n   Visualizations:")
    print("      - results/learning_curves.png")
    print("      - results/comparison_table.png")
    for opt_name in optimizers_config.keys():
        print(f"      - results/confusion_matrix_{opt_name}.png")
    
    print("\n   Data:")
    print("      - results/training_results.json")
    
    print("\n Best Optimizer:", max(all_results.items(), key=lambda x: x[1]['best_acc'])[0])
    print(f"   Accuracy: {max(r['best_acc'] for r in all_results.values()):.2f}%")
    
    print("\n" + "="*60 + "\n")


if __name__ == "__main__":
    main()