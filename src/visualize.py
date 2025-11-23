"""
Visualization utilities
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import os


def plot_learning_curves(histories, optimizer_names, save_path='results/learning_curves.png'):
    """Plot training and validation curves"""
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    for history, name in zip(histories, optimizer_names):
        axes[0].plot(history['train_loss'], label=f'{name} (Train)', linestyle='--', linewidth=2)
        axes[0].plot(history['val_loss'], label=f'{name} (Val)', linewidth=2)
    
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy curves
    for history, name in zip(histories, optimizer_names):
        axes[1].plot(history['train_acc'], label=f'{name} (Train)', linestyle='--', linewidth=2)
        axes[1].plot(history['val_acc'], label=f'{name} (Val)', linewidth=2)
    
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved learning curves to {save_path}")
    plt.close()


def plot_confusion_matrix(true_labels, predictions, optimizer_name, save_path=None):
    """Plot confusion matrix"""
    
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']
    
    cm = confusion_matrix(true_labels, predictions)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes,
                cbar_kws={'label': 'Count'})
    
    plt.title(f'Confusion Matrix - {optimizer_name}', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path is None:
        save_path = f'results/confusion_matrix_{optimizer_name}.png'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved confusion matrix to {save_path}")
    plt.close()


def plot_comparison_table(results, save_path='results/comparison_table.png'):
    """Create comparison table"""
    
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = []
    for opt_name, metrics in results.items():
        table_data.append([
            opt_name,
            f"{metrics['final_acc']:.2f}%",
            f"{metrics['best_train_acc']:.2f}%",
            f"{metrics['final_loss']:.4f}",
            f"{metrics.get('improvement', 0):+.2f}%"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Optimizer', 'Val Accuracy', 'Train Accuracy', 'Final Loss', 'Improvement'],
        cellLoc='center',
        loc='center',
        colWidths=[0.15, 0.2, 0.2, 0.2, 0.2]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    for i in range(5):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data) + 1):
        for j in range(5):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Optimizer Comparison Results', fontsize=16, fontweight='bold', pad=20)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved comparison table to {save_path}")
    plt.close()


# Test visualizations
if __name__ == "__main__":
    print("\n" + "="*50)
    print("Testing Visualization Functions")
    print("="*50 + "\n")
    
    # Dummy data
    dummy_history = {
        'train_loss': list(np.random.rand(10)),
        'val_loss': list(np.random.rand(10)),
        'train_acc': list(np.random.rand(10) * 100),
        'val_acc': list(np.random.rand(10) * 100)
    }
    
    dummy_labels = np.random.randint(0, 10, 1000)
    dummy_preds = np.random.randint(0, 10, 1000)
    
    dummy_results = {
        'SGD': {'final_acc': 72.5, 'best_train_acc': 85.0, 'final_loss': 0.85, 'improvement': 0.0},
        'Adam': {'final_acc': 78.2, 'best_train_acc': 88.5, 'final_loss': 0.65, 'improvement': 5.7}
    }
    
    # Test plots
    print("Creating test plots...")
    plot_learning_curves([dummy_history], ['Test'])
    plot_confusion_matrix(dummy_labels, dummy_preds, 'Test')
    plot_comparison_table(dummy_results)
    
    print("\n All visualization tests passed!\n")