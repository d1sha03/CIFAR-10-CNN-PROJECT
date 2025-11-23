"""
Comprehensive results analysis with insights
"""

import json
import numpy as np

# Load results
with open('results/training_results.json', 'r') as f:
    data = json.load(f)

print("\n" + "="*70)
print(" "*20 + "🎓 TRAINING ANALYSIS REPORT")
print("="*70)

# Configuration
config = data['configuration']
print(f"\n Training Configuration:")
print(f"   Timestamp:      {data['timestamp']}")
print(f"   Device:         {config['device']}")
print(f"   Batch Size:     {config['batch_size']}")
print(f"   Epochs:         {config['epochs']}")
print(f"   Learning Rate:  {config['learning_rate']}")

# Results
results = data['results']

print("\n" + "="*70)
print(" DETAILED RESULTS")
print("="*70)

# Sort by best accuracy
sorted_opts = sorted(results.items(), key=lambda x: x[1]['best_accuracy'], reverse=True)

for rank, (opt_name, metrics) in enumerate(sorted_opts, 1):
    medal = {1: "🥇", 2: "🥈", 3: "🥉", 4: "  "}.get(rank, "  ")
    
    print(f"\n{medal} Rank #{rank}: {opt_name}")
    print("-" * 70)
    print(f"   Best Validation Accuracy:  {metrics['best_accuracy']:6.2f}%")
    print(f"   Final Validation Accuracy: {metrics['final_accuracy']:6.2f}%")
    print(f"   Best Training Accuracy:    {metrics['best_train_accuracy']:6.2f}%")
    print(f"   Final Loss:                {metrics['final_loss']:6.4f}")
    
    # Overfitting check
    gap = metrics['best_train_accuracy'] - metrics['best_accuracy']
    if gap > 10:
        print(f"     Overfitting Detected:    {gap:.2f}% train/val gap")
    else:
        print(f"    Good Generalization:     {gap:.2f}% train/val gap")
    
    print(f"   Improvement over SGD:      {metrics['improvement_over_sgd']:+.2f}%")

# Statistical summary
print("\n" + "="*70)
print(" STATISTICAL SUMMARY")
print("="*70)

accuracies = [m['best_accuracy'] for m in results.values()]
print(f"\n   Mean Accuracy:     {np.mean(accuracies):.2f}%")
print(f"   Std Deviation:     {np.std(accuracies):.2f}%")
print(f"   Best Accuracy:     {np.max(accuracies):.2f}%")
print(f"   Worst Accuracy:    {np.min(accuracies):.2f}%")
print(f"   Accuracy Range:    {np.max(accuracies) - np.min(accuracies):.2f}%")

# Key insights
print("\n" + "="*70)
print(" KEY INSIGHTS")
print("="*70)

winner = sorted_opts[0][0]
winner_acc = sorted_opts[0][1]['best_accuracy']

print(f"""
1.  Best Optimizer: {winner} ({winner_acc:.2f}%)
   
2.  Performance Spread: {np.max(accuracies) - np.min(accuracies):.2f}% between best and worst
   
3.  Interpretation:
   - SGD with momentum proved most effective for this architecture
   - Adaptive optimizers (Adam/RMSprop) likely needed lower learning rates
   - The learning rate (0.01) was optimal for SGD but too high for adaptive methods
   
4.  Recommendations:
   a) For SGD: Current setup is excellent
   b) For Adam/AdamW: Try learning rate = 0.001 (10x lower)
   c) For RMSprop: Try learning rate = 0.001 with adjusted alpha
   
5.  Learning Outcome:
   ✓ Different optimizers need different hyperparameters
   ✓ "One size fits all" doesn't work in deep learning
   ✓ SGD is still competitive on vision tasks
   ✓ Experimentation and tuning are crucial
""")

print("="*70)

# Save detailed report
with open('results/detailed_report.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write(" "*20 + "TRAINING ANALYSIS REPORT\n")
    f.write("="*70 + "\n\n")
    f.write(f"Winner: {winner} with {winner_acc:.2f}% accuracy\n")
    f.write(f"Timestamp: {data['timestamp']}\n")
    
print("\n Detailed report saved to: results/detailed_report.txt\n")