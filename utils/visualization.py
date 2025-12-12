import torch
import matplotlib.pyplot as plt
import json
import numpy as np

def plot_learning_curves(history, save_path='results/learning_curve.png'):
    """Plot and save learning curves"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o')
    plt.plot(history['val_loss'], label='Val Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc', marker='o')
    plt.plot(history['val_acc'], label='Val Acc', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Learning curve saved to {save_path}")

def save_training_history(history, save_path='results/training_history.json'):
    """Save training history to JSON"""
    with open(save_path, 'w') as f:
        json.dump(history, f)
    print(f"Training history saved to {save_path}")

def save_test_results(
    test_loss,
    test_acc,
    test_preds,
    test_labels,
    save_path='results/test_results.json',
    confusion_matrix=None,
    avg_inference_ms=None,
    model_size_mb=None,
):
    """Save test results to JSON, optionally with confusion matrix and speed/size metrics."""
    results = {
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'predictions': test_preds.tolist(),
        'labels': test_labels.tolist(),
    }

    if confusion_matrix is not None:
        results['confusion_matrix'] = confusion_matrix.tolist()
    if avg_inference_ms is not None:
        results['avg_inference_ms_per_image'] = float(avg_inference_ms)
    if model_size_mb is not None:
        results['model_size_mb'] = float(model_size_mb)
    
    with open(save_path, 'w') as f:
        json.dump(results, f)
    print(f"Test results saved to {save_path}")

def save_model_weights(model, save_path='checkpoints/lsnet_final.pth'):
    """Save model weights"""
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def plot_learning_curves_f1(history, save_path='results/learning_curve_f1.png'):
    """Plot and save learning curves with F1 scores"""
    plt.figure(figsize=(18, 5))
    
    # Loss plot
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    plt.plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Loss Curve', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Macro F1 plot
    plt.subplot(1, 3, 2)
    plt.plot(history['train_f1_macro'], label='Train Macro F1', marker='o', linewidth=2, color='green')
    plt.plot(history['val_f1_macro'], label='Val Macro F1', marker='s', linewidth=2, color='orange')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Macro F1 Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Macro F1 Score', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    # Weighted F1 plot
    plt.subplot(1, 3, 3)
    plt.plot(history['train_f1_weighted'], label='Train Weighted F1', marker='o', linewidth=2, color='blue')
    plt.plot(history['val_f1_weighted'], label='Val Weighted F1', marker='s', linewidth=2, color='red')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Weighted F1 Score', fontsize=12)
    plt.legend(fontsize=10)
    plt.title('Weighted F1 Score', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"F1 learning curves saved to {save_path}")

def plot_learning_curves_combined(history, save_path='results/learning_curves_combined.png'):
    """Plot and save learning curves with both accuracy and F1 scores"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss plot
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss', marker='o', linewidth=2)
    ax.plot(history['val_loss'], label='Val Loss', marker='s', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title('Loss Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax = axes[0, 1]
    ax.plot(history['train_acc'], label='Train Accuracy', marker='o', linewidth=2, color='blue')
    ax.plot(history['val_acc'], label='Val Accuracy', marker='s', linewidth=2, color='red')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title('Accuracy Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Macro F1 plot
    ax = axes[1, 0]
    ax.plot(history['train_f1_macro'], label='Train Macro F1', marker='o', linewidth=2, color='green')
    ax.plot(history['val_f1_macro'], label='Val Macro F1', marker='s', linewidth=2, color='orange')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Macro F1 Score', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title('Macro F1 Score', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Weighted F1 plot
    ax = axes[1, 1]
    ax.plot(history['train_f1_weighted'], label='Train Weighted F1', marker='o', linewidth=2, color='purple')
    ax.plot(history['val_f1_weighted'], label='Val Weighted F1', marker='s', linewidth=2, color='brown')
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Weighted F1 Score', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title('Weighted F1 Score', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Combined learning curves saved to {save_path}")
