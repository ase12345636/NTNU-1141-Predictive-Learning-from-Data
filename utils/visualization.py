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

def save_test_results(test_loss, test_acc, test_preds, test_labels, save_path='results/test_results.json'):
    """Save test results to JSON"""
    results = {
        'test_loss': float(test_loss),
        'test_acc': float(test_acc),
        'predictions': test_preds.tolist(),
        'labels': test_labels.tolist()
    }
    
    with open(save_path, 'w') as f:
        json.dump(results, f)
    print(f"Test results saved to {save_path}")

def save_model_weights(model, save_path='checkpoints/lsnet_final.pth'):
    """Save model weights"""
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
