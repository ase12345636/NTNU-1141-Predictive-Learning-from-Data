import os
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import multilabel_confusion_matrix

# Import from utils
from utils.dataset import nih_chest_dataset
from utils.model import load_lsnet_model
from utils.training import train_epoch, validate_epoch, test_model
from utils.visualization import plot_learning_curves, save_training_history, save_test_results, save_model_weights
from utils.gradcam import visualize_gradcam

NUM_CLASSES = 14
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = 'cuda'
TRAIN_RATIO = 0.8

# Create output directories
os.makedirs('results', exist_ok=True)
os.makedirs('checkpoints', exist_ok=True)
os.makedirs('gradcam', exist_ok=True)

#------------------------------------
# 1. Load NIH chest dataset
#------------------------------------
print("Loading dataset...")
X_train, y_train = nih_chest_dataset(split='train')
X_test, y_test = nih_chest_dataset(split='test')

train_size = int(TRAIN_RATIO * len(X_train))
indices = torch.randperm(len(X_train))
train_indices = indices[:train_size]
val_indices = indices[train_size:]

X_train_split = X_train[train_indices]
y_train_split = y_train[train_indices]
X_val = X_train[val_indices]
y_val = y_train[val_indices]

train_ds = TensorDataset(X_train_split, y_train_split)
val_ds = TensorDataset(X_val, y_val)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

# 2. Create LSNet model
print("Loading LSNet model...")
model = load_lsnet_model(num_classes=NUM_CLASSES, device=DEVICE)

# 3. Loss & Optimizer (for multilabel)
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 4. Training with validation
print("Starting training...")
history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    # Training
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
    # Validation
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, DEVICE)

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# Save results
save_model_weights(model, 'checkpoints/lsnet_final.pth')
plot_learning_curves(history, 'results/learning_curve.png')
save_training_history(history, 'results/training_history.json')

# Grad-CAM visualization
print("\nGenerating Grad-CAM visualizations...")
visualize_gradcam(model, val_ds, num_samples=5, device=DEVICE, save_dir='gradcam')

# 5. Testing and evaluation
print("\nTesting on test set...")
test_loss, test_acc, test_preds, test_labels, avg_ms = test_model(model, test_loader, criterion, DEVICE)
conf_matrix = multilabel_confusion_matrix(test_labels, test_preds)
model_size_mb = os.path.getsize('checkpoints/lsnet_final.pth') / (1024 ** 2)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
if avg_ms is not None:
    print(f"Avg Inference Speed: {avg_ms:.2f} ms/image")
print(f"Model Size: {model_size_mb:.2f} MB")

# Save test results
save_test_results(
    test_loss,
    test_acc,
    test_preds,
    test_labels,
    'results/test_results.json',
    confusion_matrix=conf_matrix,
    avg_inference_ms=avg_ms,
    model_size_mb=model_size_mb,
)

print("\nAll results saved to results/ directory")
