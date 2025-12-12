import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import multilabel_confusion_matrix

# Import from utils
from utils.dataset import nih_chest_dataset
from utils.model import load_lsnet_model
from utils.training_combined import train_epoch_combined, validate_epoch_combined, test_model_combined
from utils.visualization import plot_learning_curves_combined, save_training_history, save_test_results, save_model_weights
from utils.visualization_advanced import plot_confusion_matrix_heatmap, plot_true_confusion_matrix, plot_per_class_statistics
from utils.gradcam import visualize_gradcam
from utils.gradcam_advanced import generate_gradcam_per_disease

NUM_CLASSES = 14
BATCH_SIZE = 32
EPOCHS = 40
DEVICE = 'cuda'
TRAIN_RATIO = 0.8
# Loss selection: 'bce', 'weighted', or 'focal'
LOSS_TYPE = 'weighted'
FOCAL_GAMMA = float(os.getenv('FOCAL_GAMMA', '2.0'))

# Create output directories
os.makedirs('results2', exist_ok=True)
os.makedirs('checkpoints2', exist_ok=True)
os.makedirs('results2/gradcam', exist_ok=True)

#------------------------------------
# 1. Load NIH chest dataset
#------------------------------------
print("Loading dataset...")
full_train_ds = nih_chest_dataset(split='train')
test_ds = nih_chest_dataset(split='test')

train_size = int(TRAIN_RATIO * len(full_train_ds))
val_size = len(full_train_ds) - train_size
train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                         num_workers=4, pin_memory=True, persistent_workers=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                       num_workers=4, pin_memory=True, persistent_workers=True)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=4, pin_memory=True, persistent_workers=True)
print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')

# 2. Create LSNet model
print("Loading LSNet model...")
model = load_lsnet_model(num_classes=NUM_CLASSES, device=DEVICE)

# Mixed precision scaler (enabled on CUDA)
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.startswith('cuda'))

# 3. Loss & Optimizer (for multilabel)

# Helper: compute per-class positive weights from the training subset
def compute_pos_weight(subset, num_classes=NUM_CLASSES):
    ds = subset.dataset
    idxs = subset.indices if hasattr(subset, 'indices') else range(len(ds.samples))
    pos = np.zeros(num_classes, dtype=np.float64)
    for i in idxs:
        pos += ds.samples[i][1]
    n = float(len(idxs))
    pos = np.clip(pos, 1.0, None)  # avoid division by zero
    neg = n - pos
    return torch.tensor(neg / pos, dtype=torch.float32)

pos_weight = compute_pos_weight(train_ds).to(DEVICE)

if LOSS_TYPE == 'focal':
    from utils.focal_loss import FocalLoss
    # Normalize alpha to [0,1] for stability (optional but helpful)
    alpha = (pos_weight / pos_weight.max()).to(DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA, reduction='mean')
    print(f"Using Focal Loss (gamma={FOCAL_GAMMA})")
elif LOSS_TYPE == 'weighted':
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("Using Weighted BCEWithLogitsLoss with computed pos_weight")
else:
    criterion = torch.nn.BCEWithLogitsLoss()
    print("Using standard BCEWithLogitsLoss")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 4. Training with validation
print("Starting training...")
history = {
    'train_loss': [], 
    'train_acc': [], 
    'train_f1_macro': [],
    'train_f1_weighted': [],
    'val_loss': [], 
    'val_acc': [],
    'val_f1_macro': [],
    'val_f1_weighted': []
}
best_val_f1 = 0.0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    # Training
    train_loss, train_acc, train_f1_macro, train_f1_weighted = train_epoch_combined(
        model, train_loader, criterion, optimizer, DEVICE, scaler=scaler
    )
    # Validation
    val_loss, val_acc, val_f1_macro, val_f1_weighted = validate_epoch_combined(
        model, val_loader, criterion, DEVICE
    )

    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_f1_macro'].append(train_f1_macro)
    history['train_f1_weighted'].append(train_f1_weighted)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1_macro'].append(val_f1_macro)
    history['val_f1_weighted'].append(val_f1_weighted)
    
    print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Macro-F1: {train_f1_macro:.4f}, Weighted-F1: {train_f1_weighted:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_f1_macro:.4f}, Weighted-F1: {val_f1_weighted:.4f}")

    # Save best model weights on improvement
    if val_f1_macro > best_val_f1:
        best_val_f1 = val_f1_macro
        save_model_weights(model, 'checkpoints2/lsnet_best.pth')
        print(f"Saved best checkpoint: checkpoints2/lsnet_best.pth (F1={best_val_f1:.4f})")

# Save results
save_model_weights(model, 'checkpoints2/lsnet_final.pth')
plot_learning_curves_combined(history, 'results2/learning_curves_combined.png')
save_training_history(history, 'results2/training_history.json')

# Grad-CAM visualization - Basic (from validation set)
print("\nGenerating basic Grad-CAM visualizations...")
visualize_gradcam(model, val_ds, num_samples=5, device=DEVICE, save_dir='results2/gradcam')

# Advanced Grad-CAM - One per disease with bounding boxes
print("\nGenerating advanced Grad-CAM with bounding boxes...")
generate_gradcam_per_disease(model, class_names, device=DEVICE, 
                            save_dir='results2/gradcam', data_path='data')

# 5. Testing and evaluation
print("\nTesting on test set...")
test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels, test_f1_per_class, avg_ms = test_model_combined(
    model, test_loader, criterion, DEVICE
)
conf_matrix = multilabel_confusion_matrix(test_labels, test_preds)
model_size_mb = os.path.getsize('checkpoints2/lsnet_final.pth') / (1024 ** 2)

# Get class names
class_names = test_ds.my_classes

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Macro F1: {test_f1_macro:.4f}")
print(f"Test Weighted F1: {test_f1_weighted:.4f}")
if avg_ms is not None:
    print(f"Avg Inference Speed: {avg_ms:.2f} ms/image")
print(f"Model Size: {model_size_mb:.2f} MB")

print("\nPer-class F1 scores:")
for i, (class_name, f1) in enumerate(zip(class_names, test_f1_per_class)):
    print(f"  {class_name:20s}: {f1:.4f}")

# Save test results with both metrics
import json
test_metrics = {
    'test_loss': float(test_loss),
    'test_accuracy': float(test_acc),
    'test_f1_macro': float(test_f1_macro),
    'test_f1_weighted': float(test_f1_weighted),
    'test_f1_per_class': {name: float(f1) for name, f1 in zip(class_names, test_f1_per_class)},
    'avg_inference_ms_per_image': float(avg_ms) if avg_ms else None,
    'model_size_mb': float(model_size_mb),
}

with open('results2/test_metrics.json', 'w') as f:
    json.dump(test_metrics, f, indent=2)

# Save confusion matrix
save_test_results(
    test_loss,
    test_acc,
    test_preds,
    test_labels,
    'results2/test_results.json',
    confusion_matrix=conf_matrix,
    avg_inference_ms=avg_ms,
    model_size_mb=model_size_mb,
)

# Create confusion matrix records for visualization
conf_records = []
for idx, mat in enumerate(conf_matrix):
    tn, fp, fn, tp = mat.ravel()
    conf_records.append({
        'class': class_names[idx],
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'tp': int(tp),
    })

with open('results2/confusion_matrix.json', 'w') as f:
    json.dump(conf_records, f, indent=2)
np.save('results2/confusion_matrix.npy', conf_matrix)

# Generate visualizations
print("\nGenerating visualizations...")
plot_true_confusion_matrix(test_labels, test_preds, class_names=class_names,
                          save_path='results2/true_confusion_matrix.png')

plot_confusion_matrix_heatmap(conf_records, class_names=class_names, 
                             save_path='results2/confusion_matrix_metrics.png')

plot_per_class_statistics(conf_records, 
                         save_path='results2/per_class_statistics.json')

print("\nAll results saved to results2/ directory")
print("="*80)
print("SUMMARY:")
print(f"  Best Validation Macro F1: {best_val_f1:.4f}")
print(f"  Final Test Accuracy:      {test_acc:.4f}")
print(f"  Final Test Macro F1:      {test_f1_macro:.4f}")
print(f"  Final Test Weighted F1:   {test_f1_weighted:.4f}")
print("="*80)
