"""
main4.py - 整合所有最佳實踐的訓練腳本

改進特點：
1. Early Stopping - 防止過擬合
2. 梯度裁剪 - 穩定訓練
3. 優化的 Focal Loss 配置
4. Cosine Annealing with Warm Restarts
5. 更詳細的訓練監控
6. 自動保存最佳模型（基於多個指標）
7. 訓練過程可視化
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import multilabel_confusion_matrix
import json

# Import from utils
from utils.dataset import nih_chest_dataset
from utils.model import load_lsnet_model
from utils.training_combined import train_epoch_combined, validate_epoch_combined, test_model_combined
from utils.visualization import plot_learning_curves_combined, save_training_history, save_test_results, save_model_weights
from utils.visualization_advanced import plot_confusion_matrix_heatmap, plot_true_confusion_matrix, plot_per_class_statistics
from utils.gradcam import visualize_gradcam
from utils.gradcam_advanced import generate_gradcam_per_disease

# ============================================================================
# 配置參數
# ============================================================================

NUM_CLASSES = 14
BATCH_SIZE = 32
MAX_EPOCHS = 60  # 最大訓練輪數
DEVICE = 'cuda'
TRAIN_RATIO = 0.8

# Loss 配置
LOSS_TYPE = 'focal'  # 'focal', 'weighted', or 'bce'
FOCAL_GAMMA = 2.0  # Focal Loss gamma (1.5-2.5，2.0 較平衡)
WEIGHT_SCALE = 0.5  # 類別權重縮放因子 (0.3-0.7，0.5 較平衡)
ALPHA_MIN = 0.35  # Alpha 最小值
ALPHA_MAX = 0.75  # Alpha 最大值

# 優化器配置
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01  # L2 正則化
GRAD_CLIP_NORM = 1.0  # 梯度裁剪

# Early Stopping 配置
EARLY_STOP_PATIENCE = 10  # 連續多少 epoch 沒改進則停止
MIN_DELTA = 0.0001  # 最小改進閾值

# 學習率調度器配置
SCHEDULER_TYPE = 'cosine'  # 'cosine' or 'plateau'
COSINE_T0 = 10  # Cosine Annealing 週期
COSINE_T_MULT = 2  # 週期倍增因子
COSINE_ETA_MIN = 1e-6  # 最小學習率

# 輸出目錄
RESULTS_DIR = 'results4'
CHECKPOINTS_DIR = 'checkpoints4'

# ============================================================================
# 創建輸出目錄
# ============================================================================

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/gradcam', exist_ok=True)

print("="*80)
print("Training Configuration - Best Practices Edition")
print("="*80)
print(f"Loss Type: {LOSS_TYPE}")
if LOSS_TYPE == 'focal':
    print(f"  Gamma: {FOCAL_GAMMA}")
    print(f"  Weight Scale: {WEIGHT_SCALE}")
    print(f"  Alpha Range: [{ALPHA_MIN}, {ALPHA_MAX}]")
print(f"Optimizer: AdamW (lr={LEARNING_RATE}, weight_decay={WEIGHT_DECAY})")
print(f"Scheduler: {SCHEDULER_TYPE}")
print(f"Gradient Clipping: {GRAD_CLIP_NORM}")
print(f"Early Stopping: patience={EARLY_STOP_PATIENCE}, min_delta={MIN_DELTA}")
print(f"Max Epochs: {MAX_EPOCHS}")
print("="*80)

# ============================================================================
# 1. 載入數據
# ============================================================================

print("\nLoading dataset...")
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

# ============================================================================
# 2. 創建模型
# ============================================================================

print("\nLoading LSNet model...")
model = load_lsnet_model(num_classes=NUM_CLASSES, device=DEVICE)

# Mixed precision scaler
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.startswith('cuda'))

# ============================================================================
# 3. 配置 Loss Function
# ============================================================================

def compute_pos_weight(subset, num_classes=NUM_CLASSES):
    """計算每個類別的正樣本權重"""
    ds = subset.dataset
    idxs = subset.indices if hasattr(subset, 'indices') else range(len(ds.samples))
    pos = np.zeros(num_classes, dtype=np.float64)
    for i in idxs:
        pos += ds.samples[i][1]
    n = float(len(idxs))
    pos = np.clip(pos, 1.0, None)
    neg = n - pos
    return torch.tensor(neg / pos, dtype=torch.float32)

pos_weight = compute_pos_weight(train_ds).to(DEVICE)
print(f"\nClass weights statistics:")
print(f"  Mean: {pos_weight.mean():.2f}")
print(f"  Std: {pos_weight.std():.2f}")
print(f"  Min: {pos_weight.min():.2f}")
print(f"  Max: {pos_weight.max():.2f}")

if LOSS_TYPE == 'focal':
    from utils.focal_loss import FocalLoss
    # 線性縮放並限制範圍
    normalized_weight = pos_weight / pos_weight.max()
    alpha = torch.clamp(
        0.5 + (normalized_weight - 0.5) * WEIGHT_SCALE, 
        min=ALPHA_MIN, 
        max=ALPHA_MAX
    ).to(DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA, reduction='mean')
    print(f"\nUsing Focal Loss:")
    print(f"  Gamma: {FOCAL_GAMMA}")
    print(f"  Alpha range: [{alpha.min():.3f}, {alpha.max():.3f}]")
elif LOSS_TYPE == 'weighted':
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("\nUsing Weighted BCEWithLogitsLoss")
else:
    criterion = torch.nn.BCEWithLogitsLoss()
    print("\nUsing standard BCEWithLogitsLoss")

# ============================================================================
# 4. 配置優化器和學習率調度器
# ============================================================================

optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=LEARNING_RATE, 
    weight_decay=WEIGHT_DECAY
)

if SCHEDULER_TYPE == 'cosine':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=COSINE_T0, 
        T_mult=COSINE_T_MULT, 
        eta_min=COSINE_ETA_MIN
    )
    print(f"\nUsing CosineAnnealingWarmRestarts:")
    print(f"  T_0: {COSINE_T0}, T_mult: {COSINE_T_MULT}, eta_min: {COSINE_ETA_MIN}")
else:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3,
        verbose=True
    )
    print(f"\nUsing ReduceLROnPlateau")

# ============================================================================
# 5. Early Stopping 類
# ============================================================================

class EarlyStopping:
    """Early stopping 以防止過擬合"""
    def __init__(self, patience=7, min_delta=0.0001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'max':
            if score < self.best_score + self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        else:  # mode == 'min'
            if score > self.best_score - self.min_delta:
                self.counter += 1
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.counter = 0
        
        return self.early_stop

early_stopping = EarlyStopping(
    patience=EARLY_STOP_PATIENCE, 
    min_delta=MIN_DELTA, 
    mode='max'
)

# ============================================================================
# 6. 訓練循環
# ============================================================================

print("\n" + "="*80)
print("Starting Training...")
print("="*80)

history = {
    'train_loss': [], 
    'train_acc': [], 
    'train_f1_macro': [],
    'train_f1_weighted': [],
    'val_loss': [], 
    'val_acc': [],
    'val_f1_macro': [],
    'val_f1_weighted': [],
    'learning_rate': []
}

best_val_f1 = 0.0
best_val_loss = float('inf')
best_epoch = 0

for epoch in range(MAX_EPOCHS):
    print(f"\nEpoch {epoch+1}/{MAX_EPOCHS}")
    print("-" * 80)
    
    # Training (with gradient clipping)
    train_loss, train_acc, train_f1_macro, train_f1_weighted = train_epoch_combined(
        model, train_loader, criterion, optimizer, DEVICE, 
        scaler=scaler, grad_clip_norm=GRAD_CLIP_NORM
    )
    
    # Validation
    val_loss, val_acc, val_f1_macro, val_f1_weighted = validate_epoch_combined(
        model, val_loader, criterion, DEVICE
    )
    
    # 記錄當前學習率
    current_lr = optimizer.param_groups[0]['lr']
    
    # 更新歷史記錄
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['train_f1_macro'].append(train_f1_macro)
    history['train_f1_weighted'].append(train_f1_weighted)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['val_f1_macro'].append(val_f1_macro)
    history['val_f1_weighted'].append(val_f1_weighted)
    history['learning_rate'].append(current_lr)
    
    # 打印訓練統計
    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
          f"Macro-F1: {train_f1_macro:.4f}, Weighted-F1: {train_f1_weighted:.4f}")
    print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, "
          f"Macro-F1: {val_f1_macro:.4f}, Weighted-F1: {val_f1_weighted:.4f}")
    print(f"Learning Rate: {current_lr:.6f}")
    
    # 更新學習率調度器
    if SCHEDULER_TYPE == 'cosine':
        scheduler.step()
    else:
        scheduler.step(val_f1_macro)
    
    # 保存最佳模型（基於 Macro F1）
    if val_f1_macro > best_val_f1:
        best_val_f1 = val_f1_macro
        best_epoch = epoch + 1
        save_model_weights(model, f'{CHECKPOINTS_DIR}/lsnet_best_f1.pth')
        print(f"✓ New best F1! Saved to {CHECKPOINTS_DIR}/lsnet_best_f1.pth (F1={best_val_f1:.4f})")
    
    # 也保存最佳 loss 模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model_weights(model, f'{CHECKPOINTS_DIR}/lsnet_best_loss.pth')
        print(f"✓ New best loss! Saved to {CHECKPOINTS_DIR}/lsnet_best_loss.pth (Loss={best_val_loss:.4f})")
    
    # Early Stopping 檢查
    if early_stopping(val_f1_macro):
        print(f"\n{'='*80}")
        print(f"Early stopping triggered at epoch {epoch+1}")
        print(f"Best validation F1: {best_val_f1:.4f} at epoch {best_epoch}")
        print(f"{'='*80}")
        break
    
    # 顯示 early stopping 進度
    if early_stopping.counter > 0:
        print(f"⚠ Early stopping counter: {early_stopping.counter}/{EARLY_STOP_PATIENCE}")

# ============================================================================
# 7. 保存最終模型和訓練歷史
# ============================================================================

print("\n" + "="*80)
print("Training Completed!")
print("="*80)

save_model_weights(model, f'{CHECKPOINTS_DIR}/lsnet_final.pth')
plot_learning_curves_combined(history, f'{RESULTS_DIR}/learning_curves_combined.png')
save_training_history(history, f'{RESULTS_DIR}/training_history.json')

print(f"\nTraining Summary:")
print(f"  Total Epochs: {len(history['train_loss'])}")
print(f"  Best Val F1: {best_val_f1:.4f} (Epoch {best_epoch})")
print(f"  Best Val Loss: {best_val_loss:.4f}")
print(f"  Final Train F1: {history['train_f1_macro'][-1]:.4f}")
print(f"  Final Val F1: {history['val_f1_macro'][-1]:.4f}")

# ============================================================================
# 8. 載入最佳模型進行測試
# ============================================================================

print("\n" + "="*80)
print("Loading best model for testing...")
print("="*80)

# 載入最佳 F1 模型
model = load_lsnet_model(
    num_classes=NUM_CLASSES, 
    device=DEVICE, 
    checkpoint_path=f'{CHECKPOINTS_DIR}/lsnet_best_f1.pth'
)

# Grad-CAM 可視化
print("\nGenerating Grad-CAM visualizations...")
visualize_gradcam(model, val_ds, num_samples=5, device=DEVICE, save_dir=f'{RESULTS_DIR}/gradcam')

# 獲取類別名稱
class_names = test_ds.my_classes

# Advanced Grad-CAM
print("Generating advanced Grad-CAM with bounding boxes...")
generate_gradcam_per_disease(model, class_names, device=DEVICE, 
                            save_dir=f'{RESULTS_DIR}/gradcam', data_path='data')

# ============================================================================
# 9. 測試集評估
# ============================================================================

print("\n" + "="*80)
print("Testing on test set...")
print("="*80)

test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels, test_f1_per_class, avg_ms = test_model_combined(
    model, test_loader, criterion, DEVICE
)

conf_matrix = multilabel_confusion_matrix(test_labels, test_preds)
model_size_mb = os.path.getsize(f'{CHECKPOINTS_DIR}/lsnet_best_f1.pth') / (1024 ** 2)

print(f"\nTest Results:")
print(f"  Loss: {test_loss:.4f}")
print(f"  Accuracy: {test_acc:.4f}")
print(f"  Macro F1: {test_f1_macro:.4f}")
print(f"  Weighted F1: {test_f1_weighted:.4f}")
if avg_ms is not None:
    print(f"  Avg Inference Speed: {avg_ms:.2f} ms/image")
print(f"  Model Size: {model_size_mb:.2f} MB")

print("\nPer-class F1 scores:")
for i, (class_name, f1) in enumerate(zip(class_names, test_f1_per_class)):
    print(f"  {class_name:20s}: {f1:.4f}")

# 保存測試指標
test_metrics = {
    'test_loss': float(test_loss),
    'test_accuracy': float(test_acc),
    'test_f1_macro': float(test_f1_macro),
    'test_f1_weighted': float(test_f1_weighted),
    'test_f1_per_class': {name: float(f1) for name, f1 in zip(class_names, test_f1_per_class)},
    'avg_inference_ms_per_image': float(avg_ms) if avg_ms else None,
    'model_size_mb': float(model_size_mb),
    'best_val_f1': float(best_val_f1),
    'best_epoch': int(best_epoch),
    'total_epochs': len(history['train_loss'])
}

with open(f'{RESULTS_DIR}/test_metrics.json', 'w') as f:
    json.dump(test_metrics, f, indent=2)

# 保存混淆矩陣
save_test_results(
    test_loss,
    test_acc,
    test_preds,
    test_labels,
    f'{RESULTS_DIR}/test_results.json',
    confusion_matrix=conf_matrix,
    avg_inference_ms=avg_ms,
    model_size_mb=model_size_mb,
)

# 創建混淆矩陣記錄
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

with open(f'{RESULTS_DIR}/confusion_matrix.json', 'w') as f:
    json.dump(conf_records, f, indent=2)
np.save(f'{RESULTS_DIR}/confusion_matrix.npy', conf_matrix)

# ============================================================================
# 10. 生成可視化
# ============================================================================

print("\n" + "="*80)
print("Generating visualizations...")
print("="*80)

plot_true_confusion_matrix(test_labels, test_preds, class_names=class_names,
                          save_path=f'{RESULTS_DIR}/true_confusion_matrix.png')

plot_confusion_matrix_heatmap(conf_records, class_names=class_names, 
                             save_path=f'{RESULTS_DIR}/confusion_matrix_metrics.png')

plot_per_class_statistics(conf_records, 
                         save_path=f'{RESULTS_DIR}/per_class_statistics.json')

# ============================================================================
# 11. 最終總結
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)
print(f"Training Configuration:")
print(f"  Loss: {LOSS_TYPE}")
print(f"  Optimizer: AdamW (lr={LEARNING_RATE}, wd={WEIGHT_DECAY})")
print(f"  Scheduler: {SCHEDULER_TYPE}")
print(f"  Early Stopping: patience={EARLY_STOP_PATIENCE}")
print()
print(f"Training Results:")
print(f"  Total Epochs: {len(history['train_loss'])} / {MAX_EPOCHS}")
print(f"  Best Val Macro F1: {best_val_f1:.4f} (Epoch {best_epoch})")
print(f"  Best Val Loss: {best_val_loss:.4f}")
print()
print(f"Test Results (Best F1 Model):")
print(f"  Test Accuracy: {test_acc:.4f}")
print(f"  Test Macro F1: {test_f1_macro:.4f}")
print(f"  Test Weighted F1: {test_f1_weighted:.4f}")
print(f"  Model Size: {model_size_mb:.2f} MB")
if avg_ms is not None:
    print(f"  Inference Speed: {avg_ms:.2f} ms/image")
print()
print(f"All results saved to '{RESULTS_DIR}/' directory")
print(f"All checkpoints saved to '{CHECKPOINTS_DIR}/' directory")
print("="*80)
