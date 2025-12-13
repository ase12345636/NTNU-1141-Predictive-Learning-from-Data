"""
配置不同的訓練策略來平衡precision和recall

使用方法：
1. 選擇一個配置（取消註解）
2. 運行: python main2_balanced.py

推薦順序測試：
1. BALANCED (平衡版)
2. RECALL_FOCUS (召回優先) 
3. PRECISION_FOCUS (精確優先)
"""

import os
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import multilabel_confusion_matrix

from utils.dataset import nih_chest_dataset
from utils.model import load_lsnet_model
from utils.training_combined import train_epoch_combined, validate_epoch_combined, test_model_combined
from utils.visualization import plot_learning_curves_combined, save_training_history, save_test_results, save_model_weights
from utils.visualization_advanced import plot_confusion_matrix_heatmap, plot_true_confusion_matrix, plot_per_class_statistics
from utils.gradcam import visualize_gradcam
from utils.gradcam_advanced import generate_gradcam_per_disease

# ============================================================================
# 配置選擇 - 只保留一個取消註解
# ============================================================================

CONFIG = "BALANCED"        # 推薦：平衡precision和recall
# CONFIG = "RECALL_FOCUS"    # 提高召回率（減少漏診）
# CONFIG = "PRECISION_FOCUS"   # 提高精確率（減少誤診）
# CONFIG = "AGGRESSIVE"      # 激進版本（大幅提高召回率）

# ============================================================================
# 根據配置設置參數
# ============================================================================

CONFIGS = {
    "BALANCED": {
        "loss_type": "focal",
        "focal_gamma": 2.0,        # 中等focusing
        "weight_scale": 0.5,       # 中等權重
        "alpha_min": 0.35,
        "alpha_max": 0.75,
        "description": "平衡版：適度關注少數類，gamma=2.0"
    },
    "RECALL_FOCUS": {
        "loss_type": "focal",
        "focal_gamma": 2.5,        # 較高focusing
        "weight_scale": 0.7,       # 較高權重
        "alpha_min": 0.4,
        "alpha_max": 0.8,
        "description": "召回優先：較強關注少數類，適合醫療場景"
    },
    "PRECISION_FOCUS": {
        "loss_type": "focal",
        "focal_gamma": 1.5,        # 較低focusing
        "weight_scale": 0.3,       # 較低權重
        "alpha_min": 0.3,
        "alpha_max": 0.65,
        "description": "精確優先：減少誤報，適合需要高可信度的場景"
    },
    "AGGRESSIVE": {
        "loss_type": "focal",
        "focal_gamma": 3.0,        # 高focusing
        "weight_scale": 0.8,       # 高權重
        "alpha_min": 0.45,
        "alpha_max": 0.85,
        "description": "激進版：大幅提高召回率，可能降低精確率"
    }
}

config = CONFIGS[CONFIG]
print("="*80)
print(f"CONFIGURATION: {CONFIG}")
print(f"Description: {config['description']}")
print("="*80)

# ============================================================================
# 訓練參數
# ============================================================================

NUM_CLASSES = 14
BATCH_SIZE = 32
EPOCHS = 40
DEVICE = 'cuda'
TRAIN_RATIO = 0.8

LOSS_TYPE = config['loss_type']
FOCAL_GAMMA = config['focal_gamma']
WEIGHT_SCALE = config['weight_scale']
ALPHA_MIN = config['alpha_min']
ALPHA_MAX = config['alpha_max']

# 輸出目錄（根據配置命名）
RESULTS_DIR = f'results_{CONFIG.lower()}'
CHECKPOINTS_DIR = f'checkpoints_{CONFIG.lower()}'

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(f'{RESULTS_DIR}/gradcam', exist_ok=True)

# ============================================================================
# 載入數據
# ============================================================================

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

# ============================================================================
# 模型和優化器
# ============================================================================

print("Loading LSNet model...")
model = load_lsnet_model(num_classes=NUM_CLASSES, device=DEVICE)
scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.startswith('cuda'))

# 計算類別權重
def compute_pos_weight(subset, num_classes=NUM_CLASSES):
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
print(f"Positive weights per class (mean={pos_weight.mean():.2f}, max={pos_weight.max():.2f}):")
print(pos_weight.cpu().numpy())

# 配置Loss function
if LOSS_TYPE == 'focal':
    from utils.focal_loss import FocalLoss
    
    # 使用配置的參數計算alpha
    normalized_weight = pos_weight / pos_weight.max()
    alpha = torch.clamp(
        0.5 + (normalized_weight - 0.5) * WEIGHT_SCALE,
        min=ALPHA_MIN,
        max=ALPHA_MAX
    ).to(DEVICE)
    
    criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA, reduction='mean')
    print(f"\nLoss Configuration:")
    print(f"  Type: Focal Loss")
    print(f"  Gamma: {FOCAL_GAMMA}")
    print(f"  Alpha range: [{alpha.min():.3f}, {alpha.max():.3f}]")
    print(f"  Weight scale: {WEIGHT_SCALE}")
elif LOSS_TYPE == 'weighted':
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("Using Weighted BCEWithLogitsLoss")
else:
    criterion = torch.nn.BCEWithLogitsLoss()
    print("Using standard BCEWithLogitsLoss")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3
)
print(f"Learning rate scheduler: ReduceLROnPlateau (mode=max, patience=3)")

# ============================================================================
# 訓練循環
# ============================================================================

print("\nStarting training...")
history = {
    'train_loss': [], 'train_acc': [], 'train_f1_macro': [], 'train_f1_weighted': [],
    'val_loss': [], 'val_acc': [], 'val_f1_macro': [], 'val_f1_weighted': []
}
best_val_f1 = 0.0

for epoch in range(EPOCHS):
    print(f"\n{'='*80}")
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"{'='*80}")
    
    train_loss, train_acc, train_f1_macro, train_f1_weighted = train_epoch_combined(
        model, train_loader, criterion, optimizer, DEVICE, scaler=scaler
    )
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
    
    print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | "
          f"F1-Macro: {train_f1_macro:.4f} | F1-Weighted: {train_f1_weighted:.4f}")
    print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | "
          f"F1-Macro: {val_f1_macro:.4f} | F1-Weighted: {val_f1_weighted:.4f}")
    print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    scheduler.step(val_f1_macro)

    if val_f1_macro > best_val_f1:
        best_val_f1 = val_f1_macro
        save_model_weights(model, f'{CHECKPOINTS_DIR}/lsnet_best.pth')
        print(f"✓ Saved best checkpoint (F1={best_val_f1:.4f})")

# ============================================================================
# 保存和評估
# ============================================================================

save_model_weights(model, f'{CHECKPOINTS_DIR}/lsnet_final.pth')
plot_learning_curves_combined(history, f'{RESULTS_DIR}/learning_curves.png')
save_training_history(history, f'{RESULTS_DIR}/training_history.json')

print("\n" + "="*80)
print("Testing on test set...")
print("="*80)

test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels, test_f1_per_class, avg_ms = test_model_combined(
    model, test_loader, criterion, DEVICE
)

conf_matrix = multilabel_confusion_matrix(test_labels, test_preds)
model_size_mb = os.path.getsize(f'{CHECKPOINTS_DIR}/lsnet_final.pth') / (1024 ** 2)
class_names = test_ds.my_classes

print(f"\nTest Results:")
print(f"  Accuracy:     {test_acc:.4f}")
print(f"  F1 Macro:     {test_f1_macro:.4f}")
print(f"  F1 Weighted:  {test_f1_weighted:.4f}")
print(f"  Inference:    {avg_ms:.2f} ms/image")
print(f"  Model Size:   {model_size_mb:.2f} MB")

print("\nPer-class F1 scores:")
for class_name, f1 in zip(class_names, test_f1_per_class):
    print(f"  {class_name:20s}: {f1:.4f}")

# 保存結果
import json
test_metrics = {
    'config': CONFIG,
    'config_params': config,
    'test_loss': float(test_loss),
    'test_accuracy': float(test_acc),
    'test_f1_macro': float(test_f1_macro),
    'test_f1_weighted': float(test_f1_weighted),
    'test_f1_per_class': {name: float(f1) for name, f1 in zip(class_names, test_f1_per_class)},
    'avg_inference_ms': float(avg_ms) if avg_ms else None,
    'model_size_mb': float(model_size_mb),
}

with open(f'{RESULTS_DIR}/test_metrics.json', 'w') as f:
    json.dump(test_metrics, f, indent=2)

# 混淆矩陣
conf_records = []
for idx, mat in enumerate(conf_matrix):
    tn, fp, fn, tp = mat.ravel()
    conf_records.append({
        'class': class_names[idx],
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    })

with open(f'{RESULTS_DIR}/confusion_matrix.json', 'w') as f:
    json.dump(conf_records, f, indent=2)
np.save(f'{RESULTS_DIR}/confusion_matrix.npy', conf_matrix)

# 視覺化
print("\nGenerating visualizations...")
plot_true_confusion_matrix(test_labels, test_preds, class_names=class_names,
                          save_path=f'{RESULTS_DIR}/confusion_matrix_full.png')
plot_confusion_matrix_heatmap(conf_records, class_names=class_names,
                             save_path=f'{RESULTS_DIR}/confusion_matrix_metrics.png')
plot_per_class_statistics(conf_records, save_path=f'{RESULTS_DIR}/per_class_statistics.json')

print("\n" + "="*80)
print(f"TRAINING COMPLETE - {CONFIG}")
print("="*80)
print(f"Best Val F1:      {best_val_f1:.4f}")
print(f"Final Test F1:    {test_f1_macro:.4f}")
print(f"Results saved to: {RESULTS_DIR}/")
print("="*80)
