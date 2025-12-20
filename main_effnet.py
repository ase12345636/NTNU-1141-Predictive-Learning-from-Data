import os
import torch
import datetime
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.metrics import multilabel_confusion_matrix
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

# Import from utils
from utils.dataset import nih_chest_dataset
from utils.model import load_lsnet_model
from utils.model_effnet import load_efficientnet_model
from utils.training_combined import train_epoch_combined, validate_epoch_combined, test_model_combined, train_epoch_combined_auc, validate_epoch_combined_auc, test_model_combined_auc
from utils.visualization import plot_learning_curves_combined, plot_learning_curves_combined_auc, save_training_history, save_test_results, save_model_weights
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
# Focal Loss 參數調整 - 平衡版本
FOCAL_GAMMA = float(os.getenv('FOCAL_GAMMA', '2.5'))  # 2.5較為平衡（不要太高）
# 控制類別權重的強度：1.0=完全使用pos_weight, 0.5=減半, 0=不使用
WEIGHT_SCALE = float(os.getenv('WEIGHT_SCALE', '0.6'))  # 降低權重強度

# 0 ~ 7
EFFNET_COMPLEXITY = 2

# 輸出目錄（根據配置命名）
RESULTS_DIR = f'results_effnet_b{EFFNET_COMPLEXITY}'
CHECKPOINTS_DIR = f'checkpoints_effnet_b{EFFNET_COMPLEXITY}'
CHECKPOINTS_FILENAME = f'effnetb{EFFNET_COMPLEXITY}_best'
CHECKPOINTS_FILENAME_FINAL = f'effnetb{EFFNET_COMPLEXITY}_final'

def load_dataset():
    full_train_ds = nih_chest_dataset(split='train')
    test_ds = nih_chest_dataset(split='test')

    train_size = int(TRAIN_RATIO * len(full_train_ds))
    val_size = len(full_train_ds) - train_size
    # train_ds, val_ds = random_split(full_train_ds, [train_size, val_size])
    msss = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=None)
    indices = np.arange(len(full_train_ds))
    # 假設 full_train_ds.samples[i][1] 是 multi-hot label
    labels = np.array([s[1] for s in full_train_ds.samples])
    train_idx, val_idx = next(msss.split(indices, labels))
    train_ds = Subset(full_train_ds, train_idx)
    val_ds   = Subset(full_train_ds, val_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    print(f'Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}')
    
    return train_ds, train_loader, val_ds, val_loader, test_ds, test_loader

# Helper: compute per-class positive weights from the training subset
def compute_pos_weight(subset, num_classes=NUM_CLASSES):
    ds = subset.dataset if hasattr(subset, 'dataset') else subset
    idxs = subset.indices if hasattr(subset, 'indices') else range(len(ds.samples))
    pos = np.zeros(num_classes, dtype=np.float64)
    for i in idxs:
        pos += ds.samples[i][1]
    n = float(len(idxs))
    pos = np.clip(pos, 1.0, None)  # avoid division by zero
    neg = n - pos
    return torch.tensor(neg / pos, dtype=torch.float32)

def get_criterion(loss_type, pos_weight):
    if loss_type == 'focal':
        from utils.focal_loss import FocalLoss
        # 方法1: 使用平方根縮放（較溫和）
        # alpha = torch.sqrt(pos_weight / pos_weight.max()).to(DEVICE)
        
        # 方法2: 線性縮放 + clamp（推薦）
        # 先normalize到[0,1]，然後縮放到合適範圍
        normalized_weight = pos_weight / pos_weight.max()
        # 使用WEIGHT_SCALE控制強度：越小越平衡，越大越關注少數類
        alpha = torch.clamp(0.5 + (normalized_weight - 0.5) * WEIGHT_SCALE, min=0.3, max=0.8).to(DEVICE)
        
        criterion = FocalLoss(alpha=alpha, gamma=FOCAL_GAMMA, reduction='mean')
        print(f"Using Focal Loss (gamma={FOCAL_GAMMA}, alpha range: {alpha.min():.3f}-{alpha.max():.3f})")
        print(f"Weight scale factor: {WEIGHT_SCALE}")
    elif loss_type == 'weighted':
        criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print("Using Weighted BCEWithLogitsLoss with computed pos_weight")
    else:
        criterion = torch.nn.BCEWithLogitsLoss()
        print("Using standard BCEWithLogitsLoss")
    
    return criterion


def main():
    # Create output directories
    os.makedirs(f'{RESULTS_DIR}', exist_ok=True)
    os.makedirs(f'{CHECKPOINTS_DIR}', exist_ok=True)
    os.makedirs(f'{RESULTS_DIR}/gradcam', exist_ok=True)

    #------------------------------------
    # 1. Load NIH chest dataset
    #------------------------------------
    print("Loading dataset...")
    train_ds, train_loader, val_ds, val_loader, test_ds, test_loader = load_dataset()
    # Get class names
    class_names = test_ds.my_classes

    #------------------------------------
    # 2. Create LSNet model
    #------------------------------------
    # print("Loading LSNet model...")
    # model = load_lsnet_model(num_classes=NUM_CLASSES, device=DEVICE)
    print("Loading EfficientNet model...")
    model = load_efficientnet_model(num_classes=NUM_CLASSES, device=DEVICE, model_complexity=EFFNET_COMPLEXITY)

    # Mixed precision scaler (enabled on CUDA)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.startswith('cuda'))

    #------------------------------------
    # 3. Loss & Optimizer (for multilabel)
    #------------------------------------
    pos_weight = compute_pos_weight(train_ds).to(DEVICE)
    print(f"Positive weights per class: {pos_weight.cpu().numpy()}")
    criterion = get_criterion(LOSS_TYPE, pos_weight)

    # 使用AdamW + 學習率調度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    print(f"Learning rate scheduler: ReduceLROnPlateau (mode=max, patience=3)")

    # 4. Training with validation
    print("Starting training...")
    history = {
        'epoch': 0, 
        'best_epoch': 0, 
        'train_loss': [], 
        'train_acc': [], 
        'train_f1_macro': [],
        'train_f1_weighted': [],
        'train_auc_macro': [],
        'train_auc_weighted': [],
        'train_auprc_macro': [],
        'train_auprc_weighted': [],
        'val_loss': [], 
        'val_acc': [],
        'val_f1_macro': [],
        'val_f1_weighted': [],
        'val_auc_macro': [],
        'val_auc_weighted': [],
        'val_auprc_macro': [],
        'val_auprc_weighted': [],
    }
    best_val_f1 = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        # Training
        train_loss, train_acc, train_f1_macro, train_f1_weighted, train_auc_macro, train_auc_weighted, train_auprc_macro, train_auprc_weighted = train_epoch_combined_auc(
            model, train_loader, criterion, optimizer, DEVICE, scaler=scaler
        )
        # Validation
        val_loss, val_acc, val_f1_macro, val_f1_weighted, val_auc_macro, val_auc_weighted, val_auprc_macro, val_auprc_weighted = validate_epoch_combined_auc(
            model, val_loader, criterion, DEVICE
        )

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1_macro'].append(train_f1_macro)
        history['train_f1_weighted'].append(train_f1_weighted)
        history['train_auc_macro'].append(train_auc_macro)
        history['train_auc_weighted'].append(train_auc_weighted)
        history['train_auprc_macro'].append(train_auprc_macro)
        history['train_auprc_weighted'].append(train_auprc_weighted)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1_macro'].append(val_f1_macro)
        history['val_f1_weighted'].append(val_f1_weighted)
        history['val_auc_macro'].append(val_auc_macro)
        history['val_auc_weighted'].append(val_auc_weighted)
        history['val_auprc_macro'].append(val_auprc_macro)
        history['val_auprc_weighted'].append(val_auprc_weighted)
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Macro-F1: {train_f1_macro:.4f}, Weighted-F1: {train_f1_weighted:.4f}, Macro-AUC: {train_auc_macro:.4f}, Weighted-AUC: {train_auc_weighted:.4f}, Macro-AUPRC: {train_auprc_macro:.4f}, Weighted-AUPRC: {train_auprc_weighted:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Macro-F1: {val_f1_macro:.4f}, Weighted-F1: {val_f1_weighted:.4f}, Macro-AUC: {val_auc_macro:.4f}, Weighted-AUC: {val_auc_weighted:.4f}, Macro-AUPRC: {val_auprc_macro:.4f}, Weighted-AUPRC: {val_auprc_weighted:.4f}")
        print(f"Current LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 更新學習率調度器
        scheduler.step(val_f1_macro)

        # Save best model weights on improvement
        if val_f1_macro > best_val_f1:
            best_val_f1 = val_f1_macro
            best_epoch = epoch + 1
            save_model_weights(model, f'{CHECKPOINTS_DIR}/{CHECKPOINTS_FILENAME}.pth')
            print(f"Saved best checkpoint: {CHECKPOINTS_DIR}/{CHECKPOINTS_FILENAME}.pth (F1={best_val_f1:.4f})")

        history['epoch'] = epoch + 1
        history['best_epoch'] = best_epoch
        plot_learning_curves_combined(history, f'{RESULTS_DIR}/learning_curves_combined.png')
        plot_learning_curves_combined_auc(history, f'{RESULTS_DIR}/learning_curves_combined_auc.png')
        save_training_history(history, f'{RESULTS_DIR}/training_history.json')

    # Save results
    history['epoch'] = epoch + 1
    history['best_epoch'] = best_epoch
    save_model_weights(model, f'{CHECKPOINTS_DIR}/{CHECKPOINTS_FILENAME_FINAL}.pth')
    plot_learning_curves_combined(history, f'{RESULTS_DIR}/learning_curves_combined.png')
    plot_learning_curves_combined_auc(history, f'{RESULTS_DIR}/learning_curves_combined_auc.png')
    save_training_history(history, f'{RESULTS_DIR}/training_history.json')

    # Grad-CAM visualization - Basic (from validation set)
    print("\nGenerating basic Grad-CAM visualizations...")
    visualize_gradcam(model, val_ds, num_samples=5, device=DEVICE, save_dir=f'{RESULTS_DIR}/gradcam')

    # Advanced Grad-CAM - One per disease with bounding boxes
    print("\nGenerating advanced Grad-CAM with bounding boxes...")
    generate_gradcam_per_disease(model, class_names, device=DEVICE, 
                                save_dir=f'{RESULTS_DIR}/gradcam', data_path='data')

    # 5. Testing and evaluation
    print("\nTesting on test set...")
    test_loss, test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels, test_f1_per_class, avg_ms, test_scores, test_auc_macro, test_auc_weighted, test_auprc_macro, test_auprc_weighted = test_model_combined_auc(
        model, test_loader, criterion, DEVICE
    )
    conf_matrix = multilabel_confusion_matrix(test_labels, test_preds)
    model_size_mb = os.path.getsize(f'{CHECKPOINTS_DIR}/{CHECKPOINTS_FILENAME_FINAL}.pth') / (1024 ** 2)


    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Macro F1: {test_f1_macro:.4f}")
    print(f"Test Weighted F1: {test_f1_weighted:.4f}")
    print(f"Test Macro AUC / AUROC: {test_auc_macro:.4f}")
    print(f"Test Weighted AUC / AUROC: {test_auc_weighted:.4f}")
    print(f"Test Macro AUPRC: {test_auprc_macro:.4f}")
    print(f"Test Weighted AUPRC: {test_auprc_weighted:.4f}")
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
        'test_auc_macro': float(test_auc_macro),
        'test_auc_weighted': float(test_auc_weighted),
        'test_auprc_macro': float(test_auprc_macro),
        'test_auprc_weighted': float(test_auprc_weighted),
        'test_f1_per_class': {name: float(f1) for name, f1 in zip(class_names, test_f1_per_class)},
        'avg_inference_ms_per_image': float(avg_ms) if avg_ms else None,
        'model_size_mb': float(model_size_mb),
    }

    with open(f'{RESULTS_DIR}/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # Save confusion matrix
    save_test_results(
        test_loss,
        test_acc,
        test_preds,
        test_labels,
        f'{RESULTS_DIR}/test_results.json',
        confusion_matrix=conf_matrix,
        avg_inference_ms=avg_ms,
        model_size_mb=model_size_mb,
        test_scores=test_scores,
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

    with open(f'{RESULTS_DIR}/confusion_matrix.json', 'w') as f:
        json.dump(conf_records, f, indent=2)
    np.save(f'{RESULTS_DIR}/confusion_matrix.npy', conf_matrix)

    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_true_confusion_matrix(test_labels, test_preds, class_names=class_names,
                            save_path=f'{RESULTS_DIR}/true_confusion_matrix.png')

    plot_confusion_matrix_heatmap(conf_records, class_names=class_names, 
                                save_path=f'{RESULTS_DIR}/confusion_matrix_metrics.png')

    plot_per_class_statistics(conf_records, 
                            save_path=f'{RESULTS_DIR}/per_class_statistics.json')

    print(f"\nAll results saved to {RESULTS_DIR}/ directory")
    print("="*80)
    print("SUMMARY:")
    print(f"  Best Validation Macro F1:     {best_val_f1:.4f}")
    print(f"  Final Test Accuracy:          {test_acc:.4f}")
    print(f"  Final Test Macro F1:          {test_f1_macro:.4f}")
    print(f"  Final Test Weighted F1:       {test_f1_weighted:.4f}")
    print(f"  Final Test Macro AUC:         {test_auc_macro:.4f}")
    print(f"  Final Test Weighted AUC:      {test_auc_weighted:.4f}")
    print(f"  Final Test Macro AUPRC:       {test_auprc_macro:.4f}")
    print(f"  Final Test Weighted AUPRC:    {test_auprc_weighted:.4f}")
    print("="*80)


if __name__ == "__main__":
    start = datetime.datetime.now()
    print(f"START: {start.strftime('%Y-%m-%d %H:%M:%S')}")
    main()
    end = datetime.datetime.now()
    print(f"FINISH: {end.strftime('%Y-%m-%d %H:%M:%S')}")
    duration = end - start
    print(f"EXE TIME: {duration}")
    print("="*80)
