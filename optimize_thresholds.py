"""
Optimize per-class decision thresholds for best F1 score
針對已訓練的模型找出最佳的決策閾值，從而提高召回率
"""
import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_recall_curve
from tqdm import tqdm

from utils.dataset import nih_chest_dataset
from utils.model import load_lsnet_model

DISEASE_NAMES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion',
    'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule',
    'Mass', 'Hernia'
]

def get_predictions_and_labels(model, data_loader, device):
    """獲取模型的預測概率和真實標籤"""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader, desc='Collecting predictions'):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    
    return np.vstack(all_probs), np.vstack(all_labels)

def find_optimal_threshold_f1(y_true, y_probs, search_range=(0.1, 0.9), num_points=81):
    """
    針對單一類別找出使F1 score最大化的閾值
    """
    thresholds = np.linspace(search_range[0], search_range[1], num_points)
    best_f1 = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        y_pred = (y_probs >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    return best_threshold, best_f1

def find_optimal_threshold_recall_target(y_true, y_probs, target_recall=0.7):
    """
    找出能達到目標召回率的最低閾值（同時盡量保持精確率）
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # 找出召回率 >= target_recall 的所有閾值
    valid_indices = np.where(recall >= target_recall)[0]
    
    if len(valid_indices) == 0:
        # 如果無法達到目標召回率，返回能達到最高召回率的閾值
        max_recall_idx = np.argmax(recall)
        return thresholds[max_recall_idx] if max_recall_idx < len(thresholds) else 0.1, recall[max_recall_idx]
    
    # 在滿足召回率的閾值中，選擇精確率最高的
    best_idx = valid_indices[np.argmax(precision[valid_indices])]
    
    # precision_recall_curve 的 thresholds 長度比 precision/recall 少1
    if best_idx < len(thresholds):
        return thresholds[best_idx], recall[best_idx]
    else:
        return 0.1, recall[best_idx]

def optimize_all_thresholds(model, val_loader, device, method='f1'):
    """
    為所有類別優化閾值
    method: 'f1' (最大化F1) 或 'recall' (目標召回率)
    """
    print("Getting predictions on validation set...")
    y_probs, y_true = get_predictions_and_labels(model, val_loader, device)
    
    num_classes = y_true.shape[1]
    optimal_thresholds = []
    
    print(f"\nOptimizing thresholds using method: {method}")
    print("=" * 80)
    
    for i in range(num_classes):
        class_name = DISEASE_NAMES[i]
        
        if method == 'f1':
            # 最大化 F1 score
            optimal_thresh, best_score = find_optimal_threshold_f1(
                y_true[:, i], y_probs[:, i]
            )
            score_name = "F1"
        else:
            # 達到目標召回率
            optimal_thresh, best_score = find_optimal_threshold_recall_target(
                y_true[:, i], y_probs[:, i], target_recall=0.6
            )
            score_name = "Recall"
        
        optimal_thresholds.append(optimal_thresh)
        
        # 用找到的閾值計算各項指標
        y_pred = (y_probs[:, i] >= optimal_thresh).astype(int)
        
        # 計算混淆矩陣元素
        tp = np.sum((y_pred == 1) & (y_true[:, i] == 1))
        fp = np.sum((y_pred == 1) & (y_true[:, i] == 0))
        tn = np.sum((y_pred == 0) & (y_true[:, i] == 0))
        fn = np.sum((y_pred == 0) & (y_true[:, i] == 1))
        
        # 計算指標
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = f1_score(y_true[:, i], y_pred, zero_division=0)
        
        print(f"{class_name:20s} | Thresh: {optimal_thresh:.3f} | "
              f"{score_name}: {best_score:.3f} | "
              f"Sens: {sensitivity:.3f} | Spec: {specificity:.3f} | "
              f"Prec: {precision:.3f} | F1: {f1:.3f}")
    
    return np.array(optimal_thresholds)

def save_thresholds(thresholds, filename='optimal_thresholds.npy'):
    """保存閾值到文件"""
    np.save(filename, thresholds)
    print(f"\nThresholds saved to {filename}")
    
    # 也保存為可讀的文本文件
    txt_filename = filename.replace('.npy', '.txt')
    with open(txt_filename, 'w') as f:
        f.write("# Optimal thresholds for each disease class\n")
        for i, thresh in enumerate(thresholds):
            f.write(f"{DISEASE_NAMES[i]}: {thresh:.4f}\n")
    print(f"Thresholds also saved to {txt_filename}")

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT = 'checkpoints3/lsnet_best.pth'  # 使用最佳模型
    NUM_CLASSES = 14
    BATCH_SIZE = 32
    
    print("="*80)
    print("THRESHOLD OPTIMIZATION FOR NIH CHEST X-RAY MODEL")
    print("="*80)
    
    # 載入模型
    print(f"\nLoading model from {CHECKPOINT}...")
    model = load_lsnet_model(num_classes=NUM_CLASSES, device=DEVICE)
    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)
    
    # 處理不同的checkpoint格式
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # 假設checkpoint本身就是state_dict
            model.load_state_dict(checkpoint)
    else:
        raise ValueError(f"Unexpected checkpoint format: {type(checkpoint)}")
    
    model.eval()
    print("Model loaded successfully!")
    
    # 載入驗證集
    print("\nLoading validation dataset...")
    from torch.utils.data import random_split
    full_train_ds = nih_chest_dataset(split='train')
    train_size = int(0.8 * len(full_train_ds))
    val_size = len(full_train_ds) - train_size
    _, val_ds = random_split(full_train_ds, [train_size, val_size])
    
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=4, pin_memory=True)
    
    print(f"Validation set size: {len(val_ds)}")
    
    # 優化閾值 - 兩種方法
    print("\n" + "="*80)
    print("METHOD 1: Maximize F1 Score")
    print("="*80)
    thresholds_f1 = optimize_all_thresholds(model, val_loader, DEVICE, method='f1')
    save_thresholds(thresholds_f1, 'optimal_thresholds_f1.npy')
    
    print("\n" + "="*80)
    print("METHOD 2: Target Recall (≥60%)")
    print("="*80)
    thresholds_recall = optimize_all_thresholds(model, val_loader, DEVICE, method='recall')
    save_thresholds(thresholds_recall, 'optimal_thresholds_recall.npy')
    
    print("\n" + "="*80)
    print("Optimization complete!")
    print("="*80)
    print("\nNext steps:")
    print("1. Review the thresholds above")
    print("2. Use evaluate_with_thresholds.py to test on test set")
    print("3. Choose the method that best balances your precision/recall needs")

if __name__ == '__main__':
    main()
