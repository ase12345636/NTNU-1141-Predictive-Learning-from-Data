"""
使用優化後的閾值重新評估模型在測試集上的表現
"""
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score, classification_report
from tqdm import tqdm

from utils.dataset import nih_chest_dataset
from utils.model import load_lsnet_model

DISEASE_NAMES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion',
    'Pneumonia', 'Pleural_Thickening', 'Cardiomegaly', 'Nodule',
    'Mass', 'Hernia'
]

def evaluate_with_custom_thresholds(model, test_loader, device, thresholds):
    """
    使用自定義閾值評估模型
    thresholds: shape [num_classes]
    """
    model.eval()
    all_probs = []
    all_labels = []
    
    print("Collecting predictions...")
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc='Testing'):
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    
    y_probs = np.vstack(all_probs)
    y_true = np.vstack(all_labels)
    
    # 使用自定義閾值進行預測
    y_pred = np.zeros_like(y_probs)
    for i in range(len(thresholds)):
        y_pred[:, i] = (y_probs[:, i] >= thresholds[i]).astype(int)
    
    # 計算各項指標
    print("\n" + "="*80)
    print("PER-CLASS METRICS WITH OPTIMIZED THRESHOLDS")
    print("="*80)
    
    results = []
    for i, class_name in enumerate(DISEASE_NAMES):
        tp = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 1))
        fp = np.sum((y_pred[:, i] == 1) & (y_true[:, i] == 0))
        tn = np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 0))
        fn = np.sum((y_pred[:, i] == 0) & (y_true[:, i] == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
        
        results.append({
            'class': class_name,
            'threshold': float(thresholds[i]),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'npv': float(npv),
            'f1_score': float(f1)
        })
        
        print(f"{class_name:20s} | Thresh: {thresholds[i]:.3f} | "
              f"Sens: {sensitivity:.3f} | Spec: {specificity:.3f} | "
              f"Prec: {precision:.3f} | F1: {f1:.3f} | "
              f"TP: {tp:4d} | FP: {fp:4d} | FN: {fn:4d}")
    
    # 計算整體指標
    accuracy = accuracy_score(y_true.ravel(), y_pred.ravel())
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
    
    print("\n" + "="*80)
    print("OVERALL METRICS")
    print("="*80)
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"F1 Macro:        {f1_macro:.4f}")
    print(f"F1 Weighted:     {f1_weighted:.4f}")
    print(f"F1 Micro:        {f1_micro:.4f}")
    
    return {
        'per_class': results,
        'overall': {
            'accuracy': float(accuracy),
            'f1_macro': float(f1_macro),
            'f1_weighted': float(f1_weighted),
            'f1_micro': float(f1_micro)
        }
    }

def compare_thresholds(model, test_loader, device):
    """比較不同閾值策略的效果"""
    
    # 1. 默認閾值 0.5
    print("\n" + "="*80)
    print("BASELINE: Using threshold=0.5 for all classes")
    print("="*80)
    thresholds_05 = np.ones(14) * 0.5
    results_05 = evaluate_with_custom_thresholds(model, test_loader, device, thresholds_05)
    
    # 2. F1優化的閾值
    if os.path.exists('optimal_thresholds_f1.npy'):
        print("\n" + "="*80)
        print("OPTIMIZED: Using F1-maximizing thresholds")
        print("="*80)
        thresholds_f1 = np.load('optimal_thresholds_f1.npy')
        results_f1 = evaluate_with_custom_thresholds(model, test_loader, device, thresholds_f1)
    else:
        print("\nWarning: optimal_thresholds_f1.npy not found. Run optimize_thresholds.py first.")
        results_f1 = None
    
    # 3. 召回率優化的閾值
    if os.path.exists('optimal_thresholds_recall.npy'):
        print("\n" + "="*80)
        print("OPTIMIZED: Using Recall-targeting thresholds")
        print("="*80)
        thresholds_recall = np.load('optimal_thresholds_recall.npy')
        results_recall = evaluate_with_custom_thresholds(model, test_loader, device, thresholds_recall)
    else:
        print("\nWarning: optimal_thresholds_recall.npy not found. Run optimize_thresholds.py first.")
        results_recall = None
    
    # 保存結果
    comparison = {
        'baseline_0.5': results_05,
        'f1_optimized': results_f1,
        'recall_optimized': results_recall
    }
    
    with open('threshold_comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    print("\n" + "="*80)
    print("Results saved to threshold_comparison.json")
    print("="*80)
    
    # 打印比較摘要
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"{'Method':<25} | {'Accuracy':<10} | {'F1 Macro':<10} | {'F1 Weighted':<10}")
    print("-" * 80)
    print(f"{'Baseline (0.5)':<25} | {results_05['overall']['accuracy']:<10.4f} | "
          f"{results_05['overall']['f1_macro']:<10.4f} | {results_05['overall']['f1_weighted']:<10.4f}")
    if results_f1:
        print(f"{'F1-Optimized':<25} | {results_f1['overall']['accuracy']:<10.4f} | "
              f"{results_f1['overall']['f1_macro']:<10.4f} | {results_f1['overall']['f1_weighted']:<10.4f}")
    if results_recall:
        print(f"{'Recall-Optimized':<25} | {results_recall['overall']['accuracy']:<10.4f} | "
              f"{results_recall['overall']['f1_macro']:<10.4f} | {results_recall['overall']['f1_weighted']:<10.4f}")

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    CHECKPOINT = 'checkpoints3/lsnet_best.pth'
    NUM_CLASSES = 14
    BATCH_SIZE = 32
    
    print("="*80)
    print("EVALUATE MODEL WITH OPTIMIZED THRESHOLDS")
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
    
    # 載入測試集
    print("\nLoading test dataset...")
    test_ds = nih_chest_dataset(split='test')
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=4, pin_memory=True)
    print(f"Test set size: {len(test_ds)}")
    
    # 比較不同閾值策略
    compare_thresholds(model, test_loader, DEVICE)
    
    print("\n" + "="*80)
    print("Evaluation complete!")
    print("="*80)

if __name__ == '__main__':
    main()
