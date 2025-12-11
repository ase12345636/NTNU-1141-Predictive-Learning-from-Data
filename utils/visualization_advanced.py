import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import torch
from PIL import Image
from tqdm import tqdm


def plot_confusion_matrix_heatmap(conf_records, class_names=None, save_path='results/confusion_matrix_heatmap.png'):
    """Plot per-class metrics as heatmap"""
    if not conf_records:
        print("No confusion matrix records found")
        return
    
    classes = [rec['class'] for rec in conf_records]
    n_classes = len(classes)
    metrics_names = ['Sensitivity\n(Recall)', 'Specificity', 'Precision', 'F1-Score']
    metrics_matrix = np.zeros((n_classes, len(metrics_names)))
    
    for idx, rec in enumerate(conf_records):
        tn, fp, fn, tp = rec['tn'], rec['fp'], rec['fn'], rec['tp']
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        metrics_matrix[idx, 0] = sensitivity
        metrics_matrix[idx, 1] = specificity
        metrics_matrix[idx, 2] = precision
        metrics_matrix[idx, 3] = f1
    
    fig, ax = plt.subplots(figsize=(12, 14))
    sns.heatmap(metrics_matrix, annot=True, fmt='.3f', cmap='RdYlGn', 
                cbar_kws={'label': 'Score'}, xticklabels=metrics_names,
                yticklabels=classes, ax=ax, vmin=0, vmax=1, square=False,
                cbar=True, linewidths=0.5, linecolor='gray')
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Disease / Condition', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Classification Performance Metrics', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved per-class metrics heatmap to {save_path}")
    plt.close()


def plot_true_confusion_matrix(y_true, y_pred, class_names, save_path='results/true_confusion_matrix.png'):
    """Plot true confusion matrix for multilabel classification"""
    from sklearn.metrics import multilabel_confusion_matrix
    
    cm = multilabel_confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)
    confusion_matrix = np.zeros((n_classes, n_classes))
    
    for class_idx in range(n_classes):
        true_mask = y_true[:, class_idx] == 1
        if true_mask.sum() > 0:
            for pred_class_idx in range(n_classes):
                pred_mask = y_pred[:, pred_class_idx] == 1
                confusion_matrix[class_idx, pred_class_idx] = (true_mask & pred_mask).sum()
    
    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(confusion_matrix, annot=True, fmt='.0f', cmap='Blues',
                cbar_kws={'label': 'Number of Samples'}, xticklabels=class_names,
                yticklabels=class_names, ax=ax, square=True, linewidths=0.5,
                linecolor='lightgray')
    
    ax.set_xlabel('Predicted Class', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Class', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix - Multilabel Classification', fontsize=14, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved true confusion matrix to {save_path}")
    plt.close()


def plot_per_class_statistics(conf_records, save_path='results/per_class_statistics.json'):
    """Save detailed per-class statistics"""
    statistics = {}
    
    for rec in conf_records:
        class_name = rec['class']
        tn, fp, fn, tp = rec['tn'], rec['fp'], rec['fn'], rec['tp']
        total = tn + fp + fn + tp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        accuracy = (tp + tn) / total if total > 0 else 0
        fpr = fp / (tn + fp) if (tn + fp) > 0 else 0
        fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
        
        statistics[class_name] = {
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'total_samples': int(total),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
            'npv': float(npv),
            'f1_score': float(f1),
            'accuracy': float(accuracy),
            'fpr': float(fpr),
            'fnr': float(fnr),
        }
    
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(statistics, f, indent=2)
    
    print(f"Saved per-class statistics to {save_path}")
    print("\n" + "="*80)
    print("PER-CLASS CLASSIFICATION METRICS")
    print("="*80)
    for class_name, metrics in statistics.items():
        print(f"\n{class_name}:")
        print(f"  Sensitivity (Recall):     {metrics['sensitivity']:.4f}")
        print(f"  Specificity:              {metrics['specificity']:.4f}")
        print(f"  Precision:                {metrics['precision']:.4f}")
        print(f"  F1-Score:                 {metrics['f1_score']:.4f}")
        print(f"  TP={metrics['true_positives']:5d}, TN={metrics['true_negatives']:6d}, FP={metrics['false_positives']:5d}, FN={metrics['false_negatives']:5d}")
    print("="*80)


def load_bbox_data(bbox_csv_path='data/BBox_List_2017.csv'):
    """Load bounding box data from CSV"""
    bbox_dict = {}
    try:
        df = pd.read_csv(bbox_csv_path)
        for _, row in df.iterrows():
            img_name = row['Image Index']
            label = row['Finding Label']
            try:
                bbox = (float(row.iloc[2]), float(row.iloc[3]), float(row.iloc[4]), float(row.iloc[5]))
            except (ValueError, IndexError):
                continue
            
            if img_name not in bbox_dict:
                bbox_dict[img_name] = []
            bbox_dict[img_name].append({'label': label, 'bbox': bbox})
        
        print(f"Loaded bbox data for {len(bbox_dict)} images")
        return bbox_dict
    except Exception as e:
        print(f"Error loading bbox data: {e}")
        return {}


def visualize_gradcam_with_bbox(model, dataset, bbox_data, class_names, 
                                 num_samples=5, device='cuda', 
                                 save_dir='results/gradcam_with_bbox',
                                 img_size=224):
    """Generate Grad-CAM from images WITH BBox annotations only"""
    from utils.gradcam import GradCAM
    import warnings
    warnings.filterwarnings('ignore')
    
    os.makedirs(save_dir, exist_ok=True)
    
    if not bbox_data:
        print("No BBox data available - skipping Grad-CAM")
        return False
    
    bbox_image_names = set(bbox_data.keys())
    print(f"Found {len(bbox_image_names)} images with BBox")
    
    try:
        target_layer = None
        try:
            target_layer = model.blocks4[-1].mixer
        except:
            target_layer = model.blocks4[-1] if hasattr(model, 'blocks4') else model
        
        grad_cam = GradCAM(model, target_layer)
        model.eval()
        
        # Find images with BBox
        max_attempts = min(len(dataset), 500)
        sampled_items = []
        
        print(f"Searching for {num_samples} images with BBox (max {max_attempts} attempts)...")
        for attempt in range(max_attempts):
            idx = np.random.randint(0, len(dataset))
            try:
                sample = dataset[idx]
                if isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    img = sample[0]
                    label = sample[1]
                    img_name = sample[2] if len(sample) > 2 else f"sample_{idx}.png"
                    
                    if img_name in bbox_image_names:
                        sampled_items.append((idx, img, label, img_name))
                        if len(sampled_items) >= num_samples:
                            break
            except:
                continue
        
        print(f"Found {len(sampled_items)} images with BBox annotations")
        
        if len(sampled_items) == 0:
            print("Warning: No images with BBox found")
            return False
        
        success_count = 0
        
        for vis_idx, (sample_idx, img, label, img_name) in enumerate(sampled_items):
            try:
                if not isinstance(img, torch.Tensor) or len(img.shape) != 3:
                    continue
                if img.shape[0] not in [1, 3]:
                    continue
                
                img_batch = img.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = model(img_batch)
                    probs = torch.sigmoid(logits)[0].cpu().numpy()
                
                top_class = int(np.argsort(probs)[-1])
                top_prob = float(probs[top_class])
                
                cam = grad_cam.generate(img_batch, top_class)
                
                if cam is None or not isinstance(cam, np.ndarray):
                    continue
                
                if len(cam.shape) == 2:
                    cam_vis = cam
                elif len(cam.shape) == 3 and cam.shape[0] == 1:
                    cam_vis = cam[0]
                else:
                    continue
                
                if cam_vis.max() > cam_vis.min():
                    cam_vis = (cam_vis - cam_vis.min()) / (cam_vis.max() - cam_vis.min())
                else:
                    continue
                
                if img.shape[0] == 1:
                    img_np = img[0].cpu().numpy()
                    img_np = np.stack([img_np] * 3, axis=-1)
                else:
                    img_np = img.permute(1, 2, 0).cpu().numpy()
                
                img_np = np.clip(img_np, 0, 1)
                
                # Upsample CAM
                try:
                    from PIL import Image as PILImage
                    cam_h, cam_w = cam_vis.shape
                    img_h, img_w = img_np.shape[0], img_np.shape[1]
                    
                    if cam_h != img_h or cam_w != img_w:
                        cam_uint8 = (cam_vis * 255).astype(np.uint8)
                        cam_pil = PILImage.fromarray(cam_uint8)
                        cam_pil = cam_pil.resize((img_w, img_h), PILImage.BILINEAR)
                        cam_vis = np.array(cam_pil, dtype=np.float32) / 255.0
                        
                        if cam_vis.max() > cam_vis.min():
                            cam_vis = (cam_vis - cam_vis.min()) / (cam_vis.max() - cam_vis.min())
                except:
                    pass
                
                fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
                class_name = class_names[top_class] if top_class < len(class_names) else f"Class {top_class}"
                
                # Display
                ax.imshow(img_np, alpha=0.85)
                ax.imshow(cam_vis, cmap='jet', alpha=0.25)
                
                # Draw BBoxes - GUARANTEED to exist
                for bbox_info in bbox_data[img_name]:
                    x, y, w, h = bbox_info['bbox']
                    try:
                        rect = Rectangle((x, y), w, h, linewidth=5, edgecolor='red', 
                                       facecolor='none', linestyle='-', alpha=1.0)
                        ax.add_patch(rect)
                    except:
                        pass
                
                ax.legend(['Ground Truth BBox'], loc='upper right', fontsize=10)
                ax.set_title(f'Grad-CAM: {class_name} ({top_prob:.2%})\n{os.path.basename(img_name)}',
                           fontsize=12, fontweight='bold')
                ax.axis('off')
                
                plt.tight_layout()
                save_path = os.path.join(save_dir, f'gradcam_{success_count + 1}.png')
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                
                success_count += 1
                print(f"  ✓ {success_count}/{len(sampled_items)}: {img_name}")
                
            except:
                continue
        
        print(f"\n✓ Saved {success_count} Grad-CAM visualizations with BBox")
        return success_count > 0
    
    except Exception as e:
        print(f"Error: {e}")
        return False
