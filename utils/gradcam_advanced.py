#!/usr/bin/env python3
"""
Advanced Grad-CAM utilities for generating visualizations with bounding boxes
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
from utils.gradcam import GradCAM


def load_bbox_data(bbox_csv_path='data/BBox_List_2017.csv'):
    """Load bounding box data from CSV and normalize labels."""
    bbox_dict = {}
    
    try:
        df = pd.read_csv(bbox_csv_path)
        
        for _, row in df.iterrows():
            img_name = row['Image Index']
            raw_label = row['Finding Label']
            # Normalize label variants
            label = {
                'Infiltrate': 'Infiltration',
                'Pleural_Thickening': 'Pleural_thickening'
            }.get(raw_label, raw_label)
            
            try:
                bbox = (float(row.iloc[2]), float(row.iloc[3]), 
                       float(row.iloc[4]), float(row.iloc[5]))
            except (ValueError, IndexError):
                continue
            
            if img_name not in bbox_dict:
                bbox_dict[img_name] = []
            
            bbox_dict[img_name].append({'label': label, 'bbox': bbox})
        
        print(f"✓ Loaded bbox data for {len(bbox_dict)} images")
        return bbox_dict
    
    except Exception as e:
        print(f"✗ Error loading bbox data: {e}")
        return {}


def get_images_by_disease(bbox_data, class_names):
    """Return mapping disease -> list of image names."""
    disease_images = {disease: [] for disease in class_names}
    for img_name, bbox_list in bbox_data.items():
        for bbox_info in bbox_list:
            disease = bbox_info['label']
            if disease in disease_images:
                disease_images[disease].append(img_name)
    return disease_images


def load_image_from_disk(img_name, data_path='data'):
    """Load image from disk and return tensor and original size."""
    for i in range(1, 13):
        folder_name = f'images_{i:03d}'
        img_path = os.path.join(data_path, folder_name, 'images', img_name)
        
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            original_size = img.size  # (W, H)
            
            img_np = np.array(img, dtype=np.float32) / 255.0
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil = img_pil.resize((224, 224), Image.BILINEAR)
            img_np = np.array(img_pil, dtype=np.float32) / 255.0
            
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
            return img_tensor, original_size
    
    return None, None


def generate_gradcam_with_bbox(model, img_tensor, img_name, bbox_info, class_names,
                                save_dir='results/gradcam', device='cuda', 
                                target_class_name=None, tta_flip=True):
    """Generate and save Grad-CAM visualization with bounding boxes."""
    
    try:
        # Select target layers
        candidate_layers = []
        try:
            candidate_layers.append(model.blocks3[-1].mixer)
        except:
            pass
        try:
            candidate_layers.append(model.blocks4[-1].mixer)
        except:
            pass
        try:
            candidate_layers.append(model.blocks4[-1])
        except:
            pass
        if not candidate_layers:
            candidate_layers = [model]
        
        model.eval()
        img_batch = img_tensor.unsqueeze(0).to(device)
        
        # Forward pass
        with torch.no_grad():
            logits = model(img_batch)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        # Determine target class
        if target_class_name is not None and target_class_name in class_names:
            class_idx = class_names.index(target_class_name)
        else:
            class_idx = int(np.argsort(probs)[-1])
        class_prob = float(probs[class_idx])
        
        # Generate CAM
        cam = None
        for tl in candidate_layers:
            grad_cam = GradCAM(model, tl)
            cam_try = grad_cam.generate(img_batch, class_idx)
            if isinstance(cam_try, np.ndarray) and cam_try.ndim == 2:
                if tta_flip:
                    img_flip = torch.flip(img_batch, dims=[3])
                    cam_flip = grad_cam.generate(img_flip, class_idx)
                    if isinstance(cam_flip, np.ndarray) and cam_flip.shape == cam_try.shape:
                        cam_try = 0.5 * (cam_try + np.fliplr(cam_flip))
                
                if (cam_try.max() - cam_try.min()) > 1e-6:
                    cam = cam_try
                    break
        
        if cam is None:
            return False
        
        # Normalize CAM
        if len(cam.shape) == 2:
            cam_vis = cam
        elif len(cam.shape) == 3 and cam.shape[0] == 1:
            cam_vis = cam[0]
        else:
            return False
        
        p_min, p_max = np.percentile(cam_vis, 1), np.percentile(cam_vis, 99)
        if p_max - p_min < 1e-6:
            cam_vis = np.ones_like(cam_vis, dtype=np.float32) * 0.5
        else:
            cam_vis = (cam_vis - p_min) / (p_max - p_min)
            cam_vis = np.clip(cam_vis, 0.0, 1.0)
        
        # Resize CAM
        if cam_vis.shape != (224, 224):
            cam_pil = Image.fromarray((cam_vis * 255).astype(np.uint8))
            cam_pil = cam_pil.resize((224, 224), Image.BILINEAR)
            cam_vis = np.array(cam_pil, dtype=np.float32) / 255.0
            if cam_vis.max() > cam_vis.min():
                cam_vis = (cam_vis - cam_vis.min()) / (cam_vis.max() - cam_vis.min())
        
        # Prepare image
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.cpu()
        
        if img_tensor.shape[0] == 1:
            img_np = img_tensor[0].numpy()
            img_np = np.stack([img_np] * 3, axis=-1)
        else:
            img_np = img_tensor.permute(1, 2, 0).numpy()
        
        img_np = np.clip(img_np, 0, 1)
        
        # Create visualization
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
        
        ax.imshow(img_np, alpha=0.90, origin='upper')
        ax.imshow(cam_vis, cmap='jet', alpha=0.35, origin='upper')
        ax.set_xlim(0, 224)
        ax.set_ylim(224, 0)
        
        # Draw bounding boxes
        if bbox_info:
            legend_labels = set()
            for bbox_dict in bbox_info:
                x, y, w, h = bbox_dict['bbox']
                try:
                    x, y, w, h = float(x), float(y), float(w), float(h)
                    x0, y0 = max(0.0, x), max(0.0, y)
                    x1, y1 = min(224.0, x + w), min(224.0, y + h)
                    iw, ih = x1 - x0, y1 - y0
                    
                    if iw > 0.5 and ih > 0.5:
                        lbl = f'GT: {bbox_dict["label"]}'
                        show_label = lbl if lbl not in legend_labels else None
                        legend_labels.add(lbl)
                        rect = Rectangle((x0, y0), iw, ih, linewidth=3, 
                                       edgecolor='red', facecolor='none',
                                       label=show_label, linestyle='-', 
                                       alpha=1.0, zorder=5)
                        ax.add_patch(rect)
                except:
                    pass
        
        ax.set_title(f'Grad-CAM: {class_name} ({class_prob:.2%})\n{img_name}',
                    fontsize=12, fontweight='bold')
        ax.axis('off')
        
        handles, labels = ax.get_legend_handles_labels()
        if any(l for l in labels):
            ax.legend(loc='upper right', fontsize=10)
        
        # Save
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        
        gt_disease = bbox_info[0]['label'] if bbox_info else 'unknown'
        safe_disease_name = gt_disease.replace('/', '_').replace(' ', '_')
        safe_img = os.path.splitext(os.path.basename(img_name))[0]
        save_path = os.path.join(save_dir, f'gradcam_{safe_disease_name}_{safe_img}.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error generating Grad-CAM: {e}")
        return False


def generate_gradcam_per_disease(model, class_names, device='cuda', 
                                 save_dir='results/gradcam', data_path='data'):
    """Generate one Grad-CAM visualization per disease with bounding boxes."""
    
    print("\n" + "="*70)
    print("GENERATING GRAD-CAM WITH BOUNDING BOXES (One per disease)")
    print("="*70)
    
    # Load BBox data
    print("\n1. Loading BBox data...")
    bbox_data = load_bbox_data(os.path.join(data_path, 'BBox_List_2017.csv'))
    if not bbox_data:
        print("✗ No BBox data loaded. Skipping Grad-CAM generation.")
        return 0
    
    # Index images by disease
    print("\n2. Indexing BBox images by disease...")
    disease_to_images = get_images_by_disease(bbox_data, class_names)
    
    # Generate Grad-CAM for each disease
    print("\n3. Generating Grad-CAM visualizations:")
    success_count = 0
    
    for disease in class_names:
        candidates = disease_to_images.get(disease, [])
        if not candidates:
            print(f"  {disease:<20} [SKIPPED - no BBox images]")
            continue
        
        # Try to find a valid image
        chosen = None
        img_tensor = None
        original_size = None
        bbox_info_original = None
        
        for img_name in candidates[:10]:  # Try up to 10 candidates
            img_tensor, original_size = load_image_from_disk(img_name, data_path)
            if img_tensor is None:
                continue
            bbox_all = bbox_data.get(img_name, [])
            bbox_info_original = [b for b in bbox_all if b['label'] == disease]
            if not bbox_info_original:
                continue
            chosen = img_name
            break
        
        if chosen is None or img_tensor is None:
            print(f"  {disease:<20} [SKIPPED - no valid image found]")
            continue
        
        print(f"  {disease:<20} <- {chosen}")
        
        # Scale bounding boxes to 224x224
        orig_w, orig_h = original_size
        scale_x, scale_y = 224.0 / float(orig_w), 224.0 / float(orig_h)
        
        bbox_info_scaled = []
        for bbox_dict in bbox_info_original:
            x, y, w, h = bbox_dict['bbox']
            x_s, y_s = x * scale_x, y * scale_y
            w_s, h_s = w * scale_x, h * scale_y
            if w_s > 0.5 and h_s > 0.5:
                bbox_info_scaled.append({
                    'label': bbox_dict['label'], 
                    'bbox': (x_s, y_s, w_s, h_s)
                })
        
        if not bbox_info_scaled:
            print(f"    ✗ No valid scaled BBox")
            continue
        
        # Generate Grad-CAM
        success = generate_gradcam_with_bbox(
            model=model,
            img_tensor=img_tensor,
            img_name=chosen,
            bbox_info=bbox_info_scaled,
            class_names=class_names,
            save_dir=save_dir,
            device=device,
            target_class_name=disease
        )
        
        if success:
            success_count += 1
            print(f"    ✓ Grad-CAM saved")
        else:
            print(f"    ✗ Failed to generate Grad-CAM")
    
    print("\n" + "="*70)
    print(f"COMPLETED: {success_count}/{len(class_names)} Grad-CAM visualizations saved")
    print(f"Output directory: {save_dir}/")
    print("="*70 + "\n")
    
    return success_count
