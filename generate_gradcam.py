#!/usr/bin/env python3
"""
Generate Grad-CAM visualizations for one representative image per disease
Samples from BBox_List_2017.csv to ensure ground truth annotations exist
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import patches
from PIL import Image
from tqdm import tqdm

from utils.dataset import nih_chest_dataset
from utils.model import load_lsnet_model
from utils.gradcam import GradCAM


# Configuration
NUM_CLASSES = 14
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32

# CAM options
TTA_FLIP = True  # horizontally flip TTA to stabilize CAM

CLASS_NAMES = [
    'Atelectasis', 'Consolidation', 'Infiltration', 'Pneumothorax',
    'Edema', 'Emphysema', 'Fibrosis', 'Effusion', 'Pneumonia',
    'Pleural_thickening', 'Cardiomegaly', 'Nodule', 'Mass', 'Hernia'
]

# Map class names to indices
CLASS_NAME_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


def load_bbox_data(bbox_csv_path='data/BBox_List_2017.csv'):
    """Load bounding box data from CSV and normalize labels to CLASS_NAMES.

    Normalizations applied:
    - 'Infiltrate' -> 'Infiltration'
    - 'Pleural_Thickening' -> 'Pleural_thickening'
    """
    bbox_dict = {}
    
    try:
        df = pd.read_csv(bbox_csv_path)
        
        for _, row in df.iterrows():
            img_name = row['Image Index']
            raw_label = row['Finding Label']
            # Normalize label variants from CSV to our CLASS_NAMES
            label = {
                'Infiltrate': 'Infiltration',
                'Pleural_Thickening': 'Pleural_thickening'
            }.get(raw_label, raw_label)
            
            try:
                bbox = (float(row.iloc[2]),   # x
                       float(row.iloc[3]),   # y
                       float(row.iloc[4]),   # w
                       float(row.iloc[5]))   # h
            except (ValueError, IndexError):
                continue
            
            if img_name not in bbox_dict:
                bbox_dict[img_name] = []
            
            bbox_dict[img_name].append({
                'label': label,
                'bbox': bbox
            })
        
        print(f"✓ Loaded bbox data for {len(bbox_dict)} images")
        return bbox_dict
    
    except Exception as e:
        print(f"✗ Error loading bbox data: {e}")
        return {}


def get_one_image_per_disease(bbox_data):
    """
    Select one random image per disease from bbox_data
    
    Returns:
        Dict mapping disease_name -> (image_name, disease_label)
        The disease_label ensures we only get BBox for that specific disease
    """
    disease_images = {disease: [] for disease in CLASS_NAMES}
    
    # Group images by disease
    for img_name, bbox_list in bbox_data.items():
        for bbox_info in bbox_list:
            disease = bbox_info['label']
            if disease in disease_images:
                disease_images[disease].append(img_name)
    
    # Select one random image per disease
    selected = {}
    for disease, images in disease_images.items():
        if images:
            img_name = np.random.choice(images)
            selected[disease] = (img_name, disease)  # Store both image name and disease label
            print(f"  {disease:<20} -> {img_name}")
        else:
            selected[disease] = (None, disease)
            print(f"  {disease:<20} -> [NO DATA]")
    
    return selected


def get_images_by_disease(bbox_data):
    """Return mapping disease -> list of image names that have a BBox for that disease."""
    disease_images = {disease: [] for disease in CLASS_NAMES}
    for img_name, bbox_list in bbox_data.items():
        for bbox_info in bbox_list:
            disease = bbox_info['label']
            if disease in disease_images:
                disease_images[disease].append(img_name)
    return disease_images


def load_image_from_disk(img_name, data_path='data'):
    """
    Load image from disk
    The images are stored in images_XXX/images/ subdirectories
    
    Returns:
        (img_tensor, original_size) - tensor is (3, 224, 224), original_size is (W, H)
    """
    # Find which images_XXX folder contains this image
    for i in range(1, 13):  # images_001 to images_012
        folder_name = f'images_{i:03d}'
        img_path = os.path.join(data_path, folder_name, 'images', img_name)
        
        if os.path.exists(img_path):
            # Load image as RGB (3 channels) - matches model's expected input
            img = Image.open(img_path).convert('RGB')
            
            # Record original size BEFORE resizing
            original_size = img.size  # (W, H)
            
            img_np = np.array(img, dtype=np.float32) / 255.0
            
            # Resize to 224x224
            img_pil = Image.fromarray((img_np * 255).astype(np.uint8))
            img_pil = img_pil.resize((224, 224), Image.BILINEAR)
            img_np = np.array(img_pil, dtype=np.float32) / 255.0
            
            # Convert to tensor (3, 224, 224) - RGB format
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1))
            
            return img_tensor, original_size
    
    print(f"  ✗ Image not found: {img_name}")
    return None, None


def generate_gradcam_for_image(model, img_tensor, img_name, bbox_info, class_names,
                               save_dir='results_balanced/gradcam', device='cuda', target_class_name=None):
    """
    Generate and save Grad-CAM visualization for a single image
    """
    
    try:
        # Prefer slightly earlier layers first for better spatial locality,
        # then fall back to deeper layers if needed.
        candidate_layers = []
        try:
            candidate_layers.append(model.blocks3[-1].mixer)
        except Exception:
            pass
        try:
            candidate_layers.append(model.blocks3[-1])
        except Exception:
            pass
        try:
            candidate_layers.append(model.blocks4[-1].mixer)
        except Exception:
            pass
        try:
            candidate_layers.append(model.blocks4[-1])
        except Exception:
            pass
        if not candidate_layers:
            candidate_layers = [model]
        model.eval()
        
        # Forward pass
        img_batch = img_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(img_batch)
            probs = torch.sigmoid(logits)[0].cpu().numpy()
        
        # Decide which class to visualize: prefer specified target_class_name
        if target_class_name is not None and target_class_name in class_names:
            class_idx = class_names.index(target_class_name)
        else:
            class_idx = int(np.argsort(probs)[-1])
        class_prob = float(probs[class_idx])
        
        # Try generating CAM across candidate layers; fall back to top-pred class if flat
        cam = None
        for tl in candidate_layers:
            grad_cam = GradCAM(model, tl)
            cam_try = grad_cam.generate(img_batch, class_idx)
            if isinstance(cam_try, np.ndarray) and cam_try.ndim == 2:
                # Optional TTA flip averaging to stabilize CAM
                if TTA_FLIP:
                    img_flip = torch.flip(img_batch, dims=[3])  # horizontal flip (W dimension)
                    cam_flip = grad_cam.generate(img_flip, class_idx)
                    if isinstance(cam_flip, np.ndarray) and cam_flip.shape == cam_try.shape:
                        cam_try = 0.5 * (cam_try + np.fliplr(cam_flip))
                # Check flatness via dynamic range or std
                if (cam_try.max() - cam_try.min()) > 1e-6 or np.std(cam_try) > 1e-6:
                    cam = cam_try
                    break
        # If still flat and we forced a disease class, try top predicted class
        if cam is None and target_class_name is not None:
            top_idx = int(np.argsort(probs)[-1])
            for tl in candidate_layers:
                grad_cam = GradCAM(model, tl)
                cam_try = grad_cam.generate(img_batch, top_idx)
                cond = (
                    isinstance(cam_try, np.ndarray)
                    and cam_try.ndim == 2
                    and ((cam_try.max() - cam_try.min()) > 1e-6 or np.std(cam_try) > 1e-6)
                )
                if cond:
                    cam = cam_try
                    class_idx = top_idx
                    class_prob = float(probs[class_idx])
                    break
        
        if cam is None or not isinstance(cam, np.ndarray):
            return False
        
        # Handle CAM shape
        if len(cam.shape) == 2:
            cam_vis = cam
        elif len(cam.shape) == 3 and cam.shape[0] == 1:
            cam_vis = cam[0]
        else:
            return False
        
        # Normalize CAM robustly using percentile to avoid outliers; keep non-negative
        cam_np = cam_vis
        p_min = np.percentile(cam_np, 1)
        p_max = np.percentile(cam_np, 99)
        if p_max - p_min < 1e-6:
            # Neutral map (avoid all-blue); mid-level 0.5
            cam_vis = np.ones_like(cam_np, dtype=np.float32) * 0.5
        else:
            cam_vis = (cam_np - p_min) / (p_max - p_min)
            cam_vis = np.clip(cam_vis, 0.0, 1.0)
        
        # Upsample CAM to 224x224
        cam_h, cam_w = cam_vis.shape
        if cam_h != 224 or cam_w != 224:
            cam_pil = Image.fromarray((cam_vis * 255).astype(np.uint8))
            cam_pil = cam_pil.resize((224, 224), Image.BILINEAR)
            cam_vis = np.array(cam_pil, dtype=np.float32) / 255.0
            
            if cam_vis.max() > cam_vis.min():
                cam_vis = (cam_vis - cam_vis.min()) / (cam_vis.max() - cam_vis.min())
        
        # Prepare image for display
        # img_tensor shape: (3, 224, 224) from dataset
        if isinstance(img_tensor, torch.Tensor):
            img_tensor = img_tensor.cpu()
        
        if img_tensor.shape[0] == 1:
            img_np = img_tensor[0].numpy()
            img_np = np.stack([img_np] * 3, axis=-1)
        else:
            # Image is already 3-channel RGB (3, 224, 224)
            img_np = img_tensor.permute(1, 2, 0).numpy()
        
        img_np = np.clip(img_np, 0, 1)
        
        # Create figure
        fig, ax = plt.subplots(1, 1, figsize=(10, 10), dpi=100)
        
        # Get class name
        class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
        
        # Display image + CAM overlay with better visibility
        ax.imshow(img_np, alpha=0.90, origin='upper')  # Ensure top-left origin
        ax.imshow(cam_vis, cmap='jet', alpha=0.35, origin='upper')  # Same origin
        # Fix axis bounds to 224x224 with top-left origin
        ax.set_xlim(0, 224)
        ax.set_ylim(224, 0)
        
        # Draw ground truth bounding box (GUARANTEED to exist)
        if bbox_info:
            # Track labels added to legend to avoid duplicates
            legend_labels = set()
            for bbox_dict in bbox_info:
                x, y, w, h = bbox_dict['bbox']
                try:
                    # Ensure coordinates are floats
                    x, y, w, h = float(x), float(y), float(w), float(h)

                    # Compute intersection with image bounds [0,224]x[0,224]
                    x0 = max(0.0, x)
                    y0 = max(0.0, y)
                    x1 = min(224.0, x + w)
                    y1 = min(224.0, y + h)

                    iw = x1 - x0
                    ih = y1 - y0
                    if iw > 0.5 and ih > 0.5:
                        lbl = f'GT: {bbox_dict["label"]}'
                        # Only include one legend entry per label
                        show_label = lbl if lbl not in legend_labels else None
                        legend_labels.add(lbl)
                        rect = Rectangle((x0, y0), iw, ih,
                                         linewidth=3, edgecolor='red', facecolor='none',
                                         label=show_label, linestyle='-', alpha=1.0, zorder=5)
                        ax.add_patch(rect)
                except Exception as e:
                    print(f"      Warning: Failed to draw bbox: {e}")
        
        # Title
        ax.set_title(f'Grad-CAM: {class_name} ({class_prob:.2%})\n{img_name}',
                     fontsize=12, fontweight='bold')
        ax.axis('off')
        # Only show legend if we actually added any labeled rectangles
        handles, labels = ax.get_legend_handles_labels()
        if any(l for l in labels):
            ax.legend(loc='upper right', fontsize=10)
        
        # Save
        plt.tight_layout()
        os.makedirs(save_dir, exist_ok=True)
        
        # Use disease name and image name as filename for clarity and uniqueness
        gt_disease = bbox_info[0]['label'] if bbox_info else 'unknown'
        safe_disease_name = gt_disease.replace('/', '_').replace(' ', '_')
        safe_img = os.path.splitext(os.path.basename(img_name))[0]
        save_path = os.path.join(save_dir, f'gradcam_{safe_disease_name}_{safe_img}.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        return True
    
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    print("="*70)
    print("GENERATING GRAD-CAM FOR ONE IMAGE PER DISEASE")
    print("="*70)
    
    # Load BBox data
    print("\n1. Loading BBox data...")
    bbox_data = load_bbox_data('data/BBox_List_2017.csv')
    if not bbox_data:
        print("✗ No BBox data loaded. Exiting.")
        return
    
    # Select one image per disease
    print("\n2. Selecting one random image per disease:")
    selected_images = get_one_image_per_disease(bbox_data)
    
    num_found = sum(1 for v in selected_images.values() if v is not None)
    print(f"\n✓ Selected {num_found}/{NUM_CLASSES} diseases")
    
    # Load model
    print("\n3. Loading model...")
    model = load_lsnet_model(num_classes=NUM_CLASSES, device=DEVICE, 
                            checkpoint_path='checkpoints_balanced/lsnet_final.pth')
    model.eval()
    print(f"✓ Model loaded on {DEVICE}")
    
    # Build disease -> candidate images directly from BBox CSV
    print("\n4. Indexing BBox images by disease...")
    disease_to_images = get_images_by_disease(bbox_data)

    # Generate Grad-CAM for each disease by loading from disk
    print("\n5. Generating Grad-CAM visualizations (from BBox images):")
    success_count = 0

    for disease in CLASS_NAMES:
        candidates = disease_to_images.get(disease, [])
        if not candidates:
            print(f"  {disease:<20} [SKIPPED - no BBox images]")
            continue

        chosen = None
        img_tensor = None
        original_size = None
        bbox_info_original = None

        # Try candidates until a valid, existing image is found
        for img_name in candidates:
            img_tensor, original_size = load_image_from_disk(img_name)
            if img_tensor is None:
                continue
            bbox_all = bbox_data.get(img_name, [])
            bbox_info_original = [b for b in bbox_all if b['label'] == disease]
            if not bbox_info_original:
                continue
            chosen = img_name
            break

        if chosen is None or img_tensor is None or not bbox_info_original:
            print(f"  {disease:<20} [SKIPPED - no valid image found]")
            continue

        print(f"  {disease:<20} <- {chosen}")

        # original_size from PIL is (W, H)
        orig_w, orig_h = original_size
        scale_x = 224.0 / float(orig_w)
        scale_y = 224.0 / float(orig_h)

        bbox_info_scaled = []
        for bbox_dict in bbox_info_original:
            x, y, w, h = bbox_dict['bbox']
            x_scaled = x * scale_x
            y_scaled = y * scale_y
            w_scaled = w * scale_x
            h_scaled = h * scale_y
            if w_scaled > 0.5 and h_scaled > 0.5:
                bbox_info_scaled.append({'label': bbox_dict['label'], 'bbox': (x_scaled, y_scaled, w_scaled, h_scaled)})

        if not bbox_info_scaled:
            print(f"    ✗ No valid scaled BBox for {disease}")
            continue

        success = generate_gradcam_for_image(
            model=model,
            img_tensor=img_tensor,
            img_name=chosen,
            bbox_info=bbox_info_scaled,
            class_names=CLASS_NAMES,
            save_dir='results_balanced/gradcam',
            device=DEVICE,
            target_class_name=disease
        )

        if success:
            success_count += 1
            print(f"    ✓ Grad-CAM saved")
        else:
            print(f"    ✗ Failed to generate Grad-CAM")
    
    print("\n" + "="*70)
    print(f"COMPLETED: {success_count}/{num_found} Grad-CAM visualizations saved")
    print(f"Output directory: results_balanced/gradcam/")
    print("="*70)


if __name__ == '__main__':
    main()
