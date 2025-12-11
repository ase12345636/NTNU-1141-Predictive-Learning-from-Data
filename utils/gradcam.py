import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class GradCAM:
    """Grad-CAM for visualization"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activations)
        # Use full backward hook to avoid deprecation and ensure correctness
        target_layer.register_full_backward_hook(self.save_gradients)
    
    def save_activations(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input, class_idx):
        """Generate Grad-CAM heatmap
        
        Args:
            input: Input tensor of shape (1, C, H, W)
            class_idx: Class index to generate CAM for
            
        Returns:
            cam: 2D numpy array of shape (H, W) with normalized values [0, 1]
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input)
        self.model.zero_grad()
        
        # Use sigmoid score for multi-label setups to align with probabilities
        # This often stabilizes Grad-CAM for medical multi-label tasks.
        score = torch.sigmoid(output)[:, class_idx]
        # Backward pass for specific class
        target = score
        target.sum().backward()
        
        # Get gradients and activations
        if self.gradients is None or self.activations is None:
            # Fallback: return a blank CAM if hooks didn't work
            h, w = input.shape[2], input.shape[3]
            return np.ones((h, w)) * 0.5
        
        # Expect tensors in (B, C, H, W). Compute channel weights by
        # averaging gradients over spatial dimensions (H, W) per channel.
        acts = self.activations[0]              # (C, H, W)
        grads = self.gradients[0]               # (C, H, W)
        
        if grads.dim() != 3 or acts.dim() != 3:
            # Fallback: return a neutral CAM if unexpected shape
            h, w = input.shape[2], input.shape[3]
            return np.ones((h, w)) * 0.5
        
        # Only positive gradients should contribute (Grad-CAM paper)
        grads = torch.relu(grads)

        # Channel-wise importance weights alpha_k (average over spatial dims)
        weights = grads.mean(dim=(1, 2))        # (C)
        
        # Weighted sum over channels
        cam_t = (weights.view(-1, 1, 1) * acts).sum(dim=0)  # (H, W)
        cam = cam_t.detach().cpu().numpy()
        
        # ReLU on CAM to keep only positive influences
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        cam_max = cam.max()
        cam_min = cam.min()
        
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            # All zeros, return blank
            cam = np.ones_like(cam) * 0.5
        
        # Ensure output is 2D
        assert len(cam.shape) == 2, f"CAM should be 2D, got shape {cam.shape}"
        
        return cam

def visualize_gradcam(model, val_ds, num_samples=5, device='cuda', save_dir='gradcam'):
    """Generate and save Grad-CAM visualizations for validation samples"""
    try:
        # Choose a deep convolutional/attention block as target layer
        target_layer = None
        try:
            target_layer = model.blocks4[-1].mixer
        except Exception:
            target_layer = model.blocks4[-1] if hasattr(model, 'blocks4') else model

        grad_cam = GradCAM(model, target_layer)
        
        num_vis = min(num_samples, len(val_ds))
        sample_indices = np.random.choice(len(val_ds), num_vis, replace=False)
        
        for idx, sample_idx in enumerate(sample_indices):
            img, label = val_ds[sample_idx]
            img_batch = img.unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = model(img_batch)
                probs = torch.sigmoid(logits)[0].cpu().numpy()
            
            # Get top predicted class
            top_class = int(np.argsort(probs)[-1])
            
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            
            # Original image (RGB)
            img_np = img.permute(1, 2, 0).cpu().numpy()
            axes[0].imshow(img_np)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Grad-CAM heatmap
            cam = grad_cam.generate(img_batch, top_class)
            axes[1].imshow(img_np, alpha=0.6)
            axes[1].imshow(cam, cmap='jet', alpha=0.4)
            axes[1].set_title(f'Grad-CAM (Class: {top_class})')
            axes[1].axis('off')
            
            plt.tight_layout()
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(f'{save_dir}/sample_{idx}.png', dpi=100, bbox_inches='tight')
            plt.close()
        
        print(f"Grad-CAM visualizations saved to {save_dir}/")
        return True
    except Exception as e:
        print(f"Grad-CAM visualization skipped: {e}")
        return False
