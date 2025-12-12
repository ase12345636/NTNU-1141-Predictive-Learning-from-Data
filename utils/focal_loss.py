import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Multi-label focal loss with logits.
    - alpha: tensor of shape [C] or scalar in [0,1] weighting positives vs negatives per class
      Applied as alpha * y + (1 - alpha) * (1 - y)
    - gamma: focusing parameter
    - reduction: 'none' | 'mean' | 'sum'
    """

    def __init__(self, alpha=None, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

        if alpha is None:
            self.alpha = None
        elif isinstance(alpha, (float, int)):
            self.alpha = float(alpha)
        else:
            self.register_buffer('alpha', alpha.float())

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # BCE with logits per-element
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        # Probabilities
        p = torch.sigmoid(logits)
        # pt = p if y=1 else 1-p
        pt = p * targets + (1.0 - p) * (1.0 - targets)
        focal = (1.0 - pt).pow(self.gamma)

        # Alpha factor: per-class or scalar
        if hasattr(self, 'alpha') and isinstance(self.alpha, torch.Tensor):
            # alpha_y = alpha for positives, (1 - alpha) for negatives
            alpha_factor = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        elif hasattr(self, 'alpha') and isinstance(self.alpha, float):
            a = self.alpha
            alpha_factor = a * targets + (1.0 - a) * (1.0 - targets)
        else:
            alpha_factor = 1.0

        loss = alpha_factor * focal * bce

        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        return loss
