"""
Asymmetric Loss for Multi-Label Image Classification
Reference: "Asymmetric Loss For Multi-Label Classification" (2021)
https://arxiv.org/abs/2009.14119
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.
    
    Args:
        gamma_neg: negative focusing parameter (default: 4)
        gamma_pos: positive focusing parameter (default: 1)
        clip: gradient clipping for numerical stability (default: 0.05)
        eps: small value for numerical stability (default: 1e-8)
        disable_torch_grad_focal_loss: disable gradient for focal term (default: True)
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-8,
        disable_torch_grad_focal_loss: bool = True,
    ):
        super(AsymmetricLoss, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss

    def forward(self, x, y):
        """
        Args:
            x: model logits, shape (B, C) where C is number of classes
            y: target labels, shape (B, C) with 0/1 values
        
        Returns:
            loss: scalar tensor
        """
        # Sigmoid activation to get probabilities
        p = torch.sigmoid(x)
        
        # Asymmetric clipping for numerical stability
        p_clipped = torch.clamp(p, min=self.clip, max=1 - self.clip)
        
        # Log probabilities for numerical stability
        log_p = torch.log(p_clipped)
        log_1_p = torch.log(1 - p_clipped)
        
        # Asymmetric loss computation
        # For positive samples: focus on hard negatives more
        # For negative samples: focus on hard positives more
        
        # Positive class loss
        pos_loss = -(y * log_p * torch.pow(1 - p, self.gamma_pos))
        
        # Negative class loss
        neg_loss = -(
            (1 - y) * log_1_p * torch.pow(p, self.gamma_neg)
        )
        
        # Combine
        loss = pos_loss + neg_loss
        
        return loss.mean()


class AsymmetricLossOptimized(nn.Module):
    """
    Optimized version of Asymmetric Loss with better numerical stability.
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        eps: float = 1e-6,
    ):
        super(AsymmetricLossOptimized, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps

    def forward(self, x, y):
        """
        Args:
            x: model logits, shape (B, C)
            y: target labels, shape (B, C) with 0/1 values
        
        Returns:
            loss: scalar tensor
        """
        # Clamp logits for numerical stability
        x_clipped = torch.clamp(x, min=-20, max=20)
        
        # Compute sigmoid and its complement
        sigmoid_x = torch.sigmoid(x_clipped)
        
        # Asymmetric loss: handle positive and negative differently
        # For positive samples (y=1): use high gamma_neg to focus on hard negatives
        # For negative samples (y=0): use high gamma_pos to focus on hard positives
        
        # Using log-sum-exp trick for stability
        pos_weight = y * (1 - sigmoid_x) ** self.gamma_pos
        neg_weight = (1 - y) * sigmoid_x ** self.gamma_neg
        
        # Weighted cross-entropy
        loss = F.binary_cross_entropy_with_logits(
            x, y, reduction='none'
        )
        
        # Apply asymmetric weighting
        loss = (pos_weight + neg_weight) * loss
        
        return loss.mean()


class AsymmetricLossWithBalance(nn.Module):
    """
    Asymmetric Loss with per-class balancing weights.
    
    Args:
        pos_weight: per-class positive weights (1D tensor of shape (num_classes,))
        gamma_neg: negative focusing parameter
        gamma_pos: positive focusing parameter
        clip: gradient clipping value
    """

    def __init__(
        self,
        pos_weight: torch.Tensor = None,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
    ):
        super(AsymmetricLossWithBalance, self).__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, x, y):
        """
        Args:
            x: logits (B, C)
            y: labels (B, C)
        
        Returns:
            weighted asymmetric loss
        """
        # Clamp for stability
        x_clipped = torch.clamp(x, min=-20, max=20)
        
        # Sigmoid
        sigmoid_x = torch.sigmoid(x_clipped)
        
        # Base asymmetric weights
        pos_weight_term = y * (1 - sigmoid_x) ** self.gamma_pos
        neg_weight_term = (1 - y) * sigmoid_x ** self.gamma_neg
        
        # BCE loss
        loss = F.binary_cross_entropy_with_logits(
            x, y, reduction='none'
        )
        
        # Apply asymmetric weighting
        weighted_loss = (pos_weight_term + neg_weight_term) * loss
        
        # Apply per-class balancing if available
        if self.pos_weight is not None:
            weighted_loss = weighted_loss * self.pos_weight.unsqueeze(0)
        
        return weighted_loss.mean()
