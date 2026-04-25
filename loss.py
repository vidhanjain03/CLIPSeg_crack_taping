import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalDiceLoss(nn.Module):
    """
    Focal Loss   — down-weights easy negatives, focuses on hard crack edges.
    Dice Loss    — directly optimises spatial overlap (F1 in mask space).
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        inputs  = inputs.flatten()
        targets = targets.flatten()

        bce   = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt    = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        focal = focal.mean()

        probs = torch.sigmoid(inputs)
        inter = (probs * targets).sum()
        dice  = 1 - (2.0 * inter + 1e-6) / (probs.sum() + targets.sum() + 1e-6)

        return focal + dice