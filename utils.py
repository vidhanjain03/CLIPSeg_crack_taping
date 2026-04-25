import random
import numpy as np
import torch

def set_seed(seed: int):
    """Sets the seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def calculate_metrics(pred, target):
    """
    Returns (iou, dice) or (None, None) when both masks are empty.
    Skips pairs where both prediction and ground-truth are blank.
    """
    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection

    if union == 0 and target.sum() == 0:
        return None, None

    iou  = (intersection + 1e-6) / (union + 1e-6)
    dice = (2.0 * intersection + 1e-6) / (pred.sum() + target.sum() + 1e-6)
    return float(iou), float(dice)