"""Loss functions used during semantic-segmentation training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Dice loss over the full class set for multi-class segmentation."""

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute one minus the mean Dice score across classes.

        Args:
            preds:   (B, C, H, W) raw logits before softmax.
            targets: (B, H, W) integer class labels.
        """
        C = preds.shape[1]
        # Convert logits into per-class probabilities for soft overlap.
        preds = F.softmax(preds, dim=1)
        # Expand integer labels into one-hot channels so they align with the
        # prediction tensor during class-wise overlap computation.
        targets_one_hot = F.one_hot(targets, num_classes=C).permute(0, 3, 1, 2).float()

        intersection = (preds * targets_one_hot).sum(dim=(0, 2, 3))
        union = preds.sum(dim=(0, 2, 3)) + targets_one_hot.sum(dim=(0, 2, 3))

        # Average the Dice score across classes, then convert similarity into
        # a minimization objective for optimization.
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()
