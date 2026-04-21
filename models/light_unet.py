"""Lightweight LR-ASPP-based segmentation model used for comparison."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.segmentation import (
    lraspp_mobilenet_v3_large,
    LRASPP_MobileNet_V3_Large_Weights,
)


class LightSeg(nn.Module):
    """LR-ASPP model with a MobileNetV3-Large feature extractor."""

    def __init__(self, num_classes: int = 20, pretrained_backbone: bool = True):
        super().__init__()
        weights = LRASPP_MobileNet_V3_Large_Weights.DEFAULT if pretrained_backbone else None
        self.model = lraspp_mobilenet_v3_large(weights=None)
        # Replace both LR-ASPP prediction heads so they emit the Cityscapes
        # class count used throughout the project.
        low_in = self.model.classifier.low_classifier.in_channels
        high_in = self.model.classifier.high_classifier.in_channels
        self.model.classifier.low_classifier = nn.Conv2d(low_in, num_classes, kernel_size=1)
        self.model.classifier.high_classifier = nn.Conv2d(high_in, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def load_LightSeg(num_classes: int = 20, pretrained_backbone: bool = True,
                  pretrained_path: str = "", device: str = "cpu") -> LightSeg:
    """Instantiate LightSeg and optionally load checkpoint weights."""
    model = LightSeg(num_classes=num_classes, pretrained_backbone=pretrained_backbone)
    if pretrained_path:
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state["net"])
        print(f"Loaded weights from '{pretrained_path}'")
    return model
