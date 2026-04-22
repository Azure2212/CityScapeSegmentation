"""DeepLabV3 wrapper used by the training and inference entrypoints."""

import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights


class DeepLabV3(nn.Module):
    """DeepLabV3+ with a pretrained ResNet101 backbone (COCO weights)."""

    def __init__(self, num_classes: int = 20, pretrained_backbone: bool = True):
        super().__init__()
        weights = DeepLabV3_ResNet101_Weights.DEFAULT if pretrained_backbone else None
        self.model = deeplabv3_resnet101(weights=weights)
        # Replace the default classifier heads so the network emits logits for
        # the project-specific class count instead of the torchvision default.
        self.model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
        if self.model.aux_classifier is not None:
            self.model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def load_DeepLabV3(num_classes: int = 20, pretrained_backbone: bool = True,
                   pretrained_path: str = "", device: str = "cpu") -> DeepLabV3:
    """Instantiate DeepLabV3 and optionally restore checkpoint weights."""
    model = DeepLabV3(num_classes=num_classes, pretrained_backbone=pretrained_backbone)

    if pretrained_path:
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state["net"])
        print(f"Pretrained '{pretrained_path}' model loaded successfully!")

    return model
