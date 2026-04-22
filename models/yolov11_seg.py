"""Project-local interface stub for the YOLOv11 segmentation entrypoint."""

import torch
import torch.nn as nn


class YOLOv11Seg(nn.Module):
    """Stub module kept so the model registry can expose a consistent interface."""

    def __init__(self, num_classes: int = 20):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x):
        """Signal that the architecture is not implemented in this repository."""
        raise NotImplementedError("YOLOv11Seg architecture not yet implemented.")


def load_YOLOv11Seg(num_classes: int = 20, pretrained_path: str = "", device: str = "cpu") -> YOLOv11Seg:
    """Instantiate the YOLOv11Seg stub and optionally load checkpoint weights."""
    model = YOLOv11Seg(num_classes=num_classes)
    if pretrained_path:
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state["net"])
    return model
