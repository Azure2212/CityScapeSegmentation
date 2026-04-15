import torch
import torch.nn as nn
from torchvision.models.segmentation import fcn_resnet50, fcn_resnet101


class FCN(nn.Module):
    def __init__(self, num_classes: int = 20, backbone: str = "resnet50"):
        super().__init__()
        if backbone == "resnet101":
            self.model = fcn_resnet101(weights=None, num_classes=num_classes)
        else:
            self.model = fcn_resnet50(weights=None, num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def load_FCN(num_classes: int = 20, backbone: str = "resnet50",
             pretrained_path: str = "", device: str = "cpu") -> FCN:
    model = FCN(num_classes=num_classes, backbone=backbone)

    if pretrained_path:
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state["net"])
        print(f"Pretrained '{pretrained_path}' model loaded successfully!")

    return model
