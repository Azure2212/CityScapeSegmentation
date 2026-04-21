"""SwinV2-based segmentation head used in comparison experiments."""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class SwinV2B(nn.Module):
    """Transformer encoder with a lightweight convolutional segmentation head."""

    def __init__(self, num_classes: int = 20, n_channels: int = 3):
        super().__init__()
        # timm returns a multi-stage feature extractor; the project uses the
        # deepest stage and adds a compact projection head for class logits.
        self.encoder = timm.create_model(
            "swinv2_base_window8_256",
            pretrained=True,
            features_only=True,
            in_chans=n_channels,
        )
        last_ch = self.encoder.feature_info.channels()[-1]   # 1024 for base

        self.head = nn.Sequential(
            nn.Conv2d(last_ch, 256, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def _to_bchw(self, t: torch.Tensor) -> torch.Tensor:
        """Normalize encoder outputs to the channel-first layout expected by the head."""
        if t.ndim == 4 and t.shape[-1] < t.shape[1]:
            t = t.permute(0, 3, 1, 2).contiguous()
        return t

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The head predicts on the deepest feature map, then upsamples the
        # logits back to the original image resolution.
        H, W   = x.shape[2:]
        feats  = self.encoder(x)
        last   = self._to_bchw(feats[-1])
        out    = self.head(last)
        return F.interpolate(out, size=(H, W), mode="bilinear", align_corners=False)


def load_SwinV2B(num_classes: int = 20, n_channels: int = 3,
                 pretrained_path: str = "", device: str = "cpu") -> SwinV2B:
    """Instantiate SwinV2B and optionally restore checkpoint weights."""
    model = SwinV2B(num_classes=num_classes, n_channels=n_channels)

    if pretrained_path:
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state["net"])
        print(f"Pretrained '{pretrained_path}' model loaded successfully!")

    return model
