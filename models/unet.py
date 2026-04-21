"""U-Net family used for pixel-wise Cityscapes segmentation."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cbam import CBAM


class DoubleConv(nn.Module):
    """Apply the standard two-convolution block used throughout U-Net."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Encode features and keep a same-resolution skip tensor for decoding."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x: torch.Tensor):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return down_out, skip_out


class UpBlock(nn.Module):
    """Upsample decoder features and fuse them with the matching skip tensor."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.up_sample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, down_input: torch.Tensor, skip_input: torch.Tensor) -> torch.Tensor:
        x = self.up_sample(down_input)
        # Bilinear interpolation resolves any one-pixel mismatches introduced by
        # repeated pooling and transpose-convolution operations.
        if x.shape[2:] != skip_input.shape[2:]:
            x = F.interpolate(x, size=skip_input.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    """Encoder-decoder segmentation network with optional CBAM skip refinement."""

    def __init__(self, n_channels: int = 3, cls_classes: int = 20, use_cbam: bool = False):
        super().__init__()
        self.n_channels = n_channels
        self.cls_classes = cls_classes
        self.use_cbam = use_cbam

        # The initial block lifts the input image into the feature space used by
        # the rest of the encoder-decoder stack.
        self.inc = DoubleConv(n_channels, 64)

        # The encoder progressively increases channel capacity while reducing
        # spatial resolution to capture broader context.
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.down3 = DownBlock(256, 512)
        self.down4 = DownBlock(512, 1024)

        # The decoder restores spatial detail by combining coarse context with
        # high-resolution skip tensors from the encoder path.
        self.up1 = UpBlock(1024, 512, 512)
        self.up2 = UpBlock(512, 256, 256)
        self.up3 = UpBlock(256, 128, 128)
        self.up4 = UpBlock(128, 64, 64)

        # Optional CBAM modules recalibrate the skip features before they are
        # concatenated back into the decoder.
        if use_cbam:
            print("CBAM Activated!")
            self.cbam_skip1 = CBAM(64)    # skip from inc (x1)
            self.cbam_skip2 = CBAM(128)   # skip from down1
            self.cbam_skip3 = CBAM(256)   # skip from down2

        # A 1x1 convolution projects the final decoder tensor to per-class logits.
        self.outc = nn.Conv2d(64, cls_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Preserve intermediate tensors for skip connections before pooling
        # removes fine spatial detail.
        x1 = self.inc(x)

        x2, skip2 = self.down1(x1)
        x3, skip3 = self.down2(x2)
        x4, skip4 = self.down3(x3)
        x5, _     = self.down4(x4)

        # Decode from the bottleneck back to the original resolution.
        x = self.up1(x5, skip4)
        x = self.up2(x, self.cbam_skip3(skip3) if self.use_cbam else skip3)
        x = self.up3(x, self.cbam_skip2(skip2) if self.use_cbam else skip2)
        x = self.up4(x, self.cbam_skip1(x1)    if self.use_cbam else x1)

        return self.outc(x)


def load_UNet(n_channels: int = 3, cls_classes: int = 20, use_cbam: bool = False, pretrained_path: str = "", device: str = "cpu") -> UNet:
    """Instantiate U-Net and optionally restore a saved checkpoint."""
    model = UNet(n_channels=n_channels, cls_classes=cls_classes, use_cbam=use_cbam)
    if pretrained_path:
        state = torch.load(pretrained_path, map_location=device)
        model.load_state_dict(state["net"])
        print(f"Loaded weights from '{pretrained_path}'")
    return model
