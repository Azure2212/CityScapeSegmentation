"""Shared constants and baseline local inference helpers.

This module predates the Flask app and still provides the canonical class map,
colormap, device selection, and a minimal single-image inference path used by
the repository's standalone utilities.
"""

import sys
import io
import os
import numpy as np
import cv2
from scipy import ndimage
import torch
import gdown
from matplotlib.colors import ListedColormap, to_rgb
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, os.path.dirname(__file__))
from models import load_UNet

# The class map defines the semantic labels used across training, evaluation,
# visualization, and the live inference application.
CLASSES = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
    19: "others"
}

COLOR_MAPPING = {
    "road": "#666666", "sidewalk": "#282828",
    "building": "#FF3232", "wall": "#6a329f", "fence": "#FFC1C1",
    "pole": "#523415", "traffic light": "#FFFF66", "traffic sign": "#FFFF00",
    "vegetation": "#008000", "terrain": "#6BAF6B",
    "sky": "#00b1ff",
    "person": "#E8BEAC", "rider": "#beace8",
    "car": "#FFA500", "truck": "#FFF6E5", "bus": "#E5FFF6",
    "train": "#FFE5EE", "motorcycle": "#FFE5FB", "bicycle": "#E5FBFF",
    "others": "#E5ACB6"
}

# The colormap stays aligned with the numeric class IDs so mask visualization
# remains consistent anywhere these constants are reused.
COLORS = [to_rgb(COLOR_MAPPING[CLASSES[i]]) for i in sorted(CLASSES)]
CONFIG_CMAP = ListedColormap(COLORS)

IMAGE_SIZE  = 224
NUM_CLASSES = len(CLASSES)  # 20

WEIGHT_URL = "https://drive.google.com/uc?export=download&id=1LMc5DzE3cPtRx5_SyjTkpYP6vp1pV4Gg"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEST_TF = A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    ToTensorV2()
])

def load_model_from_url(url: str = WEIGHT_URL) -> torch.nn.Module:
    """Download a pretrained U-Net checkpoint and return an evaluation model."""
    print("Downloading weights from Google Drive ...")
    buffer = io.BytesIO()
    gdown.download(url, buffer, quiet=False, fuzzy=True)
    buffer.seek(0)

    state = torch.load(buffer, map_location=str(DEVICE))
    model = load_UNet(n_channels=3, cls_classes=NUM_CLASSES)
    model.load_state_dict(state["net"])
    model.to(DEVICE)
    model.eval()
    print(f"Model loaded on device: {DEVICE}")
    return model


def segmentation_prediction(model, img_path: str) -> torch.Tensor:
    """Run the baseline single-image prediction path used by local utilities."""
    model.eval()
    with torch.no_grad():
        img_path = os.path.abspath(img_path)
        image = cv2.imread(img_path)
        # The training and inference pipeline expects normalized RGB input at
        # the fixed model resolution.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        transformed = TEST_TF(image=image)
        tensor = transformed["image"].unsqueeze(0).to(DEVICE)  # (1, C, H, W)
        output = model(tensor)                                  # (1, num_classes, H, W)
        pred_mask = torch.argmax(output, dim=1)                # (1, H, W)
    return pred_mask


def countObject(pred_mask, class_ids: list = [13], min_area: int = 50):
    """Approximate object counts by connected components in one class mask."""

    count, labeled_mask = 0, None
    lines = []
    for class_id in class_ids:
        # Each connected component is treated as one approximate object region
        # after small blobs below the minimum area threshold are discarded.
        mask = (pred_mask.squeeze(0).cpu().numpy() == class_id).astype(np.uint8)
        labeled_mask, num_objects = ndimage.label(mask)
        count = 0
        for i in range(1, num_objects + 1):
            if np.sum(labeled_mask == i) >= min_area:
                count += 1
        line = f"Class '{CLASSES[class_id]}' (id={class_id}): {count} object(s) found (min_area={min_area})"
        print(line)
        lines.append(line)
    return count, labeled_mask, lines

def run(img_path: str):
    """Execute the standalone local inference flow on one image path."""
    # Load the pretrained model checkpoint into memory.
    model = load_model_from_url()

    # Predict the semantic segmentation mask.
    pred_mask = segmentation_prediction(model, img_path)  # (1, H, W)
    print(f"Prediction shape: {pred_mask.shape}")

    # Derive connected-component counts for the selected review classes.
    count, labeled_mask, lines = countObject(pred_mask, class_ids=[11, 13], min_area=25)

    return pred_mask, count, labeled_mask, lines


if __name__ == "__main__":
    IMG_PATH = r"C:\Users\Dasan\OneDrive\Desktop\cityScapeSegmentationWebsite\2798.png"
    run(IMG_PATH)
