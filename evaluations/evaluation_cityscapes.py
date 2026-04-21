"""
Evaluation utilities: training-metrics charts and test-set inference.
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap, to_rgb
from matplotlib.ticker import FormatStrFormatter

from utils.datasets import get_transforms
from utils.metrics import iou_score

# The evaluation colormap mirrors the class-to-color mapping used by the
# application so qualitative outputs stay visually consistent across modules.
_classes = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
    19: "others",
}
_color_mapping = {
    "road": "#666666", "sidewalk": "#282828",
    "building": "#FF3232", "wall": "#6a329f", "fence": "#F110A6",
    "pole": "#523415", "traffic light": "#FFFF66", "traffic sign": "#FFFF00",
    "vegetation": "#008000", "terrain": "#6BAF6B",
    "sky": "#00b1ff",
    "person": "#E8BEAC", "rider": "#ac95e1",
    "car": "#FFA500", "truck": "#B07A15", "bus": "#101DCE",
    "train": "#3A2908", "motorcycle": "#E298D6", "bicycle": "#BEDFE5",
    "others": "#E5ACB6",
}
config_cmap = ListedColormap([to_rgb(_color_mapping[_classes[i]]) for i in sorted(_classes)])


def _base_chart(df: pd.DataFrame, y_cols: list, ylabel: str, title: str,
                path2save: str, scale: float = 1.0):
    """Render one or more CSV metric columns into a saved line chart."""
    x = df["epoch"].tolist()
    plt.figure(figsize=(20, 6))
    for col in y_cols:
        plt.plot(x, df[col] * scale, marker="o", label=col)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    ticks = list(x[::10])
    if x[-1] not in ticks:
        ticks.append(x[-1])
    plt.xticks(ticks)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path2save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved chart → {path2save}")


def plot_learning_rate(tracking_csv: str, path2save: str):
    """Plot the per-epoch learning rate recorded during training."""
    df = pd.read_csv(tracking_csv)
    x  = df["epoch"].tolist()
    plt.figure(figsize=(20, 6))
    plt.plot(x, df["learning_rate"], marker="o", label="learning_rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule")
    plt.legend()
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter("%.7f"))
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path2save, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved chart → {path2save}")


def plot_loss(tracking_csv: str, path2save: str):
    """Plot training and validation loss from the tracking CSV."""
    _base_chart(pd.read_csv(tracking_csv), ["train_loss", "val_loss"],
                "Loss", "Loss Curve", path2save)


def plot_pixel_accuracy(tracking_csv: str, path2save: str):
    """Plot training and validation pixel accuracy."""
    _base_chart(pd.read_csv(tracking_csv), ["train_pixel_acc", "val_pixel_acc"],
                "Pixel Accuracy (%)", "Pixel Accuracy", path2save, scale=100)


def plot_iou(tracking_csv: str, path2save: str):
    """Plot training and validation mean IoU."""
    _base_chart(pd.read_csv(tracking_csv), ["train_iou", "val_iou"],
                "IoU (%)", "IoU Score", path2save, scale=100)


def plot_dice(tracking_csv: str, path2save: str):
    """Plot training and validation Dice score."""
    _base_chart(pd.read_csv(tracking_csv), ["train_dice", "val_dice"],
                "Dice (%)", "Dice Score", path2save, scale=100)


def plot_all_metrics(tracking_csv: str, save_dir: str):
    """Generate the standard chart set for one completed training run."""
    os.makedirs(save_dir, exist_ok=True)
    plot_learning_rate(tracking_csv, os.path.join(save_dir, "lr.png"))
    plot_loss(tracking_csv,          os.path.join(save_dir, "loss.png"))
    plot_pixel_accuracy(tracking_csv, os.path.join(save_dir, "pixel_acc.png"))
    plot_iou(tracking_csv,           os.path.join(save_dir, "iou.png"))
    plot_dice(tracking_csv,          os.path.join(save_dir, "dice.png"))


def run_test_evaluation(
    configs: dict,
    model: torch.nn.Module,
    device: torch.device,
    n_samples: int = 10,
    save_dir: str = None,
):
    """
    Sample n_samples from the test split, run inference, visualize results,
    and print mean IoU.
    """
    # The project stores test samples as the held-out tail of the validation
    # directory, matching the split used in the dataset loader.
    val_tf = get_transforms(configs["image_size"], "val")
    val_image_dir = os.path.join(configs["cityscape_path"], "val", "image")
    all_images    = sorted(os.listdir(val_image_dir))
    test_images   = all_images[round(0.8 * len(all_images)):]
    selected      = random.sample(test_images, min(n_samples, len(test_images)))

    pred_masks, iou_list = [], []
    model.eval()
    with torch.no_grad():
        for img_name in selected:
            img_path  = os.path.join(val_image_dir, img_name)
            mask_path = img_path.replace("/val/image/", "/val/label/")

            image = np.load(img_path).astype(np.float32)
            mask  = np.load(mask_path).astype(np.int64)
            mask  = np.where(mask == -1, 19, mask)

            aug     = val_tf(image=image, mask=mask)
            t_image = aug["image"].unsqueeze(0).to(device)
            t_mask  = aug["mask"].long().unsqueeze(0).to(device)

            # The same forward path used during validation is reused here so
            # qualitative plots and reported metrics stay aligned.
            output = model(t_image)
            pred   = torch.argmax(output, dim=1)
            pred_masks.append(pred.cpu())
            m_iou, fw_iou, _ = iou_score(output, t_mask)
            iou_list.append((m_iou, fw_iou))

    mean_iou = sum(v[0] for v in iou_list) / len(iou_list)
    mean_fw_iou = sum(v[1] for v in iou_list) / len(iou_list)
    print(f"Mean IoU over {len(selected)} samples: {mean_iou * 100:.2f}%")
    print(f"FW IoU  over {len(selected)} samples: {mean_fw_iou * 100:.2f}%")

    # Save one qualitative grid containing the image, ground-truth mask, and
    # predicted mask for each sampled test example.
    n = len(selected)
    plt.figure(figsize=(15, 4 * n))
    for i, img_name in enumerate(selected):
        img_path  = os.path.join(val_image_dir, img_name)
        mask_path = img_path.replace("/val/image/", "/val/label/")

        image = np.load(img_path).astype(np.float32)
        mask  = np.load(mask_path).astype(np.int64)
        mask  = np.where(mask == -1, 19, mask)

        aug = val_tf(image=image, mask=mask)

        plt.subplot(n, 3, i * 3 + 1)
        plt.imshow(aug["image"].permute(1, 2, 0))
        plt.title("Image")
        plt.axis("off")

        plt.subplot(n, 3, i * 3 + 2)
        plt.imshow(aug["mask"], cmap=config_cmap, vmin=0, vmax=19)
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(n, 3, i * 3 + 3)
        plt.imshow(pred_masks[i].squeeze(0), cmap=config_cmap, vmin=0, vmax=19)
        plt.title(f"Prediction (IoU={iou_list[i][0]*100:.2f}% | FW={iou_list[i][1]*100:.2f}%)")
        plt.axis("off")

    plt.tight_layout()
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out = os.path.join(save_dir, "test_predictions.png")
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved predictions → {out}")
    else:
        plt.show()
    plt.close()

    return mean_iou
