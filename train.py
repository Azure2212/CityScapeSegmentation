import json
import os
import random
import sys

import numpy as np
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from models import load_UNet, load_FCN, load_SwinV2B, load_LightSeg, load_DeepLabV3, load_UNetLibrary
from trainer import UNet_Trainer
from utils.datasets import build_dataloaders
from evaluations import plot_all_metrics, run_test_evaluation

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

classes = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
    19: "others",
}

import argparse

MODEL_CHOICES = ["UNet", "UNet_CBAM", "FCN", "SwinV2B", "LightSeg", "YOLOv11", "DeepLabV3"]

def parse_args():
    parser = argparse.ArgumentParser(description="Train segmentation model on Cityscapes")
    parser.add_argument("--config",          default="configs.json",  help="Path to JSON config file")
    parser.add_argument("--model",           default="UNet",          choices=MODEL_CHOICES, help="Model architecture")
    parser.add_argument("--cityscape_path",  default=None,            help="Override cityscape data root path")
    parser.add_argument("--rs_dir",          default=None,            help="Override results output directory")
    parser.add_argument("--image_size",      type=int, default=None,  help="Override input image size")
    parser.add_argument("--batch_size",      type=int, default=None,  help="Override batch size")
    parser.add_argument("--epochs",          type=int, default=None,  help="Override max_epoch_num")
    parser.add_argument("--debug",           action="store_true",     help="Single-batch debug mode")
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config) as f:
        configs = json.load(f)

    # CLI overrides (applied before deriving paths)
    if args.cityscape_path is not None:
        configs["cityscape_path"] = args.cityscape_path
    if args.rs_dir is not None:
        configs["rs_dir"] = args.rs_dir
    if args.image_size is not None:
        configs["image_size"] = args.image_size
    if args.batch_size is not None:
        configs["batch_size"] = args.batch_size
    if args.epochs is not None:
        configs["max_epoch_num"] = args.epochs
    if args.debug:
        configs["isDebug"] = 1

    # Runtime-derived paths
    os.makedirs(configs["rs_dir"], exist_ok=True)
    configs["tracking_csv"]      = os.path.join(configs["rs_dir"], "trainingTracking.csv")
    configs["weight_saved_path"] = os.path.join(configs["rs_dir"], f"{args.model}_Cityscapes.pt")

    # Save resolved config to results dir
    with open(os.path.join(configs["rs_dir"], "configs.json"), "w") as f:
        json.dump(configs, f, indent=4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"Config : {configs}")

    # Data
    train_loader, val_loader, test_loader = build_dataloaders(configs)

    # Model
    num_classes = configs["cls_classes"] + 1    # +1 for "others" class
    n_ch        = configs["n_channels"]

    if args.model == "UNet":
        model = load_UNet(n_channels=n_ch, cls_classes=num_classes)
    elif args.model == "UNet_CBAM":
        model = load_UNet(n_channels=n_ch, cls_classes=num_classes, use_cbam=True)
    elif args.model == "FCN":
        model = load_FCN(num_classes=num_classes)
    elif args.model == "SwinV2B":
        model = load_SwinV2B(num_classes=num_classes, n_channels=n_ch)
    elif args.model == "LightSeg":
        model = load_LightSeg(num_classes=num_classes, pretrained_backbone=True)
    elif args.model == "UNet_Lib":
        model = load_UNetLibrary(num_classes=num_classes)
    elif args.model == "DeepLabV3":
        model = load_DeepLabV3(num_classes=num_classes, pretrained_backbone=True)

    print(f"Model  : {args.model}  |  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trainer = UNet_Trainer(configs=configs, model=model, device=device)
    trainer.run(train_loader, val_loader, test_loader)

    # Post-training: charts + visual test evaluation
    plot_all_metrics(configs["tracking_csv"], save_dir=os.path.join(configs["rs_dir"], "charts"))
    run_test_evaluation(configs, model, device, n_samples=10,
                        save_dir=os.path.join(configs["rs_dir"], "predictions"))


if __name__ == "__main__":
    main()

#/data2/atran16/Anaconda_ForTrain/bin/python train.py --model UNet --batch_size 16 --epochs 300 --cityscape_path /data2/atran16/CityScapeSegmentationProject/datasets/data --rs_dir /data2/atran16/CityScapeSegmentationProject/trained_results/unet_run_results
# xcopy "C:\Users\Dasan\OneDrive\Desktop\cityScapeSegmentationWebsite\CityScapeSegmentation" "C:\Users\Dasan\OneDrive\Desktop\cityScapeSegmentationWebsite\CityScapeSegmentationWebsite" /E /H /Y