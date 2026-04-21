"""Epoch-level training loop, checkpointing, and metric logging."""

import csv

import torch
import tqdm
from torch.utils.data import DataLoader

from utils.losses import DiceLoss
from utils.metrics import pixel_accuracy, iou_score, dice_score
from utils.trainingStrategies import build_optimizer, build_lr_scheduler


class UNet_Trainer:
    """Coordinate optimization, validation, checkpointing, and final testing."""

    def __init__(self, configs: dict, model: torch.nn.Module, device: torch.device):
        self.configs = configs
        self.device = device
        self.model = model.to(device)
        self.val_best_iou = 0.0

        # Loss, optimizer, and scheduler are built once and reused across the
        # full training run.
        self.loss_fn     = DiceLoss()
        self.optimizer   = build_optimizer(model, configs["lr"])
        self.scheduler   = build_lr_scheduler(self.optimizer, configs["plateau_patience"], configs["min_lr"])

        # The CSV header mirrors the scalar metrics printed each epoch together
        # with class-wise IoU for both training and validation.
        num_classes = configs["cls_classes"] + 1
        cls_cols    = [f"cls{c}" for c in range(num_classes)]
        header = (
            ["epoch"]
            + ["train_loss", "train_pixel_acc", "train_iou", "train_fw_iou", "train_dice"]
            + [f"train_iou_{c}" for c in cls_cols]
            + ["val_loss", "val_pixel_acc", "val_iou", "val_fw_iou", "val_dice"]
            + [f"val_iou_{c}" for c in cls_cols]
            + ["learning_rate"]
        )
        self.num_classes = num_classes
        with open(configs["tracking_csv"], mode="w", newline="") as f:
            csv.writer(f).writerow(header)
        print("Tracking CSV created.")

    def _run_epoch(self, loader: DataLoader, epoch: int, train: bool) -> dict:
        """Execute one pass over a dataloader and aggregate scalar metrics."""
        self.model.train() if train else self.model.eval()

        metrics = {"loss": 0.0, "pixel_acc": 0.0, "iou": 0.0, "fw_iou": 0.0, "dice": 0.0}
        cls_iou_sum = [0.0] * self.num_classes
        cls_iou_cnt = [0]   * self.num_classes
        total = 0
        desc   = f"[{'Train' if train else 'Val  '}] Epoch {epoch}"
        colour = "blue" if train else "green"

        # Gradient computation is enabled only during training; validation runs
        # through the same tensor path under no_grad for metric collection.
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for images, masks in tqdm.tqdm(loader, desc=desc, colour=colour):
                images = images.to(self.device)
                masks  = masks.long().to(self.device)

                if train:
                    self.optimizer.zero_grad()

                outputs = self.model(images)
                loss    = self.loss_fn(outputs, masks)

                if train:
                    # Each training iteration follows the standard
                    # forward-loss-backward-update sequence.
                    loss.backward()
                    self.optimizer.step()

                # Batch metrics are accumulated as weighted sums so the final
                # averages remain correct even when the last batch is smaller.
                bs = images.size(0)
                total += bs
                mean_iou, fw_iou, cls_ious = iou_score(outputs, masks)
                metrics["loss"]      += loss.item() * bs
                metrics["pixel_acc"] += pixel_accuracy(outputs, masks) * bs
                metrics["iou"]       += mean_iou * bs
                metrics["fw_iou"]    += fw_iou * bs
                metrics["dice"]      += dice_score(outputs, masks) * bs
                for c, v in cls_ious.items():
                    cls_iou_sum[c] += v * bs
                    cls_iou_cnt[c] += bs

                if self.configs["isDebug"] == 1:
                    break

        # Convert the accumulated weighted sums into epoch-level averages.
        for k in metrics:
            metrics[k] /= total
        metrics["cls_iou"] = [
            cls_iou_sum[c] / cls_iou_cnt[c] if cls_iou_cnt[c] > 0 else 0.0
            for c in range(self.num_classes)
        ]
        return metrics

    def train_one_epoch(self, loader: DataLoader, epoch: int) -> dict:
        """Run one training epoch."""
        return self._run_epoch(loader, epoch, train=True)

    def evaluate(self, loader: DataLoader, epoch: int) -> dict:
        """Run one evaluation epoch without parameter updates."""
        return self._run_epoch(loader, epoch, train=False)

    def run(self, train_loader: DataLoader, val_loader: DataLoader, test_loader: DataLoader):
        """Run training until early stopping, then report held-out test metrics."""
        stop_flag = 0

        for epoch in range(1, self.configs["max_epoch_num"] + 1):
            train_m = self.train_one_epoch(train_loader, epoch)
            val_m   = self.evaluate(val_loader, epoch)

            # The scheduler watches validation IoU because the project selects
            # checkpoints by the same criterion.
            self.scheduler.step(val_m["iou"])
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:03d} | "
                f"Train loss={train_m['loss']:.4f} acc={train_m['pixel_acc']*100:.2f}% "
                f"iou={train_m['iou']*100:.2f}% fw_iou={train_m['fw_iou']*100:.2f}% "
                f"dice={train_m['dice']*100:.2f}%"
            )
            print(
                f"         | "
                f"Val   loss={val_m['loss']:.4f} acc={val_m['pixel_acc']*100:.2f}% "
                f"iou={val_m['iou']*100:.2f}% fw_iou={val_m['fw_iou']*100:.2f}% "
                f"dice={val_m['dice']*100:.2f}%  lr={lr:.7f}"
            )

            # Checkpoints store the model state together with enough run state
            # to recover the best validation epoch later.
            if val_m["iou"] > self.val_best_iou:
                self.val_best_iou = val_m["iou"]
                stop_flag = 0
                torch.save(
                    {
                        **self.configs,
                        "net": self.model.state_dict(),
                        "val_best_iou": self.val_best_iou,
                        "optimizer": self.optimizer.state_dict(),
                        "epoch": epoch,
                    },
                    self.configs["weight_saved_path"],
                )
                print(f"  => Best model saved (iou={self.val_best_iou*100:.2f}%)")
            else:
                stop_flag += 1

            # Append one row per epoch so downstream charting utilities can
            # read the full training history without recomputation.
            with open(self.configs["tracking_csv"], mode="a", newline="") as f:
                csv.writer(f).writerow(
                    [epoch]
                    + [train_m["loss"], train_m["pixel_acc"], train_m["iou"], train_m["fw_iou"], train_m["dice"]]
                    + train_m["cls_iou"]
                    + [val_m["loss"], val_m["pixel_acc"], val_m["iou"], val_m["fw_iou"], val_m["dice"]]
                    + val_m["cls_iou"]
                    + [lr]
                )

            # Early stopping counts consecutive non-improving validation epochs.
            if stop_flag >= self.configs["earlyStopping"]:
                print(f"Early stopping at epoch {epoch}.")
                break

        print("\n--- Test Evaluation ---")
        # The final test pass uses the same evaluation path as validation so
        # metric definitions remain consistent across splits.
        test_m = self.evaluate(test_loader, 0)
        print(
            f"Test: loss={test_m['loss']:.4f} acc={test_m['pixel_acc']*100:.2f}% "
            f"iou={test_m['iou']*100:.2f}% fw_iou={test_m['fw_iou']*100:.2f}% "
            f"dice={test_m['dice']*100:.2f}%"
        )
        return test_m
