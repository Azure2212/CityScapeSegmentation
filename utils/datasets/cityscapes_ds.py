"""Dataset and dataloader helpers for the preprocessed Cityscapes tensors."""

import os

import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader


def get_transforms(image_size: int, split: str) -> A.Compose:
    """Build the split-specific augmentation pipeline."""
    if split == "train":
        return A.Compose(
            [
                # Training uses a lightweight augmentation set so every image
                # is standardized in size and may be mirrored horizontally.
                A.Resize(image_size, image_size),
                A.HorizontalFlip(p=0.5),
                ToTensorV2(),
            ],
            additional_targets={"mask": "mask"},
        )
    else:
        return A.Compose(
            [
                A.Resize(image_size, image_size),
                ToTensorV2(),
            ],
            additional_targets={"mask": "mask"},
        )


class CityScapeDataset(Dataset):
    """Load preprocessed `.npy` image and mask pairs for one dataset split."""

    def __init__(self, configs: dict, data_type: str, transform: A.Compose):
        """Resolve the file list for the requested split."""
        self.transform = transform
        self.configs = configs
        # The dataset stores the held-out test subset inside the validation
        # folder, so `test` reuses the `val` directory and slices it later.
        self.folder = "val" if data_type == "test" else data_type

        data_path = os.path.join(configs["cityscape_path"], self.folder)
        all_images = sorted(os.listdir(os.path.join(data_path, "image")))
        image_paths = [os.path.join(data_path, "image", img) for img in all_images]

        if data_type == "train":
            self.image_paths = image_paths
        elif data_type == "val":
            # Validation uses the first 80% of the validation directory.
            self.image_paths = image_paths[: round(0.8 * len(image_paths))]
        elif data_type == "test":
            # Test uses the remaining 20% so the project can hold out samples
            # without requiring a separate on-disk test directory.
            self.image_paths = image_paths[round(0.8 * len(image_paths)) :]
        else:
            raise ValueError(f"data_type must be 'train', 'val', or 'test', got '{data_type}'")

        print(f"Loaded {data_type} set: {len(self.image_paths)} images")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        """Load one image-mask pair and apply the split-specific transform."""
        img_path = self.image_paths[idx]
        mask_path = img_path.replace(
            f"/data/{self.folder}/image/", f"/data/{self.folder}/label/"
        )

        image = np.load(img_path).astype(np.float32)
        mask = np.load(mask_path).astype(np.int64)
        # The preprocessing pipeline stores ignored pixels as -1; the project
        # remaps them into the explicit "others" class index.
        mask = np.where(mask == -1, 19, mask)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug["image"]
            mask = aug["mask"]

        return image, mask


def build_dataloaders(configs: dict):
    """Construct the train, validation, and test dataloaders used by training."""
    train_tf = get_transforms(configs["image_size"], "train")
    val_tf   = get_transforms(configs["image_size"], "val")

    train_data = CityScapeDataset(configs, "train", train_tf)
    val_data   = CityScapeDataset(configs, "val",   val_tf)
    test_data  = CityScapeDataset(configs, "test",  val_tf)

    train_loader = DataLoader(train_data, batch_size=configs["batch_size"],
                              num_workers=configs["num_workers"], shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=configs["batch_size"],
                              num_workers=configs["num_workers"], shuffle=False)
    test_loader  = DataLoader(test_data,  batch_size=configs["batch_size"],
                              num_workers=configs["num_workers"], shuffle=False)

    return train_loader, val_loader, test_loader
