"""
dataset.py — Custom Dataset and DataLoader factory for parasite egg classification.

Reads *_set.txt manifest files (one relative path per line, no extension).
Derives class label from the parent subfolder name.
Applies constrained augmentation (shearing x2 categories, rotation x3 categories).
"""
import os
import random
from pathlib import Path

from PIL import Image

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF

from config import Config


# ── Augmentation helpers ──────────────────────────────────────────────────────

class RandomShear:
    """
    Apply one of two shearing categories at random.
      Category 1 (mild)  : shear in range [-mild, +mild] degrees
      Category 2 (strong): shear in range [-strong, +strong] degrees
    Each image randomly gets ONE of the two categories applied.
    With 50% probability, the shear is applied at all.
    """
    def __init__(self, mild: float, strong: float):
        self.categories = [mild, strong]

    def __call__(self, img: Image.Image) -> Image.Image:
        shear_range = random.choice(self.categories)
        shear_angle = random.uniform(-shear_range, shear_range)
        return TF.affine(img, angle=0, translate=[0, 0], scale=1.0, shear=shear_angle)


class RandomCategoricalRotation:
    """
    Apply one of three rotation categories at random.
      Category 1: 90°
      Category 2: 180°
      Category 3: 270°
    """
    def __init__(self, cat1: int = 90, cat2: int = 180, cat3: int = 270):
        self.angles = [cat1, cat2, cat3]

    def __call__(self, img: Image.Image) -> Image.Image:
        angle = random.choice(self.angles)
        return TF.rotate(img, angle)


# ── Dataset ───────────────────────────────────────────────────────────────────

class ParasiteDataset(Dataset):
    """
    Loads parasite microscopy images using a manifest text file.

    Each line in the manifest is a relative path WITHOUT extension, e.g.:
        dataset/train/Hookworm/Hookworm egg_0010

    Label is derived from the parent-of-parent folder name:
        Ascaris_lumbricoides → 0
        Hookworm             → 1
        Trichuris_trichiura  → 2
    """

    def __init__(self, manifest: str, transform, config: Config):
        self.transform = transform
        self.class_to_idx = {cls: i for i, cls in enumerate(config.classes)}
        self.samples: list[tuple[str, int]] = []

        manifest_path = Path(manifest)
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest}")

        with open(manifest_path, "r", encoding="utf-8-sig") as f:
            for line in f:
                rel_path = line.strip()
                if not rel_path:
                    continue
                img_path = rel_path + ".jpg"
                # Extract class from second folder component: dataset/<split>/<class>/...
                parts = Path(rel_path).parts
                class_name = parts[2]   # dataset / split / CLASS / filename
                label = self.class_to_idx[class_name]
                self.samples.append((img_path, label))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label


# ── Transform builders ────────────────────────────────────────────────────────

def build_transforms(split: str, config: Config) -> T.Compose:
    """
    Returns the appropriate transform pipeline.

    Args:
        split : 'train', 'val', or 'test'
        config: Config object with augmentation parameters
    """
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if split == "train":
        return T.Compose([
            T.Resize(config.image_size + 32),               # slight oversize before crop
            T.CenterCrop(config.image_size),
            # ── Constrained augmentations ──────────────────
            # Shearing: 2 severity categories, applied randomly
            T.RandomApply([RandomShear(config.shear_mild, config.shear_strong)], p=config.aug_probability),
            # Rotation: 3 categorical angles, applied randomly
            T.RandomApply([RandomCategoricalRotation(
                config.rotate_cat1, config.rotate_cat2, config.rotate_cat3
            )], p=config.aug_probability),
            # ───────────────────────────────────────────────
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        # Val / Test: deterministic, no augmentation
        return T.Compose([
            T.Resize(config.image_size + 32),
            T.CenterCrop(config.image_size),
            T.ToTensor(),
            T.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])


# ── DataLoader factory ────────────────────────────────────────────────────────

def get_dataloaders(
    config: Config,
    debug: bool = False,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns (train_loader, val_loader, test_loader).

    Args:
        config : Config dataclass
        debug  : if True, uses tiny subset (2 batches) for smoke testing
    """
    train_ds = ParasiteDataset(config.train_list, build_transforms("train", config), config)
    val_ds   = ParasiteDataset(config.val_list,   build_transforms("val",   config), config)
    test_ds  = ParasiteDataset(config.test_list,  build_transforms("test",  config), config)

    if debug:
        # Tiny subsets for fast CPU smoke testing
        from torch.utils.data import Subset
        train_ds = Subset(train_ds, list(range(config.batch_size * 2)))
        val_ds   = Subset(val_ds,   list(range(config.batch_size * 2)))
        test_ds  = Subset(test_ds,  list(range(config.batch_size)))

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    train_loader = DataLoader(train_ds, shuffle=True,  drop_last=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, drop_last=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, drop_last=False, **loader_kwargs)

    return train_loader, val_loader, test_loader
