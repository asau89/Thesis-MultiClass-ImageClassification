"""
config.py — Centralized hyperparameters and paths for the ConvNeXt classifier.
All training scripts import from here; change values here to affect everything.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Dataset ──────────────────────────────────────────────────────────────
    dataset_root: str = "dataset"
    train_list: str = "train_set.txt"
    val_list:   str = "val_set.txt"
    test_list:  str = "test_set.txt"
    classes: list = field(default_factory=lambda: [
        "Ascaris_lumbricoides",
        "Hookworm",
        "Trichuris_trichiura",
    ])
    num_classes: int = 3

    # ── Model ────────────────────────────────────────────────────────────────
    # convnext_base maximises RTX 5060 Ti 16 GB at 384×384
    model_name: str = "convnext_base"
    image_size: int = 384
    pretrained: bool = True

    # ── Training ─────────────────────────────────────────────────────────────
    # Batch 16 comfortably fits 16 GB VRAM with AMP at 384×384
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-2
    num_workers: int = 6          # R7 7700 has 8 cores; leave 2 for OS
    pin_memory: bool = True       # Faster CPU→GPU transfer

    # Mixed precision (AMP) — big speed boost on RTX 5060 Ti
    use_amp: bool = True
    use_compile: bool = False  # torch.compile() is unstable on Windows

    # ── Regularization ───────────────────────────────────────────────────────
    dropout: float = 0.3
    label_smoothing: float = 0.1

    # ── Optimizer ────────────────────────────────────────────────────────────
    # backbone_lr_multiplier < 1.0 applies a lower LR to the pretrained backbone
    # vs. the randomly-initialised head (differential LR)
    backbone_lr_multiplier: float = 1.0
    gradient_clip: float = 1.0

    # ── LR Scheduler ─────────────────────────────────────────────────────────
    warmup_epochs: int = 0        # linear warmup before cosine annealing
    eta_min: float = 1e-6         # minimum LR at end of cosine schedule

    # ── Loss Function ─────────────────────────────────────────────────────────
    use_focal_loss: bool = False   # replaces CrossEntropyLoss with FocalLoss
    focal_loss_gamma: float = 2.0  # focusing parameter (0 = standard CE loss)
    use_class_weights: bool = False  # weight loss by inverse class frequency

    # ── Early Stopping ───────────────────────────────────────────────────────
    early_stop_patience: int = 10

    # ── Augmentation (constrained) ───────────────────────────────────────────
    # Shearing: two severity categories
    shear_mild: float = 15.0      # category 1: ±15°
    shear_strong: float = 30.0    # category 2: ±30°
    # Rotation: three categories (degrees for RandomRotation)
    rotate_cat1: int = 90
    rotate_cat2: int = 180
    rotate_cat3: int = 270
    aug_probability: float = 0.7  # probability each augmentation fires per image

    # ── Output ───────────────────────────────────────────────────────────────
    output_dir: str = "outputs"
    checkpoint_name: str = "best_model.pth"
    tuning_dir: str = "tuning_results"

    # ── Reproducibility ──────────────────────────────────────────────────────
    seed: int = 42

    # ── Computed (do not edit) ───────────────────────────────────────────────
    def __post_init__(self):
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    @property
    def checkpoint_path(self) -> str:
        return os.path.join(self.output_dir, self.checkpoint_name)
