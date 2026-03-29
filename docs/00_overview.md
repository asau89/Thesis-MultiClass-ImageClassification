# Project Overview

A PyTorch-based multiclass image classification system to identify parasite eggs in microscopy images using **ConvNeXt** (transfer learning).

## Problem Statement

Classify microscopy images of parasite eggs into 3 classes:

| Class | Label |
|-------|-------|
| *Ascaris lumbricoides* | 0 |
| *Hookworm* | 1 |
| *Trichuris trichiura* | 2 |

## Dataset Summary

| Split | Ascaris | Hookworm | Trichuris | Total |
|-------|---------|----------|-----------|-------|
| Train | 100 | 100 | 100 | 300 |
| Val   | 100 | 100 | 100 | 300 |
| Test  | 78  | 100 | 67  | 245  |

All images are `.jpg`. Manifest text files (`train_set.txt`, `val_set.txt`, `test_set.txt`) list relative paths without file extensions.

## Target Hardware

| Component | Spec |
|-----------|------|
| CPU | AMD Ryzen 7 7700 |
| RAM | 32 GB |
| GPU | NVIDIA RTX 5060 Ti 16 GB |
| OS | Windows |

## Project File Structure

```
Thesis-MultiClass-Image-Classification/
├── dataset/
│   ├── train/{Ascaris_lumbricoides, Hookworm, Trichuris_trichiura}/
│   ├── val/  {Ascaris_lumbricoides, Hookworm, Trichuris_trichiura}/
│   └── test/ {Ascaris_lumbricoides, Hookworm, Trichuris_trichiura}/
├── docs/
│   ├── 00_overview.md         ← This file
│   ├── 01_dataset_setup.md
│   ├── 02_model_architecture.md
│   ├── 03_training.md
│   ├── 04_evaluation.md
│   ├── 05_hyperparameter_tuning.md
│   └── 06_inference.md
├── train_set.txt
├── val_set.txt
├── test_set.txt
├── config.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── tune.py
├── compare_results.py
├── inference.py
├── utils.py
├── requirements.txt
├── .gitignore
└── PLAN.md
```

## Quick Start (on PC)

```bash
# 1. Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install timm Pillow numpy pandas matplotlib seaborn scikit-learn tqdm thop

# 2. Train (single seed)
python train.py

# 2b. Train with 5 seeds for academic mean ± std reporting
python train.py --seeds 42,123,456,789,1234

# 3. Evaluate on test set
python evaluate.py

# 4. Academic analysis (ROC/AUC, t-SNE, cost report)
python analysis.py --all

# 5. Run inference on unseen images
python inference.py --folder my_unseen_images/

# 6. (Optional) Hyperparameter tuning
python tune.py --phase 1    # Learning rate
python tune.py --phase 2    # Batch size + weight decay
python tune.py --phase 3    # Dropout + label smoothing
python tune.py --phase 4    # Model variant (optional)
python tune.py --phase 5    # LR scheduler warmup
python tune.py --phase 6    # Loss function (focal loss / class weights)
python tune.py --phase 7    # Differential LR
python tune.py --phase 8    # Augmentation probability
python compare_results.py   # Final comparison + best_config.json
```
