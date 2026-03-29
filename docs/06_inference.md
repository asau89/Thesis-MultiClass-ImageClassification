# Inference

## Overview

`inference.py` allows you to classify **any image** not in the training dataset. It uses the trained ConvNeXt checkpoint and the same preprocessing pipeline as validation/test (no augmentation).

## Supported Input Formats

`.jpg` `.jpeg` `.png` `.bmp` `.tiff` `.tif` `.webp`

## Input Modes

| Mode | Command |
|------|---------|
| Single image | `python inference.py --image sample.jpg` |
| Multiple images | `python inference.py --image img1.jpg img2.jpg img3.jpg` |
| Entire folder | `python inference.py --folder my_images/` |
| Custom checkpoint | add `--checkpoint path/to/model.pth` to any of the above |

## How to Run

```bash
# Single image
python inference.py --image test_sample.jpg

# Folder of images
python inference.py --folder my_test_images/

# With a tuned checkpoint
python inference.py --folder my_test_images/ --checkpoint tuning_results/phase_3/trial_2/best_model.pth

# Change output directory
python inference.py --folder my_test_images/ --output-dir results/

# Change grid columns (default: 4)
python inference.py --folder my_test_images/ --grid-cols 3
```

## Console Output

For every image, the predicted class + all 3 class probabilities are printed:

```
══════════════════════════════════════════════════════════════════════
INFERENCE RESULTS
Image                      Prediction            Confidence   Ascaris   Hookworm   Trichuris
────────────────────────────────────────────────────────────────────
sample_A.jpg               Hookworm               97.32%       1.10%    97.32%      1.58%
sample_B.jpg               Ascaris lumbricoides   84.15%      84.15%     9.12%      6.73%
```

## Output Files

| File | Content |
|------|---------|
| `outputs/inference_results.csv` | All predictions: filename, predicted class, confidence, per-class probability |
| `outputs/inference_grid.png` | Visual montage of images with predicted label and confidence overlaid |

## inference_grid.png

The grid image shows each input image with:
- A **colour-coded border** per class (blue = Ascaris, orange = Hookworm, green = Trichuris)
- The **predicted class name** and **confidence %** as a label overlay
- A **confidence bar** at the bottom of each image

## Checkpoint Auto-Detection

The checkpoint stores the model variant and image size used during training. `inference.py` reads these automatically — no manual config changes required even if you switch between `convnext_tiny` and `convnext_base` checkpoints.

## Key Files

| File | Role |
|------|------|
| `inference.py` | Main inference script |
| `outputs/best_model.pth` | Default checkpoint (from `train.py`) |
| `tuning_results/phase_N/trial_X/best_model.pth` | Alternative checkpoints (from `tune.py`) |
