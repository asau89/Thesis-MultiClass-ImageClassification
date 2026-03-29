# Dataset Setup

## Source Structure

Images are stored in class-named subfolders under each split:

```
dataset/
├── train/
│   ├── Ascaris_lumbricoides/   (100 images)
│   ├── Hookworm/               (100 images)
│   └── Trichuris_trichiura/    (100 images)
├── val/
│   ├── Ascaris_lumbricoides/   (100 images)
│   ├── Hookworm/               (100 images)
│   └── Trichuris_trichiura/    (100 images)
└── test/
    ├── Ascaris_lumbricoides/   (78 images)
    ├── Hookworm/               (100 images)
    └── Trichuris_trichiura/    (67 images)
```

## Manifest Text Files

`train_set.txt`, `val_set.txt`, and `test_set.txt` were generated using `generate_lists.ps1`. Each file lists one image per line as a **relative path without the file extension**:

```
dataset/train/Ascaris_lumbricoides/Ascaris lumbricoides_0010
dataset/train/Hookworm/Hookworm egg_0019
...
```

To regenerate the manifest files:
```powershell
powershell -ExecutionPolicy Bypass -File .\generate_lists.ps1
```

## Class Label Mapping

Labels are derived from the subfolder name in code (`dataset.py`):

| Folder Name | Class Index |
|-------------|-------------|
| `Ascaris_lumbricoides` | 0 |
| `Hookworm` | 1 |
| `Trichuris_trichiura` | 2 |

## Data Augmentation (constrained)

Only the following augmentations are applied to the **training set**. Val and test sets receive no augmentation.

| Type | Categories | Applied With |
|------|------------|-------------|
| Shearing | Category 1: mild ±15° · Category 2: strong ±30° | 70% probability |
| Rotation | Category 1: 90° · Category 2: 180° · Category 3: 270° | 70% probability |

One category is chosen at random each time the augmentation fires.

## ImageNet Normalization

All images are normalized with ImageNet statistics:
- **Mean**: `[0.485, 0.456, 0.406]`
- **Std**: `[0.229, 0.224, 0.225]`

## Transform Pipelines

**Train**:
```
Resize(416) → CenterCrop(384) → RandomShear(±15° or ±30°, p=0.7)
→ RandomCategoricalRotation(90°/180°/270°, p=0.7) → ToTensor → Normalize
```

**Val / Test**:
```
Resize(416) → CenterCrop(384) → ToTensor → Normalize
```

## Key Files

| File | Role |
|------|------|
| `generate_lists.ps1` | Generates the manifest `.txt` files |
| `dataset.py` | `ParasiteDataset` class + `get_dataloaders()` factory |
| `config.py` | Paths to manifest files, augmentation parameters |
