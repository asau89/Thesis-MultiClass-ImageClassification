# ConvNeXt Multiclass Parasite Egg Classification — Project Plan

Build a PyTorch image classification system using **ConvNeXt** (transfer learning) to classify microscopy images of parasite eggs into 3 classes: *Ascaris lumbricoides*, *Hookworm*, and *Trichuris trichiura*.

**Target hardware**: AMD Ryzen 7 7700 · 32 GB RAM · NVIDIA RTX 5060 Ti 16 GB

## Dataset Summary

| Split | Ascaris | Hookworm | Trichuris | Total |
|-------|---------|----------|-----------|-------|
| Train | 100     | 100      | 100       | 300   |
| Val   | 100     | 100      | 100       | 300   |
| Test  | 78      | 100      | 67        | 245   |

All images `.jpg`. Text manifests (`train_set.txt`, `val_set.txt`, `test_set.txt`) list relative paths without extensions.

## Architecture Decisions

- **Model**: `convnext_base` (pretrained on ImageNet-1K) — maximises use of 16 GB VRAM
- **Input resolution**: `384×384`
- **Batch size**: 16 (fits comfortably in VRAM with AMP)
- **Mixed precision**: `torch.amp` (FP16) for faster GPU training
- **Optimizer**: AdamW with cosine annealing LR scheduler
- **Loss**: CrossEntropyLoss
- **Epochs**: 50 (with early stopping, patience=10)

## Data Augmentation (constrained)

Only the following augmentations are applied to the **training** set:

| Type | Categories |
|------|------------|
| Shearing | Category 1: mild (shear ±15°) — Category 2: strong (shear ±30°) |
| Rotation | Category 1: 90° — Category 2: 180° — Category 3: 270° |

Val and test sets: resize + center crop + normalize only (no augmentation).

## File Structure

```
Thesis-MultiClass-Image-Classification/
├── data/
│   ├── train_set.txt
│   ├── val_set.txt
│   └── test_set.txt
├── src/
│   ├── __init__.py
│   ├── config.py           # Centralized hyperparameters & paths
│   ├── dataset.py          # Custom Dataset + DataLoader factory
│   ├── model.py            # ConvNeXt model builder
│   ├── utils.py            # EarlyStopping, plotting, seeding
│   └── visualize_cam.py    # Grad-CAM visualization logic
├── scripts/
│   ├── train.py            # Training loop (GPU-accelerated)
│   ├── evaluate.py         # Test-set evaluation + confusion matrix
│   ├── analysis.py         # Academic research tools
│   ├── inference.py        # CLI inference tool
│   ├── tune.py             # Hyperparameter tuning
│   └── compare_results.py  # Tuning comparison
├── templates/
│   └── index.html          # Web UI frontend
├── app.py                  # Flask Web UI backend
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker service definition
├── requirements.txt
├── .gitignore
└── README.md               # This file
```

## How to Run

### 1. Install dependencies (on PC)
```bash
# CUDA-enabled PyTorch (for RTX 5060 Ti, CUDA 12+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install timm Pillow numpy pandas matplotlib seaborn scikit-learn tqdm
```

### 2. Train
```bash
python train.py
```
Checkpoints and training history are saved to `outputs/`.

### 3. Evaluate
```bash
python evaluate.py
```
Prints per-class precision/recall/F1, saves confusion matrix to `outputs/confusion_matrix.png`.

### 4. Debug (CPU, 1 epoch)
```bash
python train.py --debug
```

## Verification Plan

```bash
# Forward pass shape check
python -c "from model import get_model; from config import Config; m = get_model(Config()); import torch; print(m(torch.randn(1,3,384,384)).shape)"
# Expected: torch.Size([1, 3])

# Dataset loading check
python -c "from dataset import get_dataloaders; from config import Config; t,v,te = get_dataloaders(Config()); b=next(iter(t)); print(b[0].shape, b[1].shape)"
# Expected: torch.Size([16, 3, 384, 384]) torch.Size([16])

# Debug training (CPU)
python train.py --debug
```

## Web UI & Docker

### 1. Run Web UI Locally
```bash
# Install additional requirements
pip install flask gunicorn opencv-python-headless
# Start server
python app.py
```
Open `http://localhost:5000` in your browser.

### 2. Run with Docker
```bash
# Build and start container
docker-compose up --build
```
Ensure your `outputs/best_model.pth` exists as it is mounted into the container.
