# Parasite Egg Image Classification (ConvNeXt-Base)

An academic-grade image classification system for parasite egg detection, developed for thesis research. This project uses the **ConvNeXt-Base** architecture (384x384 resolution) to classify three types of parasite eggs:
1. `Ascaris_lumbricoides`
2. `Hookworm`
3. `Trichuris_trichiura`

## 🚀 Features
- **Modern Architecture**: Leverages `convnext_base` (pre-trained on ImageNet) with `torch.compile` for speed.
- **Academic Analysis**: Built-in tools for ROC/AUC curves, t-SNE feature visualization, and computational cost reporting.
- **Robust Training**: 5-seed multi-run support for statistical significance (`mean ± std`).
- **Phased Tuning**: 8-phase hyperparameter tuning strategy (Learning Rate, Batch Size, Weight Decay, Dropout, Focal Loss, Class Weights, Differential LR, etc.).
- **Inference Pipeline**: Batch and single-image inference with visual grid outputs.

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/asau89/Thesis-MultiClass-ImageClassification.git
   cd Thesis-MultiClass-ImageClassification
   ```

2. **Install PyTorch (CUDA 12.4 recommended)**:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
   ```

3. **Install other dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📂 Project Structure

- `dataset.py`: Custom PyTorch `Dataset` and `DataLoader` with constrained augmentations (rotation/shearing).
- `model.py`: `ConvNeXtClassifier` wrapper with support for differential learning rates.
- `train.py`: Primary training script (supports single and multi-seed runs).
- `evaluate.py`: Standard test-set evaluation (Accuracy, F1, Confusion Matrix).
- `analysis.py`: Academic research tools (ROC, t-SNE, Cost Analysis).
- `tune.py` & `compare_results.py`: Hyperparameter tuning pipeline.
- `inference.py`: User-friendly script for classifying unseen images.
- `config.py`: Centralized configuration for all hyperparameters and paths.
- `docs/`: Detailed documentation for every phase of the project.

---

## 🏃 Usage

### 1. Training
For a standard single-run training:
```bash
python train.py
```
For academic reporting (5 seeds for statistical stability):
```bash
python train.py --seeds 42,123,456,789,1234
```

### 2. Evaluation
Assess the model on the test set:
```bash
python evaluate.py
```

### 3. Academic Analysis
Generate research-quality plots and reports:
```bash
python analysis.py --all
```
Outputs: `outputs/analysis/roc_curves.png`, `tsne_features.png`, `cost_report.txt`.

### 4. Hyperparameter Tuning
Follow the phased strategy:
```bash
python tune.py --phase 1  # (Run phases 1-8 sequentially)
python compare_results.py  # Analyze and find the best config
```

### 5. Inference
Test the model on new, unseen images:
```bash
python inference.py --folder path/to/unseen_images/
```

---

## 📊 Documentation
Detailed guides are available in the [docs/](docs/) folder:
0. [Overview](docs/00_overview.md)
1. [Dataset Setup](docs/01_dataset_setup.md)
2. [Model Architecture](docs/02_model_architecture.md)
3. [Training Workflow](docs/03_training.md)
4. [Evaluation Metrics](docs/04_evaluation.md)
5. [Hyperparameter Tuning](docs/05_hyperparameter_tuning.md)
6. [Inference Guide](docs/06_inference.md)
7. [Academic Analysis](docs/07_academic_analysis.md)

---

## ⚙️ Hardware Recommendations
Developed and tested on:
- **GPU**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM)
- **CPU**: AMD Ryzen 7 7700
- **RAM**: 32GB
