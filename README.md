# Parasite Egg Image Classification (ConvNeXt-Base)

An academic-grade image classification system for parasite egg detection, developed for academic project. This project uses the **ConvNeXt-Base** architecture (384x384 resolution) to classify three types of parasite eggs with high precision and explainability.

1. `Ascaris_lumbricoides`
2. `Hookworm`
3. `Trichuris_trichiura`

---

## ✨ New: Interactive Web UI & Batch Processing
The project now includes a modern, dark-themed **Web Interface** for real-time inference and model explainability.

- **Batch Upload**: Process multiple microscopy images simultaneously.
- **Grad-CAM Visualization**: Real-time attention heatmaps showing exactly where the model is "looking" to make its prediction.
- **Results Gallery**: Browse through processed samples and click to see detailed probability distributions and heatmaps.
- **Docker Ready**: Deploy the entire system with a single command.

---

## 🚀 Core Features
- **Modern Architecture**: Leverages `convnext_base` (pre-trained on ImageNet) with `torch.compile` for optimized GPU performance.
- **Explainable AI (XAI)**: Integrated Grad-CAM visualization for verifying model focus on relevant biological features.
- **Academic Analysis**: Built-in tools for ROC/AUC curves, t-SNE feature visualization, and computational cost reporting.
- **Multi-Seed Testing**: 5-seed training

## 📂 Project Structure

```text
Thesis-MultiClass-Image-Classification/
├── data/               # Dataset manifests (train/val/test lists)
├── src/                # Core logic (Config, Model, Dataset, Utils)
├── scripts/            # Command-line tools (Train, Eval, Tune, Analysis)
├── templates/          # Web UI HTML/CSS
├── app.py              # Flask Backend
├── Dockerfile          # Container configuration
└── requirements.txt    # Python dependencies
```

## 🚀 Getting Started

### 1. Local Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the Web UI
python app.py
```

### 2. Scholarly Workflow (CLI)
```bash
# Hyperparameter Tuning
python scripts/tune.py --phase 1

# Training
python scripts/train.py

# Evaluation
python scripts/evaluate.py

# Academic Analysis (ROC, t-SNE, Cost)
python scripts/analysis.py --all
```
Visit `http://localhost:5000` in your browser.

### 3. Run with Docker (Recommended for Deployment)
```bash
docker-compose up --build
```
*Note: Ensure your `outputs/best_model.pth` is present as it is mounted into the container.*

---

## 📂 Project Structure

- `app.py`: Flask backend for the interactive Web UI.
- `templates/`: Modern frontend with batch processing and gallery views.
- `visualize_cam.py`: Core logic for Grad-CAM (Gradient-weighted Class Activation Mapping).
- `train.py`: Primary training script (supports single and multi-seed runs).
- `evaluate.py`: Standard test-set evaluation (Accuracy, F1, Confusion Matrix).
- `analysis.py`: Academic research tools (ROC, t-SNE, Cost Analysis).
- `model.py`: `ConvNeXtClassifier` wrapper with differential learning rate support.
- `config.py`: Centralized configuration for hyperparameters and paths.
- `Dockerfile` & `docker-compose.yml`: Containerization for easy setup and scaling.

---

## 🏃 Academic Workflow

### 1. Training & Evaluation
```bash
# Standard training
python train.py

# Multi-seed reporting (statistical stability)
python train.py --seeds 42,123,456,789,1234

# Test-set evaluation
python evaluate.py
```

### 2. Research Analysis
Generate plots and reports for your thesis:
```bash
python analysis.py --all
```
Outputs saved to `outputs/analysis/`: ROC curves, t-SNE embeddings, and FLOPs/Speed reports.

### 3. Hyperparameter Tuning
```bash
python tune.py --phase 1  # 8 phases available
python compare_results.py  # Analyze and select best config
```

---

## 📊 Documentation
Detailed guides for every project phase:
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
