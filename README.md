# 🔬 Multi-Class Parasite Egg Classification

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Research--Ready-success.svg)]()

An academic-grade computer vision system designed for the automated identification of intestinal parasite eggs from microscopy images. Developed as a thesis project, this system leverages the **ConvNeXt-Base** architecture to achieve near-perfect classification across three major species of soil-transmitted helminths (STH).

---

## 🌟 Key Features

*   **🏆 State-of-the-Art Accuracy**: Achieves **100% Accuracy** on a balanced test set for three parasite species.
*   **💡 Explainable AI (XAI)**: Integrated Grad-CAM visualization to highlight morphological features (shells, plugs, internal masses) that the model focuses on.
*   **🌐 Interactive Web UI**: A modern, dark-themed Flask interface for batch processing, real-time inference, and results exploration.
*   **📊 Academic Tools**: Built-in scripts for multi-seed statistical reporting, ROC/AUC analysis, t-SNE feature visualization, and computational cost profiling.
*   **🐳 Deployment Ready**: Fully containerized with Docker and Docker Compose for consistent cross-platform performance.

---

## 🧬 Target Species

The system is trained to identify three of the most prevalent intestinal parasites:

1.  **Ascaris lumbricoides**: Large roundworm eggs with thick, mammillated outer shells.
2.  **Hookworm**: Necator/Ancylostoma eggs featuring thin, transparent shells.
3.  **Trichuris trichiura**: Whipworm eggs identified by their characteristic barrel shape and bipolar plugs.

---

## 🚀 Getting Started

### 📦 Installation

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Thesis-MultiClass-ImageClassification.git
cd Thesis-MultiClass-ImageClassification

# 2. Create a virtual environment and install dependencies
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 🖥️ Running the Web UI

```bash
python app.py
```
Visit `http://localhost:5000` to access the interactive dashboard.

### 🐳 Using Docker

```bash
docker-compose up --build
```

---

## 🧪 Scientific Workflow

### 1. Training & Multi-Seed Analysis
```bash
# Standard training run
python scripts/train.py

# Multi-seed run to evaluate statistical stability
python scripts/train.py --seeds 42,123,456,789,1234
```

### 2. Model Evaluation
```bash
python scripts/evaluate.py
```

### 3. Thesis Analysis (t-SNE, ROC, Cost)
```bash
python scripts/analysis.py --all
```
All outputs are saved to the `outputs/analysis/` directory.

---

## 📂 Project Architecture

```text
Thesis-MultiClass-Image-Classification/
├── src/                # Core Logic (Config, Model, Dataset, Utils)
├── scripts/            # CLI Tools (Train, Eval, Tune, Analysis)
├── data/               # Dataset manifests and split lists
├── templates/          # Modern Web UI (HTML/CSS)
├── outputs/            # Model checkpoints, logs, and research plots
├── app.py              # Flask Web Application
├── Dockerfile          # Container configuration
└── DOCUMENTATION.md    # Technical depth & project details
```

---

## 📖 Documentation
Detailed technical documentation for each phase is available in the [docs/](docs/) directory:
*   [Model Architecture](docs/02_model_architecture.md)
*   [Grad-CAM Explainability](docs/07_academic_analysis.md)
*   [Hyperparameter Tuning Strategy](docs/05_hyperparameter_tuning.md)

---

## ⚙️ Hardware Environment
Tested on:
*   **GPU**: NVIDIA GeForce RTX 5060 Ti (16GB VRAM)
*   **CPU**: AMD Ryzen 7 7700
*   **RAM**: 32GB
