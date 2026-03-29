# Academic Analysis

Post-training analysis scripts that generate outputs required for academic publication and thesis review.

## Scripts

| Script | Purpose |
|--------|---------|
| `analysis.py` | ROC/AUC curves, t-SNE embedding, computational cost |
| `train.py --seeds` | Multi-seed training for statistical significance |

---

## Multi-Seed Training

**Why it matters**: A single run is a single random sample. Academic papers report results as **mean ± std** across multiple seeds to demonstrate that performance is consistent, not due to a lucky random initialization.

### Usage

```bash
# 5-seed training (recommended for thesis)
python train.py --seeds 42,123,456,789,1234
```

### Outputs

| File | Content |
|------|---------|
| `outputs/best_model_seed42.pth` | Best checkpoint per seed |
| `outputs/best_model_seed123.pth` | Best checkpoint per seed |
| ... | |
| `outputs/multi_seed_results.json` | Per-seed metrics + mean ± std summary |

### Sample Console Output

```
════════════════════════════════════════════════════
MULTI-SEED RESULTS  (5 seeds)
════════════════════════════════════════════════════
Seed         Val Acc    Best Epoch    Epochs Run
────────────────────────────────────────────────────
42            93.47%           31            41
123           92.24%           28            38
456           93.88%           34            44
789           91.43%           25            35
1234          94.29%           37            47
────────────────────────────────────────────────────
Mean          93.06%  ±1.05%

[train] Final result to report in paper: Val Acc = 93.06 ± 1.05%
```

---

## ROC Curves + AUC

**Why it matters**: ROC (Receiver Operating Characteristic) curves show performance at all decision thresholds, not just argmax. AUC (Area Under Curve) is a threshold-independent metric. Required in most classification papers.

### Usage

```bash
python analysis.py --roc
```

### Outputs

| File | Content |
|------|---------|
| `outputs/analysis/roc_curves.png` | Per-class ROC curves with AUC in legend |
| `outputs/analysis/roc_auc_scores.json` | Numeric AUC values per class + macro average |

### Sample Output

```
Class                           AUC
────────────────────────────────────────
Ascaris_lumbricoides           0.9872
Hookworm                       0.9941
Trichuris_trichiura            0.9807
────────────────────────────────────────
Macro Average                  0.9873
```

---

## t-SNE Feature Embedding

**Why it matters**: Shows that the model has learned **separable, class-discriminative features**. A well-separated t-SNE plot visually confirms the model is not relying on spurious correlations.

### Usage

```bash
python analysis.py --tsne

# Adjust perplexity for different dataset sizes (default: 30)
python analysis.py --tsne --tsne-perplexity 20
```

### Outputs

| File | Content |
|------|---------|
| `outputs/analysis/tsne_features.png` | 2D scatter of backbone features, colored by class |

### How it works

The script attaches a hook to the ConvNeXt backbone's final layer before the classification head, captures the 1024-dim feature vector for every test image, then reduces to 2D using t-SNE for visualization.

---

## Computational Cost

**Why it matters**: Papers and thesis committees expect model efficiency metrics so readers can assess practical applicability (memory, speed, compute).

### Usage

```bash
python analysis.py --cost
```

### Outputs

| File | Content |
|------|---------|
| `outputs/analysis/cost_report.txt` | Human-readable cost table |
| `outputs/analysis/cost_report.json` | Machine-readable cost data |

### Sample Output

```
══════════════════════════════════════════════════
COMPUTATIONAL COST REPORT
══════════════════════════════════════════════════
Model            : convnext_base
Input resolution : 384 × 384
Total parameters : 88,591,464
Trainable params : 88,591,464
FLOPs            : 45.21 GFLOPs
Inference speed  : 47.3 images/sec (CUDA)
══════════════════════════════════════════════════
```

> [!NOTE]
> FLOPs calculation requires `thop` (`pip install thop`). If not installed, FLOPs will show as N/A — all other metrics still work.

---

## Run All Analyses at Once

```bash
python analysis.py --all

# With custom checkpoint
python analysis.py --all --checkpoint tuning_results/phase_3/trial_2/best_model.pth
```

All outputs are saved to `outputs/analysis/`.
