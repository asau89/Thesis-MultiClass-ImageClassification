# Hyperparameter Tuning

## Strategy: Phased Grid Search (Full Training)

Each trial runs **full 50-epoch training** with early stopping (patience 10). Each phase locks in the best values before the next phase begins. All best values carry forward automatically.

**Total trials**: ~34 · **Estimated time**: ~25.5 hours on RTX 5060 Ti

---

## All Tunable Hyperparameters

| Group | Parameter | Default | Phase |
|-------|-----------|---------|-------|
| Optimizer | `lr` | 1e-4 | 1 |
| Optimizer | `batch_size` | 16 | 2 |
| Optimizer | `weight_decay` | 1e-2 | 2 |
| Optimizer | `backbone_lr_multiplier` | 1.0 | 7 |
| Optimizer | `gradient_clip` | 1.0 | 5 |
| Regularization | `dropout` | 0.3 | 3 |
| Regularization | `label_smoothing` | 0.1 | 3 |
| LR Scheduler | `warmup_epochs` | 0 | 5 |
| LR Scheduler | `eta_min` | 1e-6 | 5 |
| Loss Function | `use_focal_loss` | False | 6 |
| Loss Function | `focal_loss_gamma` | 2.0 | 6 |
| Loss Function | `use_class_weights` | False | 6 |
| Model | `model_name` | convnext_base | 4 |
| Model | `image_size` | 384 | 4 |
| Augmentation | `aug_probability` | 0.7 | 8 |

---

## Phase 1 — Learning Rate (5 trials, ~3.75 hrs)

| Trial | `lr` |
|-------|------|
| 1-1 | 1e-5 |
| 1-2 | 5e-5 |
| 1-3 | **1e-4** (baseline) |
| 1-4 | 3e-4 |
| 1-5 | 5e-4 |

---

## Phase 2 — Batch Size + Weight Decay (6 trials, ~4.5 hrs)

| Trial | `batch_size` | `weight_decay` |
|-------|-------------|----------------|
| 2-1 | 8  | 1e-2 |
| 2-2 | 8  | 5e-2 |
| 2-3 | **16** | **1e-2** (baseline) |
| 2-4 | 16 | 5e-2 |
| 2-5 | 32 | 1e-2 |
| 2-6 | 32 | 5e-2 |

---

## Phase 3 — Dropout + Label Smoothing (5 trials, ~3.75 hrs)

| Trial | `dropout` | `label_smoothing` |
|-------|----------|-------------------|
| 3-1 | 0.2 | 0.0 |
| 3-2 | 0.2 | 0.1 |
| 3-3 | **0.3** | **0.1** (baseline) |
| 3-4 | 0.5 | 0.1 |
| 3-5 | 0.5 | 0.2 |

---

## Phase 4 — Model Variant + Resolution (3 trials, ~2.25 hrs) *(optional)*

| Trial | `model_name` | `image_size` |
|-------|-------------|-------------|
| 4-1 | `convnext_tiny`  | 224 |
| 4-2 | `convnext_small` | 384 |
| 4-3 | **`convnext_base`** | **384** (baseline) |

---

## Phase 5 — LR Scheduler: Warmup + eta_min (4 trials, ~3 hrs)

Linear warmup gradually ramps LR from near-zero to full LR over N epochs before cosine decay begins. Helps avoid shocking pretrained weights at the start.

| Trial | `warmup_epochs` | `eta_min` |
|-------|----------------|-----------|
| 5-1 | **0** | **1e-6** (baseline) |
| 5-2 | 3 | 1e-6 |
| 5-3 | 5 | 1e-6 |
| 5-4 | 5 | 1e-7 |

---

## Phase 6 — Loss Function: Focal Loss + Class Weights (5 trials, ~3.75 hrs)

**Focal Loss** focuses training on hard/misclassified examples (`gamma=0` = standard CE).
**Class weights** up-weight under-represented classes (relevant since test set has 78/100/67 images).

| Trial | `use_focal_loss` | `focal_gamma` | `use_class_weights` |
|-------|-----------------|---------------|---------------------|
| 6-1 | **False** | — | **False** (baseline) |
| 6-2 | True | 1.0 | False |
| 6-3 | True | 2.0 | False |
| 6-4 | True | 2.0 | True |
| 6-5 | False | — | True |

---

## Phase 7 — Differential LR: Backbone Multiplier (3 trials, ~2.25 hrs)

Applies a **lower learning rate** to the pretrained ConvNeXt backbone vs the new head.
`backbone_lr = lr × backbone_lr_multiplier`

| Trial | `backbone_lr_multiplier` | Effective backbone LR |
|-------|-------------------------|-----------------------|
| 7-1 | 0.1 | `lr × 0.1` |
| 7-2 | 0.3 | `lr × 0.3` |
| 7-3 | **1.0** (baseline) | Same as head LR |

---

## Phase 8 — Augmentation Probability (4 trials, ~3 hrs)

Controls how often shearing and rotation are applied per training image.

| Trial | `aug_probability` |
|-------|------------------|
| 8-1 | 0.3 |
| 8-2 | 0.5 |
| 8-3 | **0.7** (baseline) |
| 8-4 | 1.0 (always applied) |

---

## Outputs per Trial

```
tuning_results/
  phase_N/
    trial_X/
      best_model.pth           ← checkpoint at best val accuracy
      training_history.json    ← loss + accuracy per epoch
      training_curves.png      ← loss + accuracy curves
      val_metrics.json         ← val acc, F1, precision, recall, best epoch
    phase_summary.json         ← ranked results for the phase
    phase_comparison.png       ← bar chart: all trials in this phase
```

## Post-Tuning Outputs (compare_results.py)

```
tuning_results/
  best_config.json             ← best hyperparameters to paste into config.py
  all_phases_comparison.png    ← side-by-side comparison across all phases
  accuracy_trend.png           ← accuracy + F1 trend across all trials in order
```

---

## How to Run

```bash
python tune.py --phase 1          # ~3.75 hrs
python tune.py --phase 2          # ~4.5 hrs
python tune.py --phase 3          # ~3.75 hrs
python tune.py --phase 4          # ~2.25 hrs (optional)
python tune.py --phase 5          # ~3 hrs
python tune.py --phase 6          # ~3.75 hrs
python tune.py --phase 7          # ~2.25 hrs
python tune.py --phase 8          # ~3 hrs

python compare_results.py         # Final comparison + best_config.json
```

Preview a phase without training:
```bash
python tune.py --phase 5 --dry-run
```

## After Tuning

1. Open `tuning_results/best_config.json`
2. Copy the values into `config.py`
3. Run final training: `python train.py`
4. Evaluate: `python evaluate.py`
