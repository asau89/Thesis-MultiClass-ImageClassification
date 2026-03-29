# Evaluation

## Overview

`evaluate.py` loads the best trained checkpoint (`outputs/best_model.pth`) and runs inference on the **test set** (`test_set.txt`). No augmentation is applied during evaluation.

## Metrics

| Metric | Scope |
|--------|-------|
| Overall accuracy | Percentage of correctly classified test images |
| Per-class precision | TP / (TP + FP) for each class |
| Per-class recall | TP / (TP + FN) for each class |
| Per-class F1-score | Harmonic mean of precision and recall |
| Macro-averaged F1 | Unweighted mean across classes |

Metrics are computed via `sklearn.metrics.classification_report` and printed to the console.

## Outputs

| File | Content |
|------|---------|
| `outputs/confusion_matrix.png` | Heatmap of predicted vs actual class |
| `outputs/predictions.csv` | Per-image: index, true label, predicted label, class names, correct/wrong |
| `outputs/training_curves.png` | Re-plotted loss + accuracy curves (if training history exists) |

## How to Run

```bash
# Use default best_model.pth
python evaluate.py

# Use a specific checkpoint (e.g. from tuning)
python evaluate.py --checkpoint tuning_results/phase_3/trial_2/best_model.pth
```

## Sample Console Output

```
════════════════════════════════════════════════════════════
CLASSIFICATION REPORT
════════════════════════════════════════════════════════════
                        precision  recall  f1-score  support

   Ascaris_lumbricoides    0.9487  0.9231    0.9357       78
              Hookworm     0.9200  0.9200    0.9200      100
   Trichuris_trichiura     0.9118  0.9254    0.9186       67

               accuracy                       0.9265      245
              macro avg    0.9268  0.9228    0.9248      245
           weighted avg    0.9267  0.9265    0.9266      245

[evaluate] Overall test accuracy: 92.65%
```

## Confusion Matrix

The confusion matrix heatmap (`confusion_matrix.png`) shows:
- **Rows** = actual (true) class
- **Columns** = predicted class
- Numbers = count of images

A perfect model would have all values on the diagonal.

## Key Files

| File | Role |
|------|------|
| `evaluate.py` | Evaluation script |
| `outputs/best_model.pth` | Checkpoint to evaluate |
| `test_set.txt` | List of test image paths |
