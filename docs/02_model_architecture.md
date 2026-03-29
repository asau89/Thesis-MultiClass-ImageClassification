# Model Architecture

## Backbone: ConvNeXt-Base

The model uses **ConvNeXt-Base** pretrained on ImageNet-1K, loaded via the `timm` library.

| Property | Value |
|----------|-------|
| Backbone | `convnext_base` |
| Pretrained | ImageNet-1K |
| Input resolution | 384 × 384 |
| Backbone output features | 1024 |
| Number of classes | 3 |

## Architecture Diagram

```
Input image (384×384×3)
        │
        ▼
┌─────────────────────────┐
│   ConvNeXt-Base         │
│   (pretrained backbone) │  ← ImageNet weights, all layers trainable
└─────────────────────────┘
        │  (1024-dim feature vector)
        ▼
  LayerNorm(1024)
        │
  Dropout(p=0.3)           ← regularizes the small dataset
        │
  Linear(1024 → 3)
        │
        ▼
 Logits [B × 3]            ← CrossEntropyLoss applied here during training
```

## Why ConvNeXt?

- Modernized pure-CNN architecture (no attention layers) — highly efficient on GPU
- Strong pretrained features from ImageNet reduce the need for large training sets
- `convnext_base` at `384×384` maximizes the RTX 5060 Ti 16 GB VRAM capacity

## Regularization Choices

| Technique | Value | Reason |
|-----------|-------|--------|
| Dropout | 0.3 | Prevents overfitting on the small 300-image training set |
| Label smoothing | 0.1 | Reduces overconfidence, improves generalization |
| Weight decay (AdamW) | 1e-2 | L2 regularization via optimizer |

## GPU Optimizations

| Feature | Benefit |
|---------|---------|
| `torch.compile()` | JIT-compiles the model graph for faster kernel execution |
| Mixed precision (AMP FP16) | ~2× throughput, halves VRAM usage |
| `pin_memory=True` | Faster CPU → GPU data transfer |
| `num_workers=6` | Saturates data pipeline on R7 7700 (8 cores) |

## Key Files

| File | Role |
|------|------|
| `model.py` | `ConvNeXtClassifier` class and `get_model()` factory |
| `config.py` | `model_name`, `image_size`, `dropout`, `pretrained` settings |
