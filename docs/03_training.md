# Training

## Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| `epochs` | 50 | Maximum epochs; early stopping may terminate earlier |
| `batch_size` | 16 | Fits 16 GB VRAM at 384×384 with AMP |
| `lr` | 1e-4 | AdamW learning rate |
| `weight_decay` | 1e-2 | L2 regularization |
| `label_smoothing` | 0.1 | Softens target distribution |
| `early_stop_patience` | 10 | Epochs without val loss improvement before stopping |
| `seed` | 42 | Fixed for reproducibility |

All values are defined in `config.py` and can be changed there.

## Optimizer & Scheduler

- **Optimizer**: AdamW (`lr=1e-4`, `weight_decay=1e-2`)
- **Scheduler**: Cosine Annealing — smoothly decays LR from `1e-4` to `1e-6` over 50 epochs
- **Gradient clipping**: `max_norm=1.0` — prevents exploding gradients

## Training Loop (per epoch)

```
For each batch in train_loader:
  1. Move images + labels to GPU
  2. Forward pass under AMP autocast (FP16)
  3. Compute CrossEntropyLoss (label_smoothing=0.1)
  4. optimizer.zero_grad(set_to_none=True)
  5. scaler.scale(loss).backward()
  6. Gradient clip (max_norm=1.0)
  7. scaler.step(optimizer)  +  scaler.update()

After all batches:
  8. Val epoch (no_grad, no AMP grad tracking)
  9. scheduler.step()
 10. Log metrics
 11. Save checkpoint if val_acc is best
 12. Check early stopping on val_loss
```

## Checkpointing

Best model is saved to `outputs/best_model.pth` whenever validation accuracy improves. The checkpoint contains:
- `model_state_dict`
- `optimizer_state_dict`
- `epoch`
- `val_acc`
- `config` (full hyperparameter dict)

## Outputs

| File | Content |
|------|---------|
| `outputs/best_model.pth` | Best checkpoint by val accuracy |
| `outputs/training_history.json` | Epoch-by-epoch train/val loss + accuracy + LR |
| `outputs/training_curves.png` | Loss and accuracy plots |

## How to Run

```bash
# Full training on GPU
python train.py

# Debug/smoke test (CPU, 1 epoch, 2 batches)
python train.py --debug
```

## Verification

```bash
# Forward pass shape check
python -c "from model import get_model; from config import Config; m = get_model(Config()); import torch; print(m(torch.randn(1,3,384,384)).shape)"
# Expected: torch.Size([1, 3])
```
