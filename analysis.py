"""
analysis.py — Academic analysis tools for the ConvNeXt parasite classifier.

Generates research-quality outputs required for academic papers / thesis.

Usage:
  python analysis.py --roc         # ROC curves + AUC per class
  python analysis.py --tsne        # t-SNE feature embedding visualization
  python analysis.py --cost        # Computational cost: params, FLOPs, speed
  python analysis.py --all         # All of the above

  python analysis.py --all --checkpoint path/to/model.pth  # custom checkpoint

All outputs saved to: outputs/analysis/
"""
import argparse
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from config import Config
from dataset import get_dataloaders
from model import get_model


# ── Shared: collect softmax probs + true labels from test set ─────────────────

def collect_probs_and_labels(model, loader, device, use_amp) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        probs  : (N, num_classes) softmax probabilities
        labels : (N,) integer true labels
    """
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            with torch.amp.autocast(
                device_type=device,
                enabled=(use_amp and device == "cuda"),
            ):
                logits = model(images)
            probs = F.softmax(logits, dim=1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


# ── ROC / AUC ─────────────────────────────────────────────────────────────────

def run_roc(model, loader, cfg: Config, out_dir: Path):
    """One-vs-rest ROC curves + macro AUC for each class."""
    print("\n[analysis] Computing ROC curves...")

    probs, labels = collect_probs_and_labels(model, loader, cfg.device, cfg.use_amp)

    # Binarize labels for one-vs-rest
    labels_bin = label_binarize(labels, classes=list(range(cfg.num_classes)))

    # Per-class ROC
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868"]
    aucs = []

    for i, (cls_name, color) in enumerate(zip(cfg.classes, colors)):
        fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls_name.replace('_', ' ')} (AUC = {roc_auc:.4f})")

    # Macro average AUC
    macro_auc = float(np.mean(aucs))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier (AUC = 0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curves — Macro AUC = {macro_auc:.4f}", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    plt.tight_layout()
    out_path = out_dir / "roc_curves.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()

    # Print summary
    print(f"\n  {'Class':<30} {'AUC':>8}")
    print(f"  {'─'*40}")
    for cls, roc_auc in zip(cfg.classes, aucs):
        print(f"  {cls:<30} {roc_auc:.4f}")
    print(f"  {'─'*40}")
    print(f"  {'Macro Average':<30} {macro_auc:.4f}")
    print(f"\n[analysis] ROC curves saved → {out_path}")

    # Save numeric results
    results = {
        "per_class_auc": {cls: float(a) for cls, a in zip(cfg.classes, aucs)},
        "macro_auc": macro_auc,
    }
    import json
    with open(out_dir / "roc_auc_scores.json", "w") as f:
        json.dump(results, f, indent=2)

    return macro_auc


# ── t-SNE ─────────────────────────────────────────────────────────────────────

def run_tsne(model, loader, cfg: Config, out_dir: Path, perplexity: int = 30):
    """
    Extract backbone features from the test set, run t-SNE, and
    plot a 2D scatter colored by class.
    """
    print("\n[analysis] Extracting features for t-SNE...")

    # Hook to capture backbone output (before classification head)
    underlying = getattr(model, "_orig_mod", model)
    features_list, labels_list = [], []

    def _hook(module, input, output):
        features_list.append(output.detach().cpu())

    handle = underlying.backbone.register_forward_hook(_hook)

    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(cfg.device, non_blocking=True)
            with torch.amp.autocast(
                device_type=cfg.device,
                enabled=(cfg.use_amp and cfg.device == "cuda"),
            ):
                model(images)
            labels_list.append(labels.numpy())

    handle.remove()

    features = torch.cat(features_list, dim=0).numpy()
    labels   = np.concatenate(labels_list, axis=0)

    print(f"[analysis] Features shape: {features.shape}  Running t-SNE (perplexity={perplexity})...")
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42,
                n_iter=1000, init="pca", learning_rate="auto")
    embedded = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = ["#4C72B0", "#DD8452", "#55A868"]
    markers = ["o", "s", "^"]

    for i, (cls_name, color, marker) in enumerate(zip(cfg.classes, colors, markers)):
        mask = labels == i
        ax.scatter(
            embedded[mask, 0], embedded[mask, 1],
            c=color, marker=marker, s=60, alpha=0.8, edgecolors="white", linewidths=0.5,
            label=cls_name.replace("_", " "),
        )

    ax.set_title("t-SNE Feature Embeddings (ConvNeXt Backbone)", fontsize=13)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = out_dir / "tsne_features.png"
    plt.savefig(str(out_path), dpi=150)
    plt.close()
    print(f"[analysis] t-SNE plot saved → {out_path}")


# ── Computational Cost ────────────────────────────────────────────────────────

def run_cost(model, loader, cfg: Config, out_dir: Path):
    """
    Reports model parameter counts, FLOPs (via thop), and inference speed.
    """
    print("\n[analysis] Computing computational cost...")

    # Parameter count
    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # FLOPs via thop (optional — graceful fallback if not installed)
    flops_str = "N/A (install thop: pip install thop)"
    try:
        from thop import profile as thop_profile
        dummy = torch.randn(1, 3, cfg.image_size, cfg.image_size).to(cfg.device)
        underlying = getattr(model, "_orig_mod", model)
        flops, _ = thop_profile(underlying, inputs=(dummy,), verbose=False)
        flops_str = f"{flops / 1e9:.2f} GFLOPs"
    except ImportError:
        pass

    # Inference speed (images/second on test loader)
    model.eval()
    n_warmup = 5
    n_images = 0
    t_start  = None
    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(cfg.device, non_blocking=True)
        if batch_idx == n_warmup:
            t_start = time.time()
        with torch.no_grad():
            with torch.amp.autocast(
                device_type=cfg.device,
                enabled=(cfg.use_amp and cfg.device == "cuda"),
            ):
                model(images)
        if batch_idx >= n_warmup:
            n_images += images.size(0)

    elapsed = time.time() - t_start if t_start else 1.0
    speed = n_images / elapsed if elapsed > 0 else 0.0

    # Build report
    lines = [
        "═" * 50,
        "COMPUTATIONAL COST REPORT",
        "═" * 50,
        f"Model            : {cfg.model_name}",
        f"Input resolution : {cfg.image_size} × {cfg.image_size}",
        f"Total parameters : {total_params:,}",
        f"Trainable params : {trainable_params:,}",
        f"FLOPs            : {flops_str}",
        f"Inference speed  : {speed:.1f} images/sec ({cfg.device.upper()})",
        "═" * 50,
    ]
    report_text = "\n".join(lines)
    print("\n" + report_text)

    out_path = out_dir / "cost_report.txt"
    out_path.write_text(report_text + "\n")
    print(f"[analysis] Cost report saved → {out_path}")

    # Also save as JSON for programmatic use
    import json
    cost_data = {
        "model_name": cfg.model_name,
        "image_size": cfg.image_size,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "flops": flops_str,
        "inference_speed_img_per_sec": round(speed, 2),
        "device": cfg.device,
    }
    with open(out_dir / "cost_report.json", "w") as f:
        json.dump(cost_data, f, indent=2)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Academic analysis tools for ConvNeXt parasite classifier"
    )
    parser.add_argument("--roc",    action="store_true", help="ROC curves + AUC per class")
    parser.add_argument("--tsne",   action="store_true", help="t-SNE feature embedding plot")
    parser.add_argument("--cost",   action="store_true", help="Params, FLOPs, inference speed")
    parser.add_argument("--all",    action="store_true", help="Run all analyses")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to .pth checkpoint (default: outputs/best_model.pth)")
    parser.add_argument("--tsne-perplexity", type=int, default=30,
                        help="t-SNE perplexity (default: 30)")
    args = parser.parse_args()

    if not any([args.roc, args.tsne, args.cost, args.all]):
        parser.error("Please specify at least one flag: --roc, --tsne, --cost, or --all")

    cfg = Config()
    out_dir = Path(cfg.output_dir) / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = args.checkpoint or cfg.checkpoint_path
    print(f"[analysis] Loading checkpoint: {checkpoint}")
    if not Path(checkpoint).exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}\n"
            "Run `python train.py` first."
        )

    ckpt = torch.load(checkpoint, map_location=cfg.device)
    saved_cfg = ckpt.get("config", {})
    for key, val in saved_cfg.items():
        if hasattr(cfg, key) and key not in ("device", "output_dir", "tuning_dir"):
            try:
                setattr(cfg, key, val)
            except Exception:
                pass

    model = get_model(cfg)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
    except RuntimeError:
        model._orig_mod.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, _, test_loader = get_dataloaders(cfg)

    do_roc  = args.roc  or args.all
    do_tsne = args.tsne or args.all
    do_cost = args.cost or args.all

    if do_roc:
        run_roc(model, test_loader, cfg, out_dir)

    if do_tsne:
        run_tsne(model, test_loader, cfg, out_dir, perplexity=args.tsne_perplexity)

    if do_cost:
        run_cost(model, test_loader, cfg, out_dir)

    print(f"\n[analysis] All outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()
