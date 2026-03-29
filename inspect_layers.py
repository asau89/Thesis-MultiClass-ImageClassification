import torch
from config import Config
from model import get_model

cfg = Config()
cfg.device = "cpu"
model = get_model(cfg)

# If it's a CompiledModel, get the original module
if hasattr(model, "_orig_mod"):
    model = model._orig_mod

print("--- Model Architecture Layers ---")
for name, module in model.backbone.named_modules():
    if len(list(module.children())) == 0:  # Only print leaf layers
        # For ConvNeXt, we are interested in the stages
        if "stages" in name and "conv" in name:
            print(f"Layer: {name}")

print("\n--- Summary of Stages ---")
if hasattr(model.backbone, "stages"):
    for i, stage in enumerate(model.backbone.stages):
        print(f"Stage {i}: {type(stage)}")
