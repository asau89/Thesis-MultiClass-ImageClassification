import os
import shutil
import random
from pathlib import Path

random.seed(42)

dataset_dir = Path("dataset")
classes = ["Ascaris_lumbricoides", "Hookworm", "Trichuris_trichiura"]

new_dir = Path("dataset_new")
if new_dir.exists():
    shutil.rmtree(new_dir)

for split in ['train', 'test', 'val']:
    for c in classes:
        (new_dir / split / c).mkdir(parents=True, exist_ok=True)

stats = {c: {'train': 0, 'test': 0, 'val': 0, 'total': 0} for c in classes}

for c in classes:
    # Use a dictionary to keep only one file path per unique filename -> deduplication
    unique_images = {}
    
    for split in ['train', 'test', 'val']:
        class_dir = dataset_dir / split / c
        if class_dir.exists():
            for f in class_dir.iterdir():
                if f.is_file():
                    unique_images[f.name] = f
    
    # Extract the unique file paths
    all_images = list(unique_images.values())
    
    # Shuffle for random distribution
    random.shuffle(all_images)
    
    total = len(all_images)
    if total == 0:
        continue
        
    train_count = int(total * 0.70)
    test_count = int(total * 0.20)
    val_count = total - train_count - test_count
    
    train_imgs = all_images[:train_count]
    test_imgs = all_images[train_count:train_count+test_count]
    val_imgs = all_images[train_count+test_count:]
    
    for img in train_imgs:
        shutil.copy2(img, new_dir / 'train' / c / img.name)
    for img in test_imgs:
        shutil.copy2(img, new_dir / 'test' / c / img.name)
    for img in val_imgs:
        shutil.copy2(img, new_dir / 'val' / c / img.name)
        
    stats[c]['train'] = len(train_imgs)
    stats[c]['test'] = len(test_imgs)
    stats[c]['val'] = len(val_imgs)
    stats[c]['total'] = total

# Replace old dataset
shutil.rmtree(dataset_dir)
new_dir.rename(dataset_dir)

print("\n--- Redistribution Summary (Deduplicated) ---")
for c in classes:
    print(f"Class: {c}")
    print(f"  Total: {stats[c]['total']} unique images")
    print(f"  Train: {stats[c]['train']} (70%)")
    print(f"  Test : {stats[c]['test']} (20%)")
    print(f"  Val  : {stats[c]['val']} (10%)")
    print("-" * 30)
