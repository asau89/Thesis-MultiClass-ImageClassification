import os
from collections import defaultdict
from pathlib import Path

dataset_dir = Path("dataset")
name_to_paths = defaultdict(list)

for img_path in dataset_dir.rglob("*"):
    if img_path.is_file():
        name_to_paths[img_path.name].append(str(img_path))

duplicates = {name: paths for name, paths in name_to_paths.items() if len(paths) > 1}

with open("duplicates_report.txt", "w") as f:
    f.write(f"Total Unique Filenames with Duplicates: {len(duplicates)}\n\n")
    for name, paths in duplicates.items():
        f.write(f"Filename: {name}\n")
        for p in paths:
            f.write(f"  - {p}\n")
        f.write("\n")

print(f"Found {len(duplicates)} filenames that appear multiple times.")
