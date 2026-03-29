import json
from pathlib import Path

res = {}
for c in ['Ascaris_lumbricoides', 'Hookworm', 'Trichuris_trichiura']:
    res[c] = {
        'train': len(list(Path(f'dataset/train/{c}').glob('*'))),
        'test': len(list(Path(f'dataset/test/{c}').glob('*'))),
        'val': len(list(Path(f'dataset/val/{c}').glob('*')))
    }

with open('final_counts.json', 'w') as f:
    json.dump(res, f, indent=4)
