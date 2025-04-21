#json import os ✅ JSON saved to /home/u873859/BraTS2021_TrainVal_TestSplit/brats21_split_trainval.json

import random
from pathlib import Path

# Directory with the train/val data only
trainval_dir = Path("/home/u873859/BraTS2021_TrainVal_TestSplit/TrainVal")
output_json = Path("/home/u873859/BraTS2021_TrainVal_TestSplit/brats21_split_trainval.json")

# Get all case folders
case_dirs = sorted([p for p in trainval_dir.iterdir() if p.is_dir() and "BraTS2021_" in p.name])

# Shuffle and assign to 5 folds
random.seed(42)
random.shuffle(case_dirs)
num_folds = 5
folds = {f"fold_{i}": {"training": [], "validation": []} for i in range(num_folds)}

for i, case_dir in enumerate(case_dirs):
    fold_id = i % num_folds
    case_id = case_dir.name
    sample = {
        "image": [
            f"{case_id}/{case_id}_flair.nii.gz",
            f"{case_id}/{case_id}_t1ce.nii.gz",
            f"{case_id}/{case_id}_t1.nii.gz",
            f"{case_id}/{case_id}_t2.nii.gz"
        ],
        "label": f"{case_id}/{case_id}_seg.nii.gz"
    }

    for f in range(num_folds):
        if f == fold_id:
            folds[f"fold_{f}"]["validation"].append(sample)
        else:
            folds[f"fold_{f}"]["training"].append(sample)

# Save to JSON
with open(output_json, "w") as f:
    json.dump(folds, f, indent=4)

print(f"✅ JSON saved to {output_json}")
