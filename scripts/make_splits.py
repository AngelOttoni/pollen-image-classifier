#!/usr/bin/env python3
import os, csv, shutil, random
from sklearn.model_selection import train_test_split

# Config
SRC_ROOT = "data/resized"   # where resized images are organized by class 
OUT_ROOT = "data/split_80_20"   # final organized folders
CLASSES = ["abacate","aroeira","eucalipto"]
SEED = 42
TEST_RATIO = 0.20
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

os.makedirs(OUT_ROOT, exist_ok=True)

def list_images_in_folder(folder):
    return sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS)])

all_train_paths = []
all_test_paths = []

for cls in CLASSES:
    candidate_dirs = []
    d1 = os.path.join(SRC_ROOT, "train", cls)
    d2 = os.path.join(SRC_ROOT, "test", cls)
    if os.path.isdir(d1):
        candidate_dirs.append(d1)
    if os.path.isdir(d2):
        candidate_dirs.append(d2)
    if not candidate_dirs:
        raise SystemExit(f"No folder found for class {cls} under {SRC_ROOT}/train or {SRC_ROOT}/test")

    files = []
    for d in candidate_dirs:
        files.extend(list_images_in_folder(d))
    if len(files) == 0:
        raise SystemExit(f"No images found for class {cls}")

    train_files, test_files = train_test_split(files, test_size=TEST_RATIO, random_state=SEED, shuffle=True)
    # create dest folders and copy
    for split_name, file_list in [("train", train_files), ("test", test_files)]:
        dst_dir = os.path.join(OUT_ROOT, split_name, cls)
        os.makedirs(dst_dir, exist_ok=True)
        for p in file_list:
            shutil.copy(p, dst_dir)

    # record for CSV
    all_train_paths += [(p, cls) for p in train_files]
    all_test_paths += [(p, cls) for p in test_files]

# save CSVs with relative paths (relative to repo root)
def save_csv(list_of_pairs, csv_path):
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filepath","class"])
        for p,c in list_of_pairs:
            writer.writerow([p, c])

save_csv(all_train_paths, os.path.join(OUT_ROOT, "train_split.csv"))
save_csv(all_test_paths, os.path.join(OUT_ROOT, "test_split.csv"))

print("Done. Splits saved under:", OUT_ROOT)
