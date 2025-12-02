#!/usr/bin/env python3
"""
Resize images from data/raw/train and data/raw/test into data/resized/train and data/resized/test,
keeping the class subfolders and copying auxiliary files (.xml, .txt) alongside images.

Usage:
    python scripts/resize_images.py
"""

import os
import glob
import shutil
import cv2 as cv

# -------- CONFIG --------
SRC_ROOT = "data/raw"        # expected subfolders: train/ and test/
DST_ROOT = "data/resized"    # will create train/ and test/ under this
MAX_SIDE = 480               # max length of the largest side (pixels)
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
COPY_EXTRA_FILES = True      # copy .xml and .txt found next to images
# ------------------------

def is_image(fname):
    return fname.lower().endswith(IMAGE_EXTS)

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def process_split(split):
    src_split = os.path.join(SRC_ROOT, split)
    if not os.path.isdir(src_split):
        print(f"[SKIP] no folder: {src_split}")
        return

    # iterate class subfolders
    for cls in sorted(os.listdir(src_split)):
        cls_src = os.path.join(src_split, cls)
        if not os.path.isdir(cls_src):
            continue
        cls_dst = os.path.join(DST_ROOT, split, cls)
        ensure_dir(cls_dst)

        # find image files (and copy auxiliary files later)
        for fname in sorted(os.listdir(cls_src)):
            src_path = os.path.join(cls_src, fname)
            dst_path = os.path.join(cls_dst, fname)

            # skip directories
            if os.path.isdir(src_path):
                continue

            # image file: resize & save
            if is_image(fname):
                img = cv.imread(src_path)
                if img is None:
                    print(f"[WARN] cannot read image {src_path}, skipping")
                    continue
                h, w = img.shape[:2]
                scale = MAX_SIDE / max(h, w)
                if scale < 1.0:
                    new_w, new_h = int(w * scale), int(h * scale)
                    resized = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_AREA)
                else:
                    resized = img.copy()
                # save with same filename
                cv.imwrite(dst_path, resized)
            else:
                # not an image: optionally copy xml/txt
                if COPY_EXTRA_FILES and fname.lower().endswith((".xml", ".txt", ".json", ".csv")):
                    try:
                        shutil.copy(src_path, dst_path)
                    except Exception as e:
                        print(f"[WARN] failed to copy {src_path}: {e}")
                else:
                    # ignore other files like zips by default
                    continue

        print(f"[DONE] processed class '{cls}' -> {cls_dst}")

def main():
    print("SRC_ROOT:", SRC_ROOT)
    print("DST_ROOT:", DST_ROOT)
    ensure_dir(DST_ROOT)
    for split in ("train", "test"):
        print(f"Processing split: {split}")
        process_split(split)
    print("All done. Resized dataset in:", DST_ROOT)
    print("Tip: update your config root to:", DST_ROOT)

if __name__ == "__main__":
    main()
