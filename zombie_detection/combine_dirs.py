"""
combine_dataset.py — Combine multi-distortion datasets and split into train/val.

Takes images from each distortion level, combines them, splits 85/15 train/val,
and writes a dataset.yaml for YOLO training.

Usage:
    python combine_dataset.py
"""

import shutil
import glob
import os
import random
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
COMBINED_DIR = "../dataset/combined"
VAL_SPLIT    = 0.10   # 5% validation
SEED         = 42

# Max images per distortion level
LIMITS = {
    0: 1500,   # clean
    1: 1500,   # dist1
    2: 1700,   # dist2
    3: 1500,   # dist3
    4: 2000,  # dist4
    5: 2000,  # dist5
}

# ── Setup dirs ────────────────────────────────────────────────────────────────
img_dir = os.path.join(COMBINED_DIR, "images")
lbl_dir = os.path.join(COMBINED_DIR, "labels")
os.makedirs(img_dir, exist_ok=True)
os.makedirs(lbl_dir, exist_ok=True)

# ── Copy images from each distortion level ────────────────────────────────────
all_stems = []  # track all copied image stems

for dist, _ in LIMITS.items():
    src = "dataset/clean" if dist == 0 else f"dataset/dist{dist}"
    files = sorted(glob.glob(f"{src}/images/*.png"))

    copied = 0
    for f in files:
        stem = f"d{dist}_{os.path.basename(f)}"
        dst_img = os.path.join(img_dir, stem)
        shutil.copy(f, dst_img)

        lbl_src = f.replace("images", "labels").replace(".png", ".txt")
        if os.path.exists(lbl_src):
            shutil.copy(lbl_src, os.path.join(lbl_dir, stem.replace(".png", ".txt")))

        all_stems.append(stem)
        copied += 1

    print(f"Distortion {dist}: copied {copied} images")

print(f"\nTotal images combined: {len(all_stems)}")

# ── Train / val split ─────────────────────────────────────────────────────────
random.seed(SEED)
random.shuffle(all_stems)

n_val   = int(len(all_stems) * VAL_SPLIT)
n_train = len(all_stems) - n_val

train_stems = all_stems[n_val:]   # first n_train
val_stems   = all_stems[:n_val]   # first n_val

# Write train.txt and val.txt with absolute paths
abs_img_dir = os.path.abspath(img_dir)

train_txt = os.path.join(COMBINED_DIR, "train.txt")
val_txt   = os.path.join(COMBINED_DIR, "val.txt")

with open(train_txt, "w") as f:
    f.write("\n".join(os.path.join(abs_img_dir, s) for s in train_stems))

with open(val_txt, "w") as f:
    f.write("\n".join(os.path.join(abs_img_dir, s) for s in val_stems))

print(f"Train: {n_train} images → {train_txt}")
print(f"Val:   {n_val}   images → {val_txt}")

# ── Write dataset.yaml ────────────────────────────────────────────────────────
yaml_path = os.path.join(COMBINED_DIR, "dataset.yaml")
yaml_content = (
    f"path: {os.path.abspath(COMBINED_DIR)}\n"
    f"train: train.txt\n"
    f"val:   val.txt\n"
    f"nc: 1\n"
    f"names: ['zombie']\n"
)
with open(yaml_path, "w") as f:
    f.write(yaml_content)

print(f"\ndataset.yaml written → {yaml_path}")
print("\nNow train YOLO with:")
print(f"  yolo detect train data={yaml_path} model=yolo11n.pt epochs=100 imgsz=640 batch=8 workers=2 patience=20")