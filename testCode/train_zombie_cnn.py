"""
train_zombie_cnn.py — Heatmap-based zombie detector with Huber loss.

The CNN outputs a spatial heatmap where peaks indicate zombie locations.
Huber loss is used for stable training on the heatmap values.

Output: (1, HEAT_H, HEAT_W) heatmap, values in [0,1].
Peaks in heatmap → zombie center positions.

Usage:
  python train_zombie_cnn.py --images dataset/clean/images
                             --labels dataset/clean/labels
                             --output zombie_cnn.pt --epochs 100 --lr 1e-3
"""

import argparse
import os
import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import scipy.ndimage


# ── Constants ─────────────────────────────────────────────────────────────────
SCREEN_W, SCREEN_H = 1280, 720
INPUT_W,  INPUT_H  = 320, 180
HEAT_W,   HEAT_H   = 80, 45     # output heatmap size (INPUT / 4)
MAX_ZOMBIES        = 4
GAUSSIAN_SIGMA     = 3.0        # spread of Gaussian blob per zombie
SLOT_DIM           = 4          # cx, cy, w, h (used externally)
ZOMBIE_DIM         = MAX_ZOMBIES * SLOT_DIM


# ── Gaussian heatmap generation ───────────────────────────────────────────────

def make_heatmap(boxes, heat_h: int, heat_w: int, sigma: float) -> np.ndarray:
    """
    Create a 2D Gaussian heatmap from YOLO-format boxes.
    Each zombie center gets a Gaussian blob. Values in [0,1].
    """
    heatmap = np.zeros((heat_h, heat_w), dtype=np.float32)
    for (cx, cy, w, h) in boxes:
        px = int(cx * heat_w)
        py = int(cy * heat_h)
        if 0 <= px < heat_w and 0 <= py < heat_h:
            heatmap[py, px] = 1.0

    if heatmap.max() > 0:
        heatmap = scipy.ndimage.gaussian_filter(heatmap, sigma=sigma)
        heatmap = heatmap / (heatmap.max() + 1e-6)

    return heatmap


# ── Dataset ───────────────────────────────────────────────────────────────────

class ZombieDataset(Dataset):
    """
    Returns:
        image:   (3, INPUT_H, INPUT_W) float32
        heatmap: (1, HEAT_H, HEAT_W)  float32
        sizes:   (2,) float32 — avg (w, h) of zombies in this frame
    """

    def __init__(self, image_dir: str, label_dir: str):
        self.image_paths = sorted(
            glob.glob(os.path.join(image_dir, "*.png")) +
            glob.glob(os.path.join(image_dir, "*.jpg"))
        )
        self.label_dir = label_dir
        if len(self.image_paths) == 0:
            raise ValueError(f"No images found in {image_dir}")
        print(f"[Dataset] Found {len(self.image_paths)} images.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        lbl_path = os.path.join(self.label_dir, Path(img_path).stem + ".txt")

        img = Image.open(img_path).convert("RGB").resize((INPUT_W, INPUT_H))
        img_t = torch.tensor(np.array(img), dtype=torch.float32) / 255.0
        img_t = img_t.permute(2, 0, 1)

        boxes = []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        _, cx, cy, w, h = map(float, parts)
                        boxes.append((cx, cy, w, h))

        heatmap   = make_heatmap(boxes, HEAT_H, HEAT_W, GAUSSIAN_SIGMA)
        heatmap_t = torch.tensor(heatmap).unsqueeze(0)  # (1, H, W)

        avg_w = float(np.mean([b[2] for b in boxes])) if boxes else 0.03
        avg_h = float(np.mean([b[3] for b in boxes])) if boxes else 0.05

        return img_t, heatmap_t, torch.tensor([avg_w, avg_h], dtype=torch.float32)


# ── Model ─────────────────────────────────────────────────────────────────────

class ZombieCNN(nn.Module):
    """
    Fully convolutional encoder that outputs a spatial heatmap.
    No global pooling — spatial information is preserved throughout.

    Input:  (B, 3, 180, 320)
    Output: (B, 1, 45, 80) heatmap, values in [0,1]
    """

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3,  32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # → (32, 90, 160)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),   # → (64, 45, 80)
        )
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32,  1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc1(x)
        x = self.enc2(x)
        return self.head(x)   # (B, 1, 45, 80)


# ── Peak finding ──────────────────────────────────────────────────────────────

def find_peaks(heatmap: np.ndarray, threshold: float = 0.3,
               min_distance: int = 5) -> list:
    """
    Find local maxima in heatmap above threshold.
    Returns list of (cx, cy) normalised to [0,1], sorted by confidence.
    """
    from scipy.ndimage import maximum_filter, label as scipy_label

    h, w = heatmap.shape
    local_max    = (heatmap == maximum_filter(heatmap, size=min_distance))
    above_thresh = heatmap > threshold
    peaks_mask   = local_max & above_thresh

    labeled, n = scipy_label(peaks_mask)
    peaks = []
    for i in range(1, n + 1):
        coords = np.argwhere(labeled == i)
        cy_px, cx_px = coords.mean(axis=0)
        conf = float(heatmap[int(cy_px), int(cx_px)])
        peaks.append((cx_px / w, cy_px / h, conf))

    peaks.sort(key=lambda p: -p[2])
    return [(p[0], p[1]) for p in peaks[:MAX_ZOMBIES]]


# ── Training ──────────────────────────────────────────────────────────────────


class WeightedMSELoss(nn.Module):
    """
    MSE loss with higher weight on zombie pixels to handle
    the severe class imbalance (99% background vs 1% zombie).
    """
    def __init__(self, pos_weight: float = 50.0):
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Weight matrix: pos_weight where zombie, 1.0 where background
        weights = torch.ones_like(target)
        weights[target > 0.1] = self.pos_weight

        loss = weights * (pred - target) ** 2
        return loss.mean()

def train(
    image_dir:   str,
    label_dir:   str,
    output_path: str,
    epochs:      int   = 100,
    batch_size:  int   = 8,
    lr:          float = 1e-3,
    val_split:   float = 0.15,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    dataset = ZombieDataset(image_dir, label_dir)
    n_val   = max(1, int(len(dataset) * val_split))
    n_train = len(dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"[Train] {n_train} training, {n_val} validation samples.")

    model     = ZombieCNN().to(device)
    criterion = WeightedMSELoss(pos_weight=50.0)   # stable, works well on [0,1] heatmap values
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=30, T_mult=2
    )

    best_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for imgs, heatmaps, _ in train_loader:
            imgs     = imgs.to(device)
            heatmaps = heatmaps.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), heatmaps)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, heatmaps, _ in val_loader:
                imgs     = imgs.to(device)
                heatmaps = heatmaps.to(device)
                val_loss += criterion(model(imgs), heatmaps).item()
        val_loss /= len(val_loader)

        scheduler.step()
        print(f"epoch {epoch+1:03d}/{epochs} | train: {train_loss:.6f} | val: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), output_path)
            print(f"  └─ best model saved → {output_path}")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.6f}")
    print(f"Model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",  required=True)
    parser.add_argument("--labels",  required=True)
    parser.add_argument("--output",  default="zombie_cnn.pt")
    parser.add_argument("--epochs",  type=int,   default=100)
    parser.add_argument("--batch",   type=int,   default=8)
    parser.add_argument("--lr",      type=float, default=1e-3)
    args = parser.parse_args()

    train(
        image_dir=args.images,
        label_dir=args.labels,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
    )