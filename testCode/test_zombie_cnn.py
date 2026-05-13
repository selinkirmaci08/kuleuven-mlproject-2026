"""
test_zombie_cnn.py — Test heatmap-based zombie CNN.

Usage:
    python test_zombie_cnn.py --images dataset/images --labels dataset/labels
    python test_zombie_cnn.py --images dataset/images --labels dataset/labels --precision
    python test_zombie_cnn.py --images dataset/images --labels dataset/labels --save results/
"""

import argparse
import os
import glob
import random
from pathlib import Path

import numpy as np
import torch
import cv2
from PIL import Image

from testCode.train_zombie_cnn import ZombieCNN, INPUT_W, INPUT_H, find_peaks


def load_model(model_path: str) -> ZombieCNN:
    model = ZombieCNN()
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    print(f"[Test] Model loaded from {model_path}")
    return model


def load_gt_boxes(label_path: str):
    boxes = []
    if os.path.exists(label_path):
        with open(label_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    _, cx, cy, w, h = map(float, parts)
                    boxes.append((cx, cy, w, h))
    return boxes


def get_avg_box_size(label_path: str):
    """Get average zombie box size from labels for display."""
    boxes = load_gt_boxes(label_path)
    if boxes:
        return np.mean([b[2] for b in boxes]), np.mean([b[3] for b in boxes])
    return 0.03, 0.05


def predict(model: ZombieCNN, img_path: str, threshold: float = 0.3):
    """
    Run model, find peaks in heatmap.
    Returns list of (cx, cy) normalised positions and the raw heatmap.
    """
    img = Image.open(img_path).convert("RGB").resize((INPUT_W, INPUT_H))
    img_t = torch.tensor(np.array(img), dtype=torch.float32).permute(2,0,1).unsqueeze(0) / 255.0

    with torch.no_grad():
        heatmap = model(img_t).squeeze().numpy()  # (HEAT_H, HEAT_W)

    peaks = find_peaks(heatmap, threshold=threshold)
    return peaks, heatmap


def draw_results(img_path: str, label_path: str, peaks, heatmap,
                 save_path: str = None):
    img = cv2.imread(img_path)
    ih, iw = img.shape[:2]

    # Overlay heatmap (red channel)
    hmap_resized = cv2.resize(heatmap, (iw, ih))
    hmap_color = np.zeros_like(img, dtype=np.float32)
    hmap_color[:,:,2] = hmap_resized * 255  # red channel
    img = cv2.addWeighted(img.astype(np.float32), 0.7,
                          hmap_color, 0.3, 0).astype(np.uint8)

    # Get avg box size for display
    avg_w, avg_h = 0.0227, 0.0431

    # Ground truth — GREEN
    for (cx, cy, bw, bh) in load_gt_boxes(label_path):
        x1 = int((cx - bw/2) * iw);  y1 = int((cy - bh/2) * ih)
        x2 = int((cx + bw/2) * iw);  y2 = int((cy + bh/2) * ih)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Predictions — BLUE (use avg box size for display)
    for (cx, cy) in peaks:
        bw, bh = avg_w, avg_h
        x1 = int((cx - bw/2) * iw);  y1 = int((cy - bh/2) * ih)
        x2 = int((cx + bw/2) * iw);  y2 = int((cy + bh/2) * ih)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.circle(img, (int(cx*iw), int(cy*ih)), 4, (255, 0, 0), -1)

    gt_count = len(load_gt_boxes(label_path))
    cv2.putText(img, f"GT:{gt_count}  Pred:{len(peaks)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.putText(img, "GREEN=GT  BLUE=Pred", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)

    if save_path:
        cv2.imwrite(save_path, img)
        print(f"  Saved → {save_path}")
    else:
        cv2.imshow("Zombie Heatmap Detection", img)
        key = cv2.waitKey(0)
        cv2.destroyAllWindows()
        if key == ord('q'):
            return False
    return True


def compute_iou_center(pred_cx, pred_cy, pred_w, pred_h,
                        gt_cx, gt_cy, gt_w, gt_h):
    px1 = pred_cx - pred_w/2;  py1 = pred_cy - pred_h/2
    px2 = pred_cx + pred_w/2;  py2 = pred_cy + pred_h/2
    gx1 = gt_cx   - gt_w/2;    gy1 = gt_cy   - gt_h/2
    gx2 = gt_cx   + gt_w/2;    gy2 = gt_cy   + gt_h/2
    ix1 = max(px1,gx1);  iy1 = max(py1,gy1)
    ix2 = min(px2,gx2);  iy2 = min(py2,gy2)
    inter = max(0,ix2-ix1)*max(0,iy2-iy1)
    pa = (px2-px1)*(py2-py1);  ga = (gx2-gx1)*(gy2-gy1)
    return inter / (pa + ga - inter + 1e-6)


def evaluate_precision(model, image_dir, label_dir,
                       threshold=0.3, iou_threshold=0.5):
    image_paths = sorted(
        glob.glob(os.path.join(image_dir, "*.png")) +
        glob.glob(os.path.join(image_dir, "*.jpg"))
    )
    total_gt = 0;  total_correct = 0

    for img_path in image_paths:
        label_path  = os.path.join(label_dir, Path(img_path).stem + ".txt")
        gt_boxes    = load_gt_boxes(label_path)
        peaks, _    = predict(model, img_path, threshold)
        total_gt   += len(gt_boxes)
        matched     = set()

        avg_w, avg_h = 0.0227, 0.0431

        for (px, py) in peaks:
            for i, (gx, gy, gw, gh) in enumerate(gt_boxes):
                if i in matched:
                    continue
                iou = compute_iou_center(px, py, avg_w, avg_h, gx, gy, gw, gh)
                if iou >= iou_threshold:
                    total_correct += 1
                    matched.add(i)
                    break

    precision = total_correct/total_gt if total_gt > 0 else 0.0
    print(f"\n[Evaluation] threshold={threshold} | iou_threshold={iou_threshold}")
    print(f"  Total zombies (GT): {total_gt}")
    print(f"  Correctly detected: {total_correct}")
    print(f"  Precision:          {precision:.3f} ({precision*100:.1f}%)")
    return precision


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images",    required=True)
    parser.add_argument("--labels",    required=True)
    parser.add_argument("--model",     default="zombie_cnn.pt")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--iou",       type=float, default=0.5)
    parser.add_argument("--n",         type=int,   default=10)
    parser.add_argument("--save",      help="Save result images here")
    parser.add_argument("--precision", action="store_true")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.precision:
        evaluate_precision(model, args.images, args.labels,
                           args.threshold, args.iou)
        return

    image_paths = sorted(
        glob.glob(os.path.join(args.images, "*.png")) +
        glob.glob(os.path.join(args.images, "*.jpg"))
    )
    sample = random.sample(image_paths, min(args.n, len(image_paths)))
    if args.save:
        os.makedirs(args.save, exist_ok=True)

    for img_path in sample:
        label_path      = os.path.join(args.labels, Path(img_path).stem + ".txt")
        peaks, heatmap  = predict(model, img_path, args.threshold)
        gt_count        = len(load_gt_boxes(label_path))
        print(f"{Path(img_path).name} | GT: {gt_count} | Pred: {len(peaks)}")
        save_path = os.path.join(args.save, Path(img_path).name) if args.save else None
        if not draw_results(img_path, label_path, peaks, heatmap, save_path):
            break


if __name__ == "__main__":
    main()