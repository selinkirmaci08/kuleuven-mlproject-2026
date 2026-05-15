#!/usr/bin/env python3
"""
collect_dataset.py  —  Automatic zombie detection dataset generator for KAZ.

Saves images as PNG and labels in YOLO format (normalized cx cy w h),
compatible with Ultralytics YOLO training directly.

Output structure:
  <out_dir>/
    images/  000000.png  000001.png  ...
    labels/  000000.txt  000001.txt  ...

Label format (YOLO):
  One line per zombie: 0 cx cy w h
  All values normalised to [0, 1] relative to image size.
  Class 0 = zombie.

Usage:
  python collect_dataset.py --episodes 100 --distortion 0
  python collect_dataset.py --episodes 100 --distortion 3
  python collect_dataset.py --episodes 100 --distortion 5 --out /custom/path
"""

import argparse
import os
import random
from pathlib import Path

import numpy as np
from PIL import Image

from utils import create_environment

# ── Paths ─────────────────────────────────────────────────────────────────────
_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(_DIR)  # go up one level from zombie_detection to project root

SCREEN_W = 1280
SCREEN_H = 720


def unwrap_to_raw(env):
    """
    Follow the wrapper chain until we find the raw KAZ env with zombie_list.
    Tries .aec_env before .env at each level (needed to get past black_death_par).
    """
    cur = env
    seen = set()
    while cur is not None:
        if id(cur) in seen:
            break
        seen.add(id(cur))
        if hasattr(cur, "zombie_list"):
            return cur
        nxt = getattr(cur, "aec_env", None) or getattr(cur, "env", None)
        if nxt is None or nxt is cur:
            break
        cur = nxt
    return cur


def get_zombie_boxes(env):
    """
    Returns list of (x_min, y_min, w, h) in pixel coordinates.
    """
    boxes = []
    raw = unwrap_to_raw(env)
    if hasattr(raw, "zombie_list"):
        for zombie in raw.zombie_list:
            r = zombie.rect
            boxes.append((float(r.x), float(r.y), float(r.width), float(r.height)))
    else:
        print(f"  [warn] zombie_list not found, reached: {type(raw).__name__}")
    return boxes


def boxes_to_yolo(boxes, img_w, img_h):
    """
    Convert pixel [x_min, y_min, w, h] boxes to YOLO format lines:
        0 cx_norm cy_norm w_norm h_norm
    """
    lines = []
    for x, y, w, h in boxes:
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        wn = w / img_w
        hn = h / img_h
        # Clamp to [0, 1] to handle edge sprites
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        wn = max(0.0, min(1.0, wn))
        hn = max(0.0, min(1.0, hn))
        lines.append(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
    return lines


def collect(
    num_episodes: int,
    distortion_level: int,
    out_dir: Path,
    master_seed: int = 42,
    max_steps: int = 900,
    skip_empty: bool = True,
):
    img_dir = out_dir / "images"
    lbl_dir = out_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    random.seed(master_seed)
    seeds = [random.randint(0, 2**32 - 1) for _ in range(num_episodes)]
    sample_idx = 0

    for ep, seed in enumerate(seeds):
        print(f"Episode {ep + 1}/{num_episodes}  seed={seed}")
        env = create_environment(distortion_level=distortion_level, render_mode=None)
        env.reset(seed=seed)

        step = 0
        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()
            step += 1

            if termination or truncation:
                env.step(None)
                continue

            if step >= max_steps:
                env.step(None)
                break

            # Save one frame per cycle when archer_0 acts
            if agent == "archer_0":
                raw_obs = env.observe(agent)        # (720, 1280, 3) uint8
                boxes   = get_zombie_boxes(env)

                if not skip_empty or len(boxes) > 0:
                    stem = f"{sample_idx:06d}"

                    # Save PNG
                    Image.fromarray(raw_obs).save(img_dir / f"{stem}.png")

                    # Save YOLO label (empty file if no zombies)
                    yolo_lines = boxes_to_yolo(boxes, SCREEN_W, SCREEN_H)
                    (lbl_dir / f"{stem}.txt").write_text("\n".join(yolo_lines))

                    sample_idx += 1

            env.step(env.action_space(agent).sample())

        env.close()
        print(f"  → {sample_idx} samples saved so far")

    # Write a dataset.yaml for Ultralytics YOLO
    yaml_path = out_dir / "dataset.yaml"
    yaml_path.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: 1\n"
        f"names: ['zombie']\n"
    )
    print(f"\nDone. {sample_idx} samples saved to {out_dir}/")
    print(f"Dataset config written to {yaml_path}")
    print(f"\nTo train YOLO:")
    print(f"  yolo detect train data={yaml_path} model=yolo11n.pt epochs=50 imgsz=640")


def main():
    parser = argparse.ArgumentParser(description="Collect auto-labelled KAZ zombie dataset")
    parser.add_argument("--episodes",   type=int, default=50)
    parser.add_argument("--distortion", type=int, default=0, choices=range(6))
    parser.add_argument("--out",        type=str, default=None)
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--max-steps",  type=int, default=900)
    parser.add_argument("--keep-empty", action="store_true",
                        help="Also save frames with no zombies (useful for hard negative mining)")
    args = parser.parse_args()

    if args.out is None:
        dist_name = "clean" if args.distortion == 0 else f"dist{args.distortion}"
        out_dir = Path(ROOT_DIR) / "dataset" / dist_name
    else:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = Path(ROOT_DIR) / out_dir

    print(f"Output directory: {out_dir}")

    collect(
        num_episodes=args.episodes,
        distortion_level=args.distortion,
        out_dir=out_dir,
        master_seed=args.seed,
        max_steps=args.max_steps,
        skip_empty=not args.keep_empty,
    )


if __name__ == "__main__":
    main()