import torch
from PIL import Image
import numpy as np
from train_zombie_cnn import ZombieCNN, INPUT_W, INPUT_H, MAX_ZOMBIES, SLOT_DIM


import glob, os
import numpy as np

label_dir = "dataset/clean/labels"
all_w, all_h = [], []
for f in glob.glob(os.path.join(label_dir, "*.txt")):
    with open(f) as fp:
        for line in fp:
            parts = line.strip().split()
            if len(parts) == 5:
                all_w.append(float(parts[3]))
                all_h.append(float(parts[4]))

print(f"avg w: {np.mean(all_w):.4f}  avg h: {np.mean(all_h):.4f}")
print(f"std w: {np.std(all_w):.4f}   std h: {np.std(all_h):.4f}")