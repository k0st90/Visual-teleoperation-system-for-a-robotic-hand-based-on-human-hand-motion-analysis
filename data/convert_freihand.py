"""
Convert FreiHand training_xyz.json to kps_freihand.npz for mlp_ss training.

Usage:
    python data/convert_freihand.py
"""

import json
import numpy as np
import os

src = os.path.join(os.path.dirname(__file__), "training_xyz.json")
dst = os.path.join(os.path.dirname(__file__), "kps_freihand.npz")

print(f"Loading {src}...")
data = json.load(open(src))

kps = np.array(data, dtype=np.float32)   # (N, 21, 3) in camera frame
kps = kps - kps[:, 0:1, :]               # to wrist frame

print(f"Samples: {len(kps)}  shape: {kps.shape}")
print(f"Range: [{kps.min():.4f}, {kps.max():.4f}]")

np.savez_compressed(dst, kps=kps)
print(f"Saved → {dst}")
