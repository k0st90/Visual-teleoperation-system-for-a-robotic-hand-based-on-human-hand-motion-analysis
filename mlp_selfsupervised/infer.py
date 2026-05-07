"""
MLP inference wrapper — drop-in replacement for HandRetargeter.retarget().

Usage in main.py:
    from mlp_selfsupervised.infer import MLPRetargeter
    retargeter = MLPRetargeter("checkpoints/mlp_ss_leap_best.pt")
    qpos = retargeter.retarget(joint_pos)   # same API as HandRetargeter
"""

import os
import sys
import time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mlp_selfsupervised.mlp_model import RetargeterMLP


class OneEuroFilter:
    """Per-dimension One-Euro Filter for joint angle smoothing."""

    def __init__(self, min_cutoff: float = 0.5, beta: float = 0.1, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta       = beta
        self.d_cutoff   = d_cutoff
        self._x         = None
        self._dx        = None
        self._t         = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        t = time.monotonic()
        if self._x is None:
            self._x  = x.copy()
            self._dx = np.zeros_like(x)
            self._t  = t
            return x.copy()

        dt = max(t - self._t, 1e-6)
        self._t = t

        # derivative
        dx_raw = (x - self._x) / dt
        alpha_d = self._alpha(dt, self.d_cutoff)
        self._dx = alpha_d * dx_raw + (1.0 - alpha_d) * self._dx

        # signal
        cutoff = self.min_cutoff + self.beta * np.abs(self._dx)
        alpha  = self._alpha(dt, cutoff)
        self._x = alpha * x + (1.0 - alpha) * self._x
        return self._x.copy()

    @staticmethod
    def _alpha(dt: float, cutoff: np.ndarray) -> np.ndarray:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)


class MLPRetargeter:
    def __init__(self, checkpoint_path: str, device: str = "cpu",
                 min_cutoff: float = 0.3, beta: float = 0.02):
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.model = RetargeterMLP(
            n_doa=ckpt["n_doa"],
            joint_lb=ckpt["joint_lb"],
            joint_ub=ckpt["joint_ub"],
            hidden=ckpt.get("hidden", 256),
        ).to(device)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.device   = device
        self.kps_mean = ckpt.get("kps_mean", None)
        self.kps_std  = ckpt.get("kps_std",  None)
        self.filter   = OneEuroFilter(min_cutoff=min_cutoff, beta=beta)
        print(f"MLPRetargeter loaded: {checkpoint_path}  "
              f"OneEuroFilter(min_cutoff={min_cutoff}, beta={beta})")

    def retarget(self, hand_kps_in_wrist: np.ndarray) -> np.ndarray:
        """
        Args:
            hand_kps_in_wrist: (21, 3) keypoints in wrist frame (unscaled)
        Returns:
            qpos: (n_doa,)
        """
        flat = hand_kps_in_wrist.flatten().astype(np.float32)
        if self.kps_mean is not None:
            flat = (flat - self.kps_mean) / self.kps_std
        with torch.no_grad():
            x = torch.tensor(flat, dtype=torch.float32).unsqueeze(0).to(self.device)
            qpos = self.model(x).squeeze(0).cpu().numpy()

        return self.filter(qpos)
