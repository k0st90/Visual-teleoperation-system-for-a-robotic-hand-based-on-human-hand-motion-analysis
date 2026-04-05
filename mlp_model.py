"""
MLP retargeter: replaces NLopt optimizer at inference.

Input:  human keypoints in wrist frame  (21, 3) → flattened (63,)
Output: robot joint angles qpos         (n_doa,)

Output is scaled through sigmoid to stay within joint limits [lb, ub].
"""

import numpy as np
import torch
import torch.nn as nn


class RetargeterMLP(nn.Module):
    """
    63 → 256 → 256 → n_doa
    LayerNorm + ReLU hidden layers.
    Output clamped to [joint_lb, joint_ub] via sigmoid scaling.
    """

    def __init__(self, n_doa: int, joint_lb: np.ndarray, joint_ub: np.ndarray,
                 hidden: int = 256):
        super().__init__()
        self.n_doa = n_doa
        self.register_buffer("joint_lb", torch.tensor(joint_lb, dtype=torch.float32))
        self.register_buffer("joint_ub", torch.tensor(joint_ub, dtype=torch.float32))

        self.net = nn.Sequential(
            nn.Linear(63, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_doa),
        )

    def forward(self, kps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kps: (B, 21, 3) or (B, 63)  — keypoints in wrist frame (NOT scaled)
        Returns:
            qpos: (B, n_doa) — joint angles within [lb, ub]
        """
        x = kps.reshape(kps.shape[0], -1)   # (B, 63)
        raw = self.net(x)                    # (B, n_doa)
        # sigmoid maps raw → (0, 1) → [lb, ub]
        qpos = torch.sigmoid(raw) * (self.joint_ub - self.joint_lb) + self.joint_lb
        return qpos

    def predict(self, kps_np: np.ndarray) -> np.ndarray:
        """Single-sample inference (no grad). kps_np: (21, 3)"""
        self.eval()
        with torch.no_grad():
            x = torch.tensor(kps_np, dtype=torch.float32).unsqueeze(0)
            qpos = self.forward(x)
        return qpos.squeeze(0).cpu().numpy()

    @staticmethod
    def from_checkpoint(path: str, device: str = "cpu") -> "RetargeterMLP":
        ckpt = torch.load(path, map_location=device)
        model = RetargeterMLP(
            n_doa=ckpt["n_doa"],
            joint_lb=ckpt["joint_lb"],
            joint_ub=ckpt["joint_ub"],
            hidden=ckpt.get("hidden", 256),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        return model

    def save_checkpoint(self, path: str):
        torch.save({
            "n_doa":       self.n_doa,
            "joint_lb":    self.joint_lb.cpu().numpy(),
            "joint_ub":    self.joint_ub.cpu().numpy(),
            "hidden":      self.net[0].out_features,
            "state_dict":  self.state_dict(),
        }, path)
