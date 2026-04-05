"""
Train the MLP retargeter.

Loss = imitation_loss + lambda_physics * physics_loss

  imitation_loss:  MSE(qpos_pred, qpos_opt)
                   — supervised signal from the NLopt optimizer

  physics_loss:    links_vec_cost(FK(qpos_pred), ref_links_vec)
                   + joint_pos_cost(qpos_pred)
                   — same loss functions as the optimizer, backprop through pinocchio
                     via manual Jacobians (same as optimizer.py)

Usage:
    # single session
    python train_mlp.py --hand allegro --data data/allegro_session1.npz

    # multiple sessions
    python train_mlp.py --hand allegro --data data/allegro_*.npz

    # physics loss disabled (pure imitation)
    python train_mlp.py --hand allegro --data data/allegro_session1.npz --lambda-physics 0

Output:
    checkpoints/mlp_<hand>_best.pt   — best val loss checkpoint
    checkpoints/mlp_<hand>_last.pt   — last epoch checkpoint
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

from mlp_model import RetargeterMLP
from hand_retargeter import HandRetargeter


HAND_CONFIGS = {
    "leap":    "configs/leap_hand_right.yml",
    "allegro": "configs/allegro_hand_right.yml",
    "shadow":  "configs/shadow_hand_right.yml",
}


# ─── dataset ──────────────────────────────────────────────────────────────────

class RetargetDataset(Dataset):
    """
    Loads .npz files saved by collect_data.py.
    Each sample: (kps (21,3), qpos (n_doa,))
    """
    def __init__(self, paths: list[str], kps_mean=None, kps_std=None):
        kps_list, qpos_list = [], []
        for p in paths:
            d = np.load(p)
            kps_list.append(d["kps"])
            qpos_list.append(d["qpos"])
            print(f"  loaded {d['kps'].shape[0]} samples from {p}")

        self.kps  = np.concatenate(kps_list,  axis=0).astype(np.float32)  # (N, 21, 3)
        self.qpos = np.concatenate(qpos_list, axis=0).astype(np.float32)  # (N, n_doa)

        # normalise keypoints: zero-mean, unit-std per coordinate
        flat = self.kps.reshape(len(self.kps), -1)   # (N, 63)
        if kps_mean is None:
            self.kps_mean = flat.mean(axis=0)
            self.kps_std  = flat.std(axis=0) + 1e-8
        else:
            self.kps_mean = kps_mean
            self.kps_std  = kps_std

        self.kps_flat_norm = (flat - self.kps_mean) / self.kps_std  # (N, 63)
        print(f"Total samples: {len(self.kps)}")

    def __len__(self):
        return len(self.kps)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.kps_flat_norm[idx]),   # (63,)  normalised
            torch.from_numpy(self.kps[idx]),              # (21,3) raw (for physics loss)
            torch.from_numpy(self.qpos[idx]),             # (n_doa,)
        )


# ─── physics loss (same as VectorWristJointOptimizer) ─────────────────────────

class PhysicsLoss:
    """
    Computes optimizer-style losses for a batch of predicted qpos.
    Uses pinocchio FK (non-differentiable) + manual Jacobians.

    For batch training: iterates over samples in the batch (pinocchio is not
    vectorisable). Acceptable because this runs offline, not at 20 FPS.
    """

    def __init__(self, retargeter: HandRetargeter):
        self.retargeter = retargeter
        optimizer = retargeter.optimizer
        self.opt = optimizer
        self.huber = nn.SmoothL1Loss(beta=retargeter.huber_delta, reduction="mean")

    def compute(self, kps_raw_batch: torch.Tensor, qpos_pred_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kps_raw_batch:   (B, 21, 3) — raw (unscaled) keypoints
            qpos_pred_batch: (B, n_doa) — MLP output (requires_grad=True on the graph)
        Returns:
            scalar physics loss
        """
        B = kps_raw_batch.shape[0]
        total = torch.tensor(0.0, dtype=torch.float32, requires_grad=False)

        for i in range(B):
            kps_np  = kps_raw_batch[i].detach().cpu().numpy()    # (21, 3)
            qpos_np = qpos_pred_batch[i].detach().cpu().numpy()  # (n_doa,)

            # ── ref_values from retargeter logic ──────────────────────────
            ref_values = self.retargeter._build_ref_values(kps_np)

            ref_links_vec   = torch.as_tensor(ref_values["links_vec"],          dtype=torch.float32)
            weight_links_vec = torch.as_tensor(ref_values["weights"]["links_vec"], dtype=torch.float32)
            weight_joint_pos = torch.as_tensor(ref_values["weights"]["joint_pos"], dtype=torch.float32)

            # ── FK ─────────────────────────────────────────────────────────
            qpos_dof = self.opt.robot_adaptor.forward_qpos(qpos_np)
            self.opt.robot_model.compute_forward_kinematics(qpos_dof)

            links_pos_list = [
                self.opt.robot_model.get_frame_pose(name)[:3, 3]
                for name in self.opt.computed_links_name
            ]
            links_pos = torch.tensor(np.stack(links_pos_list), dtype=torch.float32)

            # ── links_vec_cost ─────────────────────────────────────────────
            origin_pos = links_pos[self.opt.origin_links_idx]
            task_pos   = links_pos[self.opt.task_links_idx]
            links_vec  = task_pos - origin_pos
            links_vec_err = torch.norm(links_vec - ref_links_vec, dim=-1)
            links_vec_cost = self.huber(weight_links_vec * links_vec_err,
                                        torch.zeros_like(links_vec_err))

            # ── joint_pos_cost (regularisation) ───────────────────────────
            qpos_torch = qpos_pred_batch[i]
            joint_pos_cost = self.huber(weight_joint_pos * qpos_torch,
                                        torch.zeros_like(qpos_torch))

            total = total + links_vec_cost + joint_pos_cost

        return total / B


# ─── training ─────────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── load retargeter (for joint limits + physics loss) ──────────────────
    yml_path = HAND_CONFIGS[args.hand]
    print(f"Loading retargeter: {yml_path}")
    retargeter = HandRetargeter(yml_path=yml_path)
    n_doa = retargeter.robot_adaptor.doa
    joint_lb = retargeter.optimizer.joint_limits[:, 0]
    joint_ub = retargeter.optimizer.joint_limits[:, 1]
    print(f"n_doa={n_doa}  joint_lb={np.round(joint_lb,2)}  joint_ub={np.round(joint_ub,2)}")

    # ── dataset ────────────────────────────────────────────────────────────
    data_paths = []
    for pattern in args.data:
        data_paths.extend(sorted(glob.glob(pattern)))
    if not data_paths:
        print(f"ERROR: no .npz files found: {args.data}")
        sys.exit(1)

    dataset = RetargetDataset(data_paths)
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device=="cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0)
    print(f"Train: {n_train}  Val: {n_val}")

    # ── model ──────────────────────────────────────────────────────────────
    model = RetargeterMLP(n_doa=n_doa, joint_lb=joint_lb, joint_ub=joint_ub,
                          hidden=args.hidden).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer_opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer_opt, T_max=args.epochs, eta_min=args.lr * 0.01)

    imitation_loss_fn = nn.MSELoss()
    physics_loss_fn   = PhysicsLoss(retargeter) if args.lambda_physics > 0 else None

    os.makedirs("checkpoints", exist_ok=True)
    best_val = float("inf")

    # ── epoch loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_imitate = train_physics = 0.0

        for kps_norm, kps_raw, qpos_opt in train_loader:
            kps_norm = kps_norm.to(device)
            kps_raw  = kps_raw.to(device)
            qpos_opt = qpos_opt.to(device)

            qpos_pred = model(kps_norm)

            L_imitate = imitation_loss_fn(qpos_pred, qpos_opt)
            L_total   = L_imitate

            if physics_loss_fn is not None and args.lambda_physics > 0:
                L_physics = physics_loss_fn.compute(kps_raw.cpu(), qpos_pred)
                L_total   = L_total + args.lambda_physics * L_physics
                train_physics += L_physics.item()

            optimizer_opt.zero_grad()
            L_total.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer_opt.step()

            train_imitate += L_imitate.item()

        scheduler.step()

        # ── validation ─────────────────────────────────────────────────────
        model.eval()
        val_imitate = 0.0
        with torch.no_grad():
            for kps_norm, kps_raw, qpos_opt in val_loader:
                kps_norm = kps_norm.to(device)
                qpos_opt = qpos_opt.to(device)
                qpos_pred = model(kps_norm)
                val_imitate += imitation_loss_fn(qpos_pred, qpos_opt).item()

        n_train_batches = len(train_loader)
        n_val_batches   = len(val_loader)
        avg_train = train_imitate / n_train_batches
        avg_val   = val_imitate   / n_val_batches
        avg_phys  = train_physics / n_train_batches if physics_loss_fn else 0.0
        elapsed   = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train_imitate={avg_train:.5f}  physics={avg_phys:.5f}  "
              f"val={avg_val:.5f}  lr={scheduler.get_last_lr()[0]:.2e}  "
              f"({elapsed:.1f}s)")

        if avg_val < best_val:
            best_val = avg_val
            model.save_checkpoint(f"checkpoints/mlp_{args.hand}_best.pt")
            print(f"  → saved best checkpoint (val={best_val:.5f})")

    model.save_checkpoint(f"checkpoints/mlp_{args.hand}_last.pt")
    print(f"\nDone. Best val loss: {best_val:.5f}")
    print(f"Best checkpoint: checkpoints/mlp_{args.hand}_best.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hand",           default="allegro", choices=list(HAND_CONFIGS))
    p.add_argument("--data",           nargs="+", required=True,
                   help=".npz file(s) from collect_data.py (supports glob patterns)")
    p.add_argument("--epochs",         type=int,   default=100)
    p.add_argument("--batch-size",     type=int,   default=256)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--hidden",         type=int,   default=256)
    p.add_argument("--lambda-physics", type=float, default=0.1,
                   help="Weight of physics loss. Set 0 to use imitation loss only.")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
