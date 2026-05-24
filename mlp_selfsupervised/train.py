"""
Self-supervised MLP retargeter training.

Підхід з оригінальної статті arXiv:2202.10448:
- Лейбли НЕ потрібні (немає пар keypoints→qpos від оптимізатора)
- Loss = геометрична відстань між векторами пальців людини і робота
- FK диференційований через pytorch_kinematics → градієнт проходить
  від loss → link positions → qpos → ваги MLP

Loss (з Mingrui retarget_optimizer.py, VectorWristJointOptimizer):
    total = links_vec_cost    ← головний: вектори пальців збігаються
          + joint_pos_cost    ← регуляризація: ABD суглоби біля нуля

Встановити залежність:
    pip install pytorch-kinematics

Usage:
    python mlp_selfsupervised/train.py --hand allegro --data "data/kps_*.npz"
    python mlp_selfsupervised/train.py --hand allegro --data "data/kps_session1.npz" --epochs 200
"""

import argparse
import glob
import os
import pathlib
import sys
import time
import xml.etree.ElementTree as ET

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from retargeting import HandRetargeter, load_retargeting_config
from mlp_selfsupervised.mlp_model import MLPModel
from utils.mano import MANO_FINGERTIP_INDEX




# ─── differentiable FK via pytorch_kinematics ─────────────────────────────────

def build_fk_chain(urdf_path: str, link_names: list[str], pino_joint_names: list[str], device: str):
    """
    Load URDF and build a differentiable FK chain for the given link names.
    Accepts qpos in pinocchio joint order. URDF joints not in pino_joint_names
    (passive/coupled joints) are set to 0.
    Returns a callable: qpos (B, n_doa) → link_positions (B, n_links, 3)
    """
    try:
        import pytorch_kinematics as pk
    except ImportError:
        print("ERROR: pytorch_kinematics not installed.")
        print("Run: pip install pytorch-kinematics")
        sys.exit(1)

    chain = pk.build_chain_from_urdf(open(urdf_path, "rb").read())
    chain = chain.to(device=device, dtype=torch.float32)

    all_pk_names = [j.name for j in chain.get_joints() if j.joint_type != "fixed"]

    # for each pk joint: index into pino_joint_names, or -1 if not actuated
    pino_idx_for_pk = [
        pino_joint_names.index(name) if name in pino_joint_names else -1
        for name in all_pk_names
    ]

    def fk(qpos_pino: torch.Tensor) -> torch.Tensor:
        """
        Args:
            qpos_pino: (B, n_doa) — joint angles in pinocchio order
        Returns:
            positions: (B, n_links, 3)
        """
        B = qpos_pino.shape[0]
        th = {}
        for pk_i, name in enumerate(all_pk_names):
            pino_i = pino_idx_for_pk[pk_i]
            th[name] = qpos_pino[:, pino_i] if pino_i >= 0 else torch.zeros(B, device=qpos_pino.device, dtype=torch.float32)
        ret = chain.forward_kinematics(th)
        positions = torch.stack(
            [ret[name].get_matrix()[:, :3, 3] for name in link_names], dim=1
        )  # (B, n_links, 3)
        return positions

    return fk


# ─── dataset ──────────────────────────────────────────────────────────────────

class KeypointsDataset(Dataset):
    """
    Loads .npz files with keypoints (keypoints only, no qpos labels).
    """
    def __init__(self, paths: list[str], kps_mean=None, kps_std=None):
        kps_list = []
        for p in paths:
            d = np.load(p)
            kps_list.append(d["kps"])
            print(f"  loaded {d['kps'].shape[0]} samples from {p}")

        self.kps = np.concatenate(kps_list, axis=0).astype(np.float32)  # (N, 21, 3)

        flat = self.kps.reshape(len(self.kps), -1)
        if kps_mean is None:
            self.kps_mean = flat.mean(axis=0)
            self.kps_std  = flat.std(axis=0) + 1e-8
        else:
            self.kps_mean = kps_mean
            self.kps_std  = kps_std

        self.kps_flat_norm = (flat - self.kps_mean) / self.kps_std
        print(f"Total samples: {len(self.kps)}")

    def __len__(self):
        return len(self.kps)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.kps_flat_norm[idx]),  # (63,)  normalised input
            torch.from_numpy(self.kps[idx]),             # (21,3) raw — for loss
        )


# ─── self-supervised loss (from Mingrui VectorWristJointOptimizer) ────────────

class SelfSupervisedLoss(nn.Module):
    """
    links_vec_cost + joint_pos_cost

    links_vec_cost: Huber loss між векторами пальців робота і людини
        robot_vec  = FK(qpos)[task_link] - FK(qpos)[origin_link]
        human_vec  = hand_kps[fingertip]  - hand_kps[wrist]   (+ pinch + orient)

    joint_pos_cost: Huber loss що тримає ABD суглоби біля нуля
        = huber(weights_joint_pos * qpos, 0)
    """

    def __init__(self, retargeter: HandRetargeter, fk_fn, device: str):
        super().__init__()
        self.retargeter = retargeter
        self.fk_fn      = fk_fn
        self.device     = device
        self.huber      = nn.SmoothL1Loss(beta=retargeter.huber_delta, reduction="none")

        setup = retargeter.setup
        # indices into FK output (link_names list) for origin and task links
        self.origin_idx = setup.origin_links_idx.to(device)  # (n_pairs,)
        self.task_idx   = setup.task_links_idx.to(device)    # (n_pairs,)

        # joint pos regularisation weights (ABD joints)
        self.wjpos = torch.tensor(
            retargeter.wjpos, dtype=torch.float32, device=device
        )  # (n_doa,)

    def forward(self, kps_raw_batch: torch.Tensor, qpos_pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            kps_raw_batch: (B, 21, 3) — raw (unscaled) keypoints
            qpos_pred:     (B, n_doa) — MLP output (differentiable)
        Returns:
            scalar loss
        """
        B = kps_raw_batch.shape[0]

        # ── differentiable FK ──────────────────────────────────────────────
        link_positions = self.fk_fn(qpos_pred)   # (B, n_links, 3)

        robot_origin = link_positions[:, self.origin_idx, :]  # (B, n_pairs, 3)
        robot_task   = link_positions[:, self.task_idx,   :]  # (B, n_pairs, 3)
        robot_vecs   = robot_task - robot_origin               # (B, n_pairs, 3)

        # ── ref vectors from human keypoints (same logic as _build_ref_values) ──
        ref_vecs, weights = self._build_ref_vecs_batch(kps_raw_batch)
        # ref_vecs: (B, n_pairs, 3),  weights: (B, n_pairs)

        # ── links_vec_cost ─────────────────────────────────────────────────
        vec_err = torch.norm(robot_vecs - ref_vecs, dim=-1)   # (B, n_pairs)
        links_vec_cost = (
            self.huber(weights * vec_err, torch.zeros_like(vec_err))
        ).mean()

        # ── joint_pos_cost (regularisation) ───────────────────────────────
        joint_pos_cost = (
            self.huber(self.wjpos * qpos_pred, torch.zeros_like(qpos_pred))
        ).mean()

        return links_vec_cost + joint_pos_cost

    def forward_with_components(self, kps_raw_batch, qpos_pred):
        """Same as forward but returns (total, links_vec, joint_pos)."""
        B = kps_raw_batch.shape[0]
        link_positions = self.fk_fn(qpos_pred)
        robot_origin = link_positions[:, self.origin_idx, :]
        robot_task   = link_positions[:, self.task_idx,   :]
        robot_vecs   = robot_task - robot_origin
        ref_vecs, weights = self._build_ref_vecs_batch(kps_raw_batch)
        vec_err = torch.norm(robot_vecs - ref_vecs, dim=-1)
        links_vec_cost = self.huber(weights * vec_err, torch.zeros_like(vec_err)).mean()
        joint_pos_cost = self.huber(self.wjpos * qpos_pred, torch.zeros_like(qpos_pred)).mean()
        return links_vec_cost + joint_pos_cost, links_vec_cost, joint_pos_cost

    def _build_ref_vecs_batch(self, kps_raw: torch.Tensor):
        """
        Vectorised version of HandRetargeter._build_ref_values() — only the
        ref_link_vec and weights_links_vec parts, as PyTorch tensors.

        Args:
            kps_raw: (B, 21, 3) unscaled keypoints
        Returns:
            ref_vecs: (B, n_pairs, 3)
            weights:  (B, n_pairs)
        """
        r = self.retargeter
        hand_kps  = kps_raw * r.hand_scale   # (B, 21, 3)
        wrist_pos = hand_kps[:, 0:1, :]      # (B, 1, 3)

        N       = r.n_fingers
        n_total = 3 * N - 1

        fi   = MANO_FINGERTIP_INDEX       # [0,4,8,12,16] — all 5 MANO fingertips
        tips = hand_kps[:, fi[:N], :]     # (B, N, 3)
        thumb_tip = tips[:, 0:1, :]       # (B, 1, 3)

        # distances thumb→other fingertips
        thumb_primary_dist = torch.norm(
            tips[:, 1:N, :] - thumb_tip, dim=-1
        )  # (B, N-1)

        def sigmoid_t(x, c, w):
            return 1.0 / (1.0 + torch.exp(w * (x - c)))

        c1 = r.pinch_thres_1
        c2 = r.pinch_thres_2

        sw_thumb = sigmoid_t(thumb_primary_dist, c=c1, w=10)   # (B, N-1)
        min_dist = thumb_primary_dist.min(dim=1, keepdim=True).values  # (B,1)
        sw_wrist = sigmoid_t(
            torch.cat([min_dist, thumb_primary_dist], dim=1), c=c1, w=-10
        )  # (B, N)

        ref_vecs = torch.zeros(kps_raw.shape[0], n_total, 3,
                               device=self.device, dtype=torch.float32)
        weights  = torch.zeros(kps_raw.shape[0], n_total,
                               device=self.device, dtype=torch.float32)

        # [0:N] wrist → fingertip
        ref_vecs[:, :N, :]  = tips - wrist_pos
        weights[:, :N]      = r.wrist_weight * sw_wrist

        # [N:2N-1] thumb → other fingertips (rescaled for pinch)
        rel_pos  = tips[:, 1:N, :] - thumb_tip             # (B, N-1, 3)
        rel_dist = torch.norm(rel_pos, dim=-1, keepdim=True)  # (B, N-1, 1)
        k = c1 / (c1 - c2)
        rrd = k * (rel_dist - c2)
        rrd = torch.clamp(rrd, min=0.0)
        # for dist > c1 use original dist
        mask_far = (rel_dist > c1).squeeze(-1)
        rrd_sq = rrd.squeeze(-1)
        rrd_sq[mask_far] = rel_dist.squeeze(-1)[mask_far]
        # normalise direction
        direction = rel_pos / (rel_dist + 1e-8)
        ref_vecs[:, N:2*N-1, :] = direction * rrd_sq.unsqueeze(-1)
        weights[:, N:2*N-1]     = r.pinch_weight * sw_thumb

        # [2N-1:3N-1] penultimate → tip (orientation)
        fi_arr = torch.tensor(fi[:N], device=self.device)
        ref_vecs[:, 2*N-1:3*N-1, :] = (
            hand_kps[:, fi_arr, :] - hand_kps[:, fi_arr - 1, :]
        )
        weights[:, 2*N-1:3*N-1] = r.orient_weight

        return ref_vecs, weights


# ─── training loop ────────────────────────────────────────────────────────────

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    yml_path = args.config
    hand_name = pathlib.Path(args.config).stem
    run_id    = args.run_id or time.strftime("%Y%m%d_%H%M%S")
    print(f"Loading retargeter: {yml_path}")
    retargeter = HandRetargeter(yml_path=yml_path, assets_path=args.assets_path)
    n_doa      = retargeter.robot_model.doa
    joint_lb   = retargeter.setup.joint_limits[:, 0]
    joint_ub   = retargeter.setup.joint_limits[:, 1]
    urdf_path  = load_retargeting_config(yml_path, args.assets_path)["urdf_path"]

    # all link names used in the loss function (origin + task + wrist)
    link_names = retargeter.setup.computed_links_name

    # pinocchio frame names can differ from URDF link names (e.g. fixed-joint frames
    # are named after the joint, not the child link). Build remap: joint_name → child_link.
    root = ET.parse(urdf_path).getroot()
    urdf_link_names = {el.get("name") for el in root.iter("link")}
    link_name_remap = {
        j.get("name"): j.find("child").get("link")
        for j in root.findall("joint")
        if j.find("child") is not None
        and j.get("name") not in urdf_link_names
        and j.find("child").get("link") in urdf_link_names
    }
    if link_name_remap:
        print(f"Link name remap: {link_name_remap}")
    link_names_urdf = [link_name_remap.get(n, n) for n in link_names]
    print(f"n_doa={n_doa}  n_links={len(link_names_urdf)}")

    # build differentiable FK
    pino_names = retargeter.actuated_joints_name
    print("Building differentiable FK chain...")
    fk_fn = build_fk_chain(urdf_path, link_names_urdf, pino_names, device)

    # dataset
    data_paths = []
    for pattern in args.data:
        data_paths.extend(sorted(glob.glob(pattern)))
    if not data_paths:
        print(f"ERROR: no .npz files found: {args.data}")
        sys.exit(1)

    dataset = KeypointsDataset(data_paths)
    n_val   = max(1, int(len(dataset) * 0.1))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(dataset, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(42))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=(device == "cuda"))
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0)
    print(f"Train: {n_train}  Val: {n_val}")

    # model
    model = MLPModel(n_doa=n_doa, joint_lb=joint_lb, joint_ub=joint_ub,
                          hidden=args.hidden).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn   = SelfSupervisedLoss(retargeter, fk_fn, device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    os.makedirs("checkpoints", exist_ok=True)
    best_val = float("inf")

    def _save(m, path, ds):
        """Save checkpoint with normalisation stats."""
        torch.save({
            "n_doa":       m.n_doa,
            "joint_lb":    m.joint_lb.cpu().numpy(),
            "joint_ub":    m.joint_ub.cpu().numpy(),
            "hidden":      m.net[0].out_features,
            "state_dict":  m.state_dict(),
            "kps_mean":    ds.kps_mean,
            "kps_std":     ds.kps_std,
        }, path)

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0

        for kps_norm, kps_raw in train_loader:
            kps_norm = kps_norm.to(device)
            kps_raw  = kps_raw.to(device)

            qpos_pred = model(kps_norm)
            loss = loss_fn(kps_raw, qpos_pred)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # validation
        model.eval()
        val_loss = val_links = val_jpos = 0.0
        with torch.no_grad():
            for kps_norm, kps_raw in val_loader:
                kps_norm = kps_norm.to(device)
                kps_raw  = kps_raw.to(device)
                qpos_pred = model(kps_norm)
                total, lv, jp = loss_fn.forward_with_components(kps_raw, qpos_pred)
                val_loss  += total.item()
                val_links += lv.item()
                val_jpos  += jp.item()

        avg_train  = train_loss / len(train_loader)
        avg_val    = val_loss   / len(val_loader)
        avg_links  = val_links  / len(val_loader)
        avg_jpos   = val_jpos   / len(val_loader)
        elapsed    = time.time() - t0
        is_best    = avg_val < best_val

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"train={avg_train:.5f}  val={avg_val:.5f}  "
              f"links={avg_links:.5f}  jpos={avg_jpos:.5f}  "
              f"lr={scheduler.get_last_lr()[0]:.2e}  "
              f"time={elapsed:.1f}  best={'yes' if is_best else 'no'}")

        if is_best:
            best_val = avg_val
            _save(model, f"checkpoints/mlp_ss_{hand_name}_{run_id}_best.pt", dataset)
            print(f"  -> saved best (val={best_val:.5f})")

    _save(model, f"checkpoints/mlp_ss_{hand_name}_{run_id}_last.pt", dataset)
    print(f"\nDone. Best val loss: {best_val:.5f}")
    print(f"Best checkpoint: checkpoints/mlp_ss_{hand_name}_{run_id}_best.pt")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/leap_hand_right.yml",
                   help="Path to retargeting config YAML")
    p.add_argument("--data",       nargs="+", required=True)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch-size", type=int,   default=256)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--hidden",     type=int,   default=256)
    p.add_argument("--run-id",      type=str,  default=None,
                   help="Unique suffix for checkpoint filenames (default: timestamp)")
    p.add_argument("--assets-path", type=str,  default=None,
                   help="Path to hand assets folder")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
