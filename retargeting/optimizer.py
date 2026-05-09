from abc import abstractmethod
from typing import Dict, List, Optional

import nlopt
import numpy as np
import torch
from .robot_adaptor import RobotAdaptor
from utils import calc as ucalc
from utils import torch_utils as utorch


def _seg_seg_dist_and_grad_np(a0, a1, b0, b1, eps=1e-8):
    """Batched numpy: minimum distance + analytical gradients for N segment pairs.
    Args: a0, a1, b0, b1 — shape (N, 3) numpy arrays
    Returns: dists (N,), grad_a0 (N,3), grad_a1 (N,3), grad_b0 (N,3), grad_b1 (N,3)
    Gradients are d(dist)/d(endpoint), ignoring clamping boundary (valid in smooth interior).
    """
    da = a1 - a0
    db = b1 - b0
    dc = b0 - a0
    aa = (da * da).sum(-1)
    bb = (db * db).sum(-1)
    ab = (da * db).sum(-1)
    ac = (da * dc).sum(-1)
    bc = (db * dc).sum(-1)
    denom = aa * bb - ab * ab
    s = np.clip((ac * bb - ab * bc) / (denom + eps), 0.0, 1.0)
    t = np.clip((ac * ab - aa * bc) / (denom + eps), 0.0, 1.0)
    closest_a = a0 + s[:, None] * da
    closest_b = b0 + t[:, None] * db
    diff = closest_a - closest_b
    dist = np.sqrt((diff * diff).sum(-1) + eps)
    unit = diff / dist[:, None]
    return (
        dist,
        unit * (1 - s)[:, None],   # grad_a0
        unit * s[:, None],          # grad_a1
        -unit * (1 - t)[:, None],   # grad_b0
        -unit * t[:, None],         # grad_b1
    )


class RetargetOptimizer:
    retargeting_type = "BASE"

    def __init__(self, robot_adaptor: RobotAdaptor):
        self.robot_adaptor = robot_adaptor
        self.robot_model = robot_adaptor.robot_model

        self.opt_dim = self.robot_adaptor.doa
        self.opt = nlopt.opt(nlopt.LD_SLSQP, self.opt_dim)
        self.joint_limits = self.robot_adaptor.backward_qpos(self.robot_model.joint_limits)
        # manually set lower bound of pip/dip joints
        if self.opt_dim >= 19:  # panda(7) + leap(16) = 23 DOF
            print("Manually set the lower bound of 9, 10, 13, 14, 17, 18 joints as zero.")
            self.joint_limits[[9, 10, 13, 14, 17, 18], 0] = -0.2
        elif self.opt_dim == 16:  # leap only
            print("Manually set the lower bound of 2, 3, 6, 7, 10, 11 joints as zero.")
            self.joint_limits[[2, 3, 6, 7, 10, 11], 0] = -0.2
        self.set_joint_limit(self.joint_limits)

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        """
        Args:
            joint_limits: shape (n_joint_dof, 2)
        """
        if joint_limits.shape != (self.opt_dim, 2):
            raise ValueError(f"Expect joint limits have shape: {(self.opt_dim, 2)}, but get {joint_limits.shape}")
        lb = joint_limits[:, 0] - epsilon
        ub = joint_limits[:, 1] + epsilon
        invalid = np.where(lb > ub)[0]
        if len(invalid) > 0:
            print(f"[set_joint_limit] WARNING: lb > ub at joint indices: {invalid.tolist()}")
            for i in invalid:
                print(f"  joint {i}: lb={lb[i]:.6f}  ub={ub[i]:.6f}")
            raise ValueError(f"Invalid joint limits: lb > ub at indices {invalid.tolist()}")
        self.opt.set_lower_bounds(lb.tolist())
        self.opt.set_upper_bounds(ub.tolist())

    # def retarget(self, ref_values: Dict[str, np.ndarray]) -> np.ndarray:
    #     """
    #     Call the optimization for retargeting.
    #     """
    #     objective_fn = self.get_objective_function(ref_values)
    #     self.opt.set_min_objective(objective_fn)

    #     x_init = ref_values["qpos_doa_last"]
    #     try:
    #         x_opt = self.opt.optimize(x_init)
    #         qpos_doa = x_opt
    #     except ValueError as e:
    #         print(e)
    #         qpos_doa = x_init

    #     qpos_doa = np.clip(qpos_doa, self.joint_limits[:, 0], self.joint_limits[:, 1])
    #     return np.array(qpos_doa, dtype=np.float32)

    def retarget(self, ref_values: Dict[str, np.ndarray], arm_qpos: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Call the optimization for retargeting.
        If arm_qpos is provided, it will be injected into ref_values.
        """
        if arm_qpos is not None:
            # 将传入的机械臂qpos放入ref_values中，供目标函数使用
            ref_values["qpos_arm_fixed"] = arm_qpos

        objective_fn = self.get_objective_function(ref_values)
        self.opt.set_min_objective(objective_fn)

        x_init = ref_values["qpos_doa_last"]
        if len(x_init) != self.opt_dim:
            raise ValueError(f"x_init dim={len(x_init)} != opt_dim={self.opt_dim}")
        # clip x_init to joint limits so SLSQP never receives an out-of-bounds initial point
        x_init = np.clip(x_init, self.joint_limits[:, 0], self.joint_limits[:, 1])
        try:
            x_opt = self.opt.optimize(x_init)
            qpos_doa = x_opt
        except Exception as e:
            print(f"[optimizer] {type(e).__name__}: {e}")
            qpos_doa = x_init

        qpos_doa = np.clip(qpos_doa, self.joint_limits[:, 0], self.joint_limits[:, 1])
        return np.array(qpos_doa, dtype=np.float32)

    @abstractmethod
    def get_objective_function(self, ref_values: Dict[str, np.ndarray]):
        pass


class VectorWristJointOptimizer(RetargetOptimizer):
    retargeting_type = "VECTOR_WRIST_JOINT"

    def __init__(self, robot_adaptor: RobotAdaptor, targets: Dict, params: Dict):
        super().__init__(robot_adaptor)

        self.origin_links_name = targets["origin_links_name"]
        self.task_links_name = targets["task_links_name"]
        self.wrist_link_name = targets["wrist_link_name"]

        if "ee_link" in self.wrist_link_name:
            self.hand_type = "shadow"
        else:
            self.hand_type = "leap"

        self.computed_links_name = list(set(self.origin_links_name + self.task_links_name + [self.wrist_link_name]))
        self.origin_links_idx = torch.tensor([self.computed_links_name.index(name) for name in self.origin_links_name])
        self.task_links_idx = torch.tensor([self.computed_links_name.index(name) for name in self.task_links_name])
        self.wrist_link_idx = self.computed_links_name.index(self.wrist_link_name)

        # validate all link names exist as Pinocchio frames before entering optimization loop
        valid_frames = set(robot_adaptor.robot_model.frame_names)
        missing = [n for n in self.computed_links_name if n not in valid_frames]
        if missing:
            raise ValueError(
                f"[VectorWristJointOptimizer] Frame names not found in Pinocchio model: {missing}\n"
                f"Available frames: {sorted(valid_frames)}"
            )

        self.huber_loss = torch.nn.SmoothL1Loss(beta=params["huber_delta"])
        self.opt.set_ftol_abs(params["opt_ftol_abs"])
        self.opt.set_maxtime(params["opt_maxtime"])

        # capsule collision avoidance — optional, disabled if not provided in targets
        if "capsule_defs" in targets and "capsule_collision_pairs" in targets:
            capsule_defs = [(str(a), str(b), float(r)) for a, b, r in targets["capsule_defs"]]
            # add capsule frames not yet in computed_links_name
            extra = [a for a, b, _ in capsule_defs] + [b for a, b, _ in capsule_defs]
            for f in extra:
                if f not in self.computed_links_name:
                    self.computed_links_name.append(f)
            # pre-compute indices and radii
            self.capsule_idx = [
                (self.computed_links_name.index(a), self.computed_links_name.index(b), r)
                for a, b, r in capsule_defs
            ]
            self.capsule_collision_pairs = [(int(i), int(j)) for i, j in targets["capsule_collision_pairs"]]
            # precompute index arrays and radii (numpy for fast constraint computation)
            self.cap_ia = np.array([self.capsule_idx[ci][0] for ci, _ in self.capsule_collision_pairs])
            self.cap_ib = np.array([self.capsule_idx[ci][1] for ci, _ in self.capsule_collision_pairs])
            self.cap_ja = np.array([self.capsule_idx[cj][0] for _, cj in self.capsule_collision_pairs])
            self.cap_jb = np.array([self.capsule_idx[cj][1] for _, cj in self.capsule_collision_pairs])
            self.cap_radii_sum = np.array(
                [self.capsule_idx[ci][2] + self.capsule_idx[cj][2]
                 for ci, cj in self.capsule_collision_pairs],
                dtype=np.float64,
            )
            self.cap_radii_sum_torch = torch.as_tensor(self.cap_radii_sum)
            self.collision_enabled = True
        else:
            self.capsule_idx             = []
            self.capsule_collision_pairs = []
            self.cap_ia = self.cap_ib = self.cap_ja = self.cap_jb = np.array([], dtype=np.int64)
            self.cap_radii_sum       = np.array([], dtype=np.float64)
            self.cap_radii_sum_torch = torch.as_tensor(self.cap_radii_sum)
            self.collision_enabled   = False
            print("[VectorWristJointOptimizer] capsule collision avoidance disabled")

        self.collision_weight = 0.0   # soft penalty disabled — using hard constraints
        self.min_finger_dist  = 0.0

        # shared FK cache: avoid recomputing FK/Jacobians when x hasn't changed
        # between constraint_fn and objective calls within the same NLopt iteration
        self._fk_cache_x         = None   # cached x (copy)
        self._fk_cache_pos       = None   # cached links_pos   (n_links, 3)
        self._fk_cache_poses     = None   # cached links_poses (n_links, 4, 4)
        self._fk_cache_jaco      = None   # cached links_jaco  (n_links, 6, n_doa)
        self._fk_cache_qpos_dof  = None   # cached qpos_dof    (n_dof,)

        # add hard inequality constraints only if collision avoidance is enabled
        if self.collision_enabled:
            self._add_capsule_constraints()

    def _fk_update(self, x: np.ndarray, need_jaco: bool = False):
        """Compute FK (and optionally Jacobians) only if x changed since last call."""
        try:
            return self._fk_update_impl(x, need_jaco)
        except Exception as e:
            # NLopt silently swallows exceptions from objective/constraint functions.
            # Print here so the user sees the real error before nlopt.invalid_argument.
            print(f"[_fk_update ERROR] {type(e).__name__}: {e}")
            raise

    def _fk_update_impl(self, x: np.ndarray, need_jaco: bool = False):
        if self._fk_cache_x is not None and np.array_equal(x, self._fk_cache_x):
            if not need_jaco or self._fk_cache_jaco is not None:
                return  # cache hit

        qpos_doa = np.zeros(self.robot_adaptor.doa)
        qpos_doa[:] = x
        qpos_dof = self.robot_adaptor.forward_qpos(qpos_doa)
        self.robot_model.compute_forward_kinematics(qpos_dof)

        poses = [self.robot_model.get_frame_pose(name) for name in self.computed_links_name]
        poses_arr = np.stack(poses, axis=0)          # (n_links, 4, 4)
        pos_arr   = poses_arr[:, :3, 3]              # (n_links, 3)

        if need_jaco:
            self.robot_model.compute_jacobians(qpos_dof)
            jaco = self.robot_adaptor.backward_jacobian(
                np.stack([self.robot_model.get_frame_space_jacobian(name)
                          for name in self.computed_links_name])
            )  # (n_links, 6, n_doa)
        else:
            jaco = None

        self._fk_cache_x        = x.copy()
        self._fk_cache_pos      = pos_arr
        self._fk_cache_poses    = poses_arr
        self._fk_cache_qpos_dof = qpos_dof
        if jaco is not None:
            self._fk_cache_jaco = jaco

    def _add_capsule_constraints(self):
        """Register capsule collision as hard NLopt inequality constraints.
        NLopt form: g(x) <= 0, so g_k = (r1+r2) - dist_k <= 0.
        Gradients are computed analytically in numpy — no PyTorch backward passes.
        """
        n_c = len(self.capsule_collision_pairs)
        tol = np.full(n_c, 1e-3)  # 1mm tolerance

        def constraint_fn(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
            try:
                self._fk_update(x, need_jaco=(grad.size > 0))
                links_pos = self._fk_cache_pos

                dists, ga0, ga1, gb0, gb1 = _seg_seg_dist_and_grad_np(
                    links_pos[self.cap_ia],
                    links_pos[self.cap_ib],
                    links_pos[self.cap_ja],
                    links_pos[self.cap_jb],
                )

                result[:] = self.cap_radii_sum - dists

                if grad.size > 0:
                    links_jaco = self._fk_cache_jaco[:, :3, :]  # (n_links, 3, n_doa)
                    grad[:] = -(
                        np.matmul(ga0[:, None, :], links_jaco[self.cap_ia]).squeeze(1)
                        + np.matmul(ga1[:, None, :], links_jaco[self.cap_ib]).squeeze(1)
                        + np.matmul(gb0[:, None, :], links_jaco[self.cap_ja]).squeeze(1)
                        + np.matmul(gb1[:, None, :], links_jaco[self.cap_jb]).squeeze(1)
                    )
            except Exception as e:
                print(f"[constraint_fn ERROR] {type(e).__name__}: {e}")
                raise

        self.opt.add_inequality_mconstraint(constraint_fn, tol)

    def get_objective_function(self, ref_values: Dict[str, np.ndarray]):
        # update dynamic params if provided
        if "params" in ref_values:
            p = ref_values["params"]
            self.opt.set_maxtime(p["opt_maxtime"])
            self.opt.set_ftol_abs(p["opt_ftol_abs"])
            self.huber_loss = torch.nn.SmoothL1Loss(beta=p["huber_delta"])

        # extract reference (target) values
        ref_links_vec = ref_values["links_vec"]
        ref_wrist_quat = ref_values["wrist_quat"]  # (w, x, y, z)
        ref_qpos_doa = ref_values["qpos_doa"]
        qpos_doa_last = ref_values["qpos_doa_last"]
        # to torch (do not requires grad)
        ref_links_vec_torch = torch.as_tensor(ref_links_vec).requires_grad_(False)
        ref_wrist_quat_torch = torch.as_tensor(ref_wrist_quat).requires_grad_(False)
        ref_qpos_doa_torch = torch.as_tensor(ref_qpos_doa).requires_grad_(False)
        qpos_doa_last_torch = torch.as_tensor(qpos_doa_last).requires_grad_(False)

        # cost weights
        weight_links_vec = torch.as_tensor(ref_values["weights"]["links_vec"])
        weight_wrist_rot = ref_values["weights"]["wrist_rot"]
        weight_joint_vel = torch.as_tensor(ref_values["weights"]["joint_vel"]).requires_grad_(False)
        weight_joint_pos = torch.as_tensor(ref_values["weights"]["joint_pos"]).requires_grad_(False)

        qpos_doa = np.zeros((self.robot_adaptor.doa))
        qpos_dof = np.zeros((self.robot_model.dof))

        # ---------------------- define the cost and gradient of the optimization ----------------------
        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            try:
                return _objective_impl(x, grad)
            except Exception as e:
                print(f"[objective ERROR] {type(e).__name__}: {e}")
                raise

        def _objective_impl(x: np.ndarray, grad: np.ndarray) -> float:
            qpos_doa[:] = x

            # 如果存在固定机械臂关节，则覆盖对应部分
            if "qpos_arm_fixed" in ref_values:
                if self.hand_type == "leap":
                    arm_indices = [0, 1, 2, 3, 4, 5, 6]
                else:
                    arm_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                fixed_arm = ref_values["qpos_arm_fixed"]
                qpos_doa[arm_indices] = fixed_arm

            self._fk_update(x, need_jaco=(grad.size > 0))
            qpos_dof[:] = self._fk_cache_qpos_dof

            # ---------------------- variables ---------------------
            links_pose = self._fk_cache_poses   # (n, 4, 4)
            links_pos  = self._fk_cache_pos     # (n, 3)

            # to torch (requires grad)
            links_pos_torch = torch.as_tensor(links_pos).requires_grad_(True)
            wrist_pose_torch = torch.as_tensor(links_pose[self.wrist_link_idx])
            wrist_quat_torch = utorch.matrix_to_quaternion(wrist_pose_torch[:3, :3])
            wrist_quat_torch.requires_grad_(True)
            qpos_doa_torch = torch.as_tensor(qpos_doa).requires_grad_(True)

            # ---------------------- costs ----------------------
            # errors
            origin_links_pos = links_pos_torch[self.origin_links_idx, :]
            task_links_pos = links_pos_torch[self.task_links_idx, :]
            links_vec = task_links_pos - origin_links_pos
            links_vec_err = torch.norm(links_vec - ref_links_vec_torch, dim=-1)
            wrist_rot_err = utorch.quaternion_angular_error(
                ref_wrist_quat_torch.unsqueeze(0), wrist_quat_torch.unsqueeze(0)
            ).squeeze()
            qpos_doa_err = qpos_doa_torch - ref_qpos_doa_torch
            qvel_doa_torch = qpos_doa_torch - qpos_doa_last_torch

            # costs with weights
            links_vec_cost = self.huber_loss(weight_links_vec * links_vec_err, torch.zeros_like(links_vec_err))
            wrist_rot_cost = weight_wrist_rot * wrist_rot_err**2
            joint_pos_cost = self.huber_loss(weight_joint_pos * qpos_doa_err, torch.zeros_like(qpos_doa_err))
            joint_vel_cost = self.huber_loss(weight_joint_vel * qvel_doa_torch, torch.zeros_like(qvel_doa_torch))

            # total cost (collision handled by hard constraints in NLopt)
            total_cost = links_vec_cost + wrist_rot_cost + joint_pos_cost + joint_vel_cost

            # ---------------------- gradients ----------------------
            if grad.size > 0:
                total_cost.backward()

                # finger gradient — Jacobians already in cache from _fk_update
                links_jaco = self._fk_cache_jaco  # (n_link, 6, n_joint_doa)
                # link pos gradient w.r.t. links pos; shape(n_link, 1, 3)
                grad_links_pos = links_pos_torch.grad.cpu().numpy()[:, None, :]
                # link pos gradient w.r.t. joint pos; (n_link, 1, 3) * (n_link, 3, n_joint_doa) = (n_link, 1, n_joint_doa)
                link_vec_grad = np.matmul(grad_links_pos, links_jaco[:, :3, :])
                link_vec_grad = link_vec_grad.mean(1).sum(0)  # shape (n_joint_doa)

                wrist_jaco = links_jaco[self.wrist_link_idx]
                # guard: wrist_quat grad may be None when weight_wrist_rot==0
                if wrist_quat_torch.grad is not None:
                    wrist_rot_grad_quat = wrist_quat_torch.grad.cpu().numpy().reshape(1, -1)
                    wrist_quat = wrist_quat_torch.detach().numpy()
                    wrist_rot_grad = (
                        wrist_rot_grad_quat @ ucalc.mapping_from_space_avel_to_dquat(wrist_quat) @ wrist_jaco[3:, :]
                    ).reshape(-1)
                else:
                    wrist_rot_grad = np.zeros(self.robot_adaptor.doa)

                # gradient w.r.t. joint pos
                grad_qpos_doa = qpos_doa_torch.grad.cpu().numpy().reshape(-1)

                # total gradient
                grad[:] = link_vec_grad[:] + wrist_rot_grad[:] + grad_qpos_doa[:]

                if "qpos_arm_fixed" in ref_values:
                    # arm_indices = [0, 1, 2, 3, 4, 5, 6]
                    grad[arm_indices] = 0.0

            return total_cost.cpu().detach().item()

        return objective


if __name__ == "__main__":
    import time

    from robot_pinocchio import RobotPinocchio
    from utils.utils_mjcf import find_actuated_joints_name, find_touch_joints_name
