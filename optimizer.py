from abc import abstractmethod
from typing import Dict, List, Optional

import nlopt
import numpy as np
import torch
from robot_adaptor import RobotAdaptor
from utils import utils_calc as ucalc
from utils import utils_torch as utorch


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


def _seg_seg_dist_batch(a0: torch.Tensor, a1: torch.Tensor,
                        b0: torch.Tensor, b1: torch.Tensor,
                        eps: float = 1e-8) -> torch.Tensor:
    """Batched differentiable minimum distance between N pairs of 3D line segments.
    Args:
        a0, a1, b0, b1: shape (N, 3)
    Returns:
        distances shape (N,) — differentiable w.r.t. all inputs
    """
    da = a1 - a0          # (N, 3)
    db = b1 - b0
    dc = b0 - a0
    aa = (da * da).sum(-1)  # (N,)
    bb = (db * db).sum(-1)
    ab = (da * db).sum(-1)
    ac = (da * dc).sum(-1)
    bc = (db * dc).sum(-1)
    denom = aa * bb - ab * ab
    s = torch.clamp((ac * bb - ab * bc) / (denom + eps), 0.0, 1.0)
    t = torch.clamp((ac * ab - aa * bc) / (denom + eps), 0.0, 1.0)
    closest_a = a0 + s.unsqueeze(-1) * da   # (N, 3)
    closest_b = b0 + t.unsqueeze(-1) * db
    return torch.norm(closest_a - closest_b + eps, dim=-1)  # (N,)


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
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())

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
        try:
            x_opt = self.opt.optimize(x_init)
            qpos_doa = x_opt
        except (ValueError, nlopt.RoundoffLimited) as e:
            print(f"[optimizer] {type(e).__name__}: {e}")
            qpos_doa = x_init

        qpos_doa = np.clip(qpos_doa, self.joint_limits[:, 0], self.joint_limits[:, 1])
        return np.array(qpos_doa, dtype=np.float32)

    @abstractmethod
    def get_objective_function(self, ref_values: Dict[str, np.ndarray]):
        pass


class PositionOptimizer(RetargetOptimizer):
    retargeting_type = "POSITION"

    def __init__(self, robot_adaptor: RobotAdaptor, targets: Dict, params: Dict):
        super().__init__(robot_adaptor)

        self.target_links_name = targets["target_links_name"]
        if "ee_link" in self.target_links_name:
            self.hand_type = "shadow"
        else:
            self.hand_type = "leap"

        self.huber_loss = torch.nn.SmoothL1Loss(beta=params["huber_delta"])
        self.opt.set_ftol_abs(params["opt_ftol_abs"])
        self.opt.set_maxtime(params["opt_maxtime"])

    def get_objective_function(self, ref_values: Dict[str, np.ndarray]):
        # extract reference (target) values
        ref_links_pos = ref_values["links_pos"]  # shape (n_link, 3)
        qpos_doa_init = ref_values["qpos_doa_last"]
        # to torch (do not requires grad)
        ref_links_pos_torch = torch.as_tensor(ref_links_pos).requires_grad_(False)
        qpos_doa_init_torch = torch.as_tensor(qpos_doa_init).requires_grad_(False)

        # cost weights
        weight_links_pos = torch.as_tensor(ref_values["weights"]["links_pos"]).requires_grad_(False)
        weight_action = torch.as_tensor(ref_values["weights"]["action"]).requires_grad_(False)

        qpos_doa = np.zeros((self.robot_adaptor.doa))
        qpos_dof = np.zeros((self.robot_model.dof))

        # ---------------------- define the cost and gradient of the optimization ----------------------
        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            # x 为优化器传入的完整关节角（包含机械臂和机械手）
            qpos_doa[:] = x
            # print(f"qpos_doa: {qpos_doa}")

            # 如果存在固定机械臂关节，则覆盖对应部分
            if "qpos_arm_fixed" in ref_values:
                if self.hand_type == "leap":
                    arm_indices = [0, 1, 2, 3, 4, 5, 6]  # 假设 robot_adaptor 中定义了机械臂的关节索引
                else:
                    arm_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                fixed_arm = ref_values["qpos_arm_fixed"]
                qpos_doa[arm_indices] = fixed_arm
                # print(f"qpos_doa after fixed arm: {qpos_doa}")

            # 根据 qpos_doa 计算完整系统的 qpos
            qpos_dof[:] = self.robot_adaptor.forward_qpos(qpos_doa)
            self.robot_model.compute_forward_kinematics(qpos_dof)

            # ---------------------- variables ---------------------
            links_pose_list = [self.robot_model.get_frame_pose(name) for name in self.target_links_name]
            links_pose = np.stack(links_pose_list, axis=0)  # shape (n, 4, 4)
            links_pos = links_pose[:, 0:3, 3]  # shape (n, 3)

            # to torch (requires grad)
            links_pos_torch = torch.as_tensor(links_pos).requires_grad_(True)
            qpos_doa_torch = torch.as_tensor(qpos_doa).requires_grad_(True)

            # ---------------------- costs ----------------------
            # errors
            links_pos_err = torch.norm(links_pos_torch - ref_links_pos_torch, dim=-1)

            # costs with weights
            links_pos_cost = self.huber_loss(weight_links_pos * links_pos_err, torch.zeros_like(links_pos_err)).sum()
            action_cost = (weight_action * (qpos_doa_torch - qpos_doa_init_torch) ** 2).sum()

            # total cost
            total_cost = links_pos_cost + action_cost

            # ---------------------- gradients ----------------------
            if grad.size > 0:
                total_cost.backward()

                # finger gradient
                links_jaco_list = []
                self.robot_model.compute_jacobians(qpos_dof)
                for i, name in enumerate(self.target_links_name):
                    link_jaco = self.robot_model.get_frame_space_jacobian(name)
                    link_linear_jaco = link_jaco[:3, :]
                    links_jaco_list.append(link_linear_jaco)
                links_jaco = self.robot_adaptor.backward_jacobian(
                    np.stack(links_jaco_list, axis=0)
                )  # shape (n_link, 3, n_joint_doa)
                grad_links_pos = links_pos_torch.grad.cpu().numpy()[
                    :, None, :
                ]  # link pos gradient w.r.t. links pos; shape(n_link, 1, 3)
                link_grad = np.matmul(
                    grad_links_pos, links_jaco
                )  # # link pos gradient w.r.t. joint pos; (n_link, 1, 3) * (n_link, 3, n_joint_doa) = (n_link, 1, n_joint_doa)
                link_grad = link_grad.mean(1).sum(0)  # shape (n_joint_doa)

                # action gradient w.r.t. joint pos
                action_grad = qpos_doa_torch.grad.cpu().numpy().reshape(-1)

                # total gradient
                total_grad = link_grad + action_grad

                # 如果存在固定机械臂关节，将这些关节的梯度置零，确保不更新
                if "qpos_arm_fixed" in ref_values:
                    # arm_indices = [0, 1, 2, 3, 4, 5, 6]
                    total_grad[arm_indices] = 0.0

                grad[:] = total_grad

            return total_cost.cpu().detach().item()

        return objective


# class PositionOptimizer(RetargetOptimizer):
#     retargeting_type = "POSITION"

#     def __init__(self, robot_adaptor: RobotAdaptor, targets: Dict, params: Dict):
#         super().__init__(robot_adaptor)

#         self.target_links_name = targets["target_links_name"]

#         self.huber_loss = torch.nn.SmoothL1Loss(beta=params["huber_delta"])
#         self.opt.set_ftol_abs(params["opt_ftol_abs"])
#         self.opt.set_maxtime(params["opt_maxtime"])

#     def get_objective_function(self, ref_values: Dict[str, np.ndarray]):
#         # extract reference (target) values
#         ref_links_pos = ref_values["links_pos"]  # shape (n_link, 3)
#         qpos_doa_init = ref_values["qpos_doa_last"]
#         # to torch (do not requires grad)
#         ref_links_pos_torch = torch.as_tensor(ref_links_pos).requires_grad_(False)
#         qpos_doa_init_torch = torch.as_tensor(qpos_doa_init).requires_grad_(False)

#         # cost weights
#         weight_links_pos = torch.as_tensor(ref_values["weights"]["links_pos"]).requires_grad_(False)
#         weight_action = torch.as_tensor(ref_values["weights"]["action"]).requires_grad_(False)

#         qpos_doa = np.zeros((self.robot_adaptor.doa))
#         qpos_dof = np.zeros((self.robot_model.dof))

#         # ---------------------- define the cost and gradient of the optimization ----------------------
#         def objective(x: np.ndarray, grad: np.ndarray) -> float:
#             qpos_doa[:] = x
#             qpos_dof[:] = self.robot_adaptor.forward_qpos(qpos_doa)
#             self.robot_model.compute_forward_kinematics(qpos_dof)

#             # ---------------------- variables ---------------------
#             links_pose_list = [self.robot_model.get_frame_pose(name) for name in self.target_links_name]
#             links_pose = np.stack(links_pose_list, axis=0)  # shape (n, 4, 4)
#             links_pos = links_pose[:, 0:3, 3]  # shape (n, 3)
#             # print(self.target_links_name)

#             # to torch (requires grad)
#             links_pos_torch = torch.as_tensor(links_pos).requires_grad_(True)
#             qpos_doa_torch = torch.as_tensor(qpos_doa).requires_grad_(True)

#             # ---------------------- costs ----------------------
#             # errors
#             links_pos_err = torch.norm(links_pos_torch - ref_links_pos_torch, dim=-1)

#             # costs with weights
#             links_pos_cost = self.huber_loss(weight_links_pos * links_pos_err, torch.zeros_like(links_pos_err)).sum()
#             action_cost = (weight_action * (qpos_doa_torch - qpos_doa_init_torch) ** 2).sum()

#             # total cost
#             total_cost = links_pos_cost + action_cost

#             # ---------------------- gradients ----------------------
#             if grad.size > 0:
#                 total_cost.backward()

#                 # finger gradient
#                 links_jaco_list = []
#                 self.robot_model.compute_jacobians(qpos_dof)
#                 for i, name in enumerate(self.target_links_name):
#                     link_jaco = self.robot_model.get_frame_space_jacobian(name)
#                     link_linear_jaco = link_jaco[:3, :]
#                     links_jaco_list.append(link_linear_jaco)
#                 links_jaco = self.robot_adaptor.backward_jacobian(
#                     np.stack(links_jaco_list, axis=0)
#                 )  # shape (n_link, 3, n_joint_doa)
#                 grad_links_pos = links_pos_torch.grad.cpu().numpy()[
#                     :, None, :
#                 ]  # link pos gradient w.r.t. links pos; shape(n_link, 1, 3)
#                 link_grad = np.matmul(
#                     grad_links_pos, links_jaco
#                 )  # # link pos gradient w.r.t. joint pos; (n_link, 1, 3) * (n_link, 3, n_joint_doa) = (n_link, 1, n_joint_doa)
#                 link_grad = link_grad.mean(1).sum(0)  # shape (n_joint_doa)

#                 # action gradient w.r.t. joint pos
#                 action_grad = qpos_doa_torch.grad.cpu().numpy().reshape(-1)

#                 # total gradient
#                 grad[:] = link_grad[:] + action_grad[:]

#             return total_cost.cpu().detach().item()

#         return objective


class VectorOptimizer(RetargetOptimizer):
    retargeting_type = "VECTOR"

    def __init__(self, robot_adaptor: RobotAdaptor, targets: Dict, params: Dict):
        super().__init__(robot_adaptor)

        self.origin_links_name = targets["origin_links_name"]
        self.task_links_name = targets["task_links_name"]
        self.computed_links_name = list(set(self.origin_links_name + self.task_links_name))

        if "ee_link" in self.origin_links_name:
            self.hand_type = "shadow"
        else:
            self.hand_type = "leap"

        self.origin_links_idx = torch.tensor([self.computed_links_name.index(name) for name in self.origin_links_name])
        self.task_links_idx = torch.tensor([self.computed_links_name.index(name) for name in self.task_links_name])

        self.huber_loss = torch.nn.SmoothL1Loss(beta=params["huber_delta"])
        self.opt.set_ftol_abs(params["opt_ftol_abs"])
        self.opt.set_maxtime(params["opt_maxtime"])

    def get_objective_function(self, ref_values: Dict[str, np.ndarray]):
        # extract reference (target) values
        ref_links_vec = ref_values["links_vec"]
        qpos_doa_init = ref_values["qpos_doa_last"]
        # to torch (do not requires grad)
        ref_links_vec_torch = torch.as_tensor(ref_links_vec).requires_grad_(False)
        qpos_doa_init_torch = torch.as_tensor(qpos_doa_init).requires_grad_(False)

        # cost weights
        weight_links_vec = torch.as_tensor(ref_values["weights"]["links_vec"]).requires_grad_(False)
        weight_action = torch.as_tensor(ref_values["weights"]["action"]).requires_grad_(False)

        qpos_doa = np.zeros((self.robot_adaptor.doa))
        qpos_dof = np.zeros((self.robot_model.dof))

        # ---------------------- define the cost and gradient of the optimization ----------------------
        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos_doa[:] = x

            # 如果存在固定机械臂关节，则覆盖对应部分
            if "qpos_arm_fixed" in ref_values:
                if self.hand_type == "leap":
                    arm_indices = [0, 1, 2, 3, 4, 5, 6]  # 假设 robot_adaptor 中定义了机械臂的关节索引
                else:
                    arm_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                fixed_arm = ref_values["qpos_arm_fixed"]
                qpos_doa[arm_indices] = fixed_arm
                print(f"qpos_doa after fixed arm: {qpos_doa}")

            qpos_dof[:] = self.robot_adaptor.forward_qpos(qpos_doa)
            self.robot_model.compute_forward_kinematics(qpos_dof)

            # ---------------------- variables ---------------------
            links_pose_list = [self.robot_model.get_frame_pose(name) for name in self.computed_links_name]
            links_pose = np.stack(links_pose_list, axis=0)  # shape (n, 4, 4)
            links_pos = links_pose[:, 0:3, 3]  # shape (n, 3)

            # to torch (requires grad)
            links_pos_torch = torch.as_tensor(links_pos).requires_grad_(True)
            qpos_doa_torch = torch.as_tensor(qpos_doa).requires_grad_(True)

            # ---------------------- costs ----------------------
            # errors
            origin_links_pos = links_pos_torch[self.origin_links_idx, :]
            task_links_pos = links_pos_torch[self.task_links_idx, :]
            links_vec = task_links_pos - origin_links_pos
            links_vec_err = torch.norm(links_vec - ref_links_vec_torch, dim=-1)

            # costs with weights
            links_vec_cost = self.huber_loss(weight_links_vec * links_vec_err, torch.zeros_like(links_vec_err))
            action_cost = (weight_action * (qpos_doa_torch - qpos_doa_init_torch) ** 2).sum()

            # total cost
            total_cost = links_vec_cost + action_cost

            # ---------------------- gradients ----------------------
            if grad.size > 0:
                total_cost.backward()

                # finger gradient
                links_jaco_list = []
                self.robot_model.compute_jacobians(qpos_dof)
                for i, name in enumerate(self.computed_links_name):
                    link_jaco = self.robot_model.get_frame_space_jacobian(name)
                    link_linear_jaco = link_jaco[:3, :]
                    links_jaco_list.append(link_linear_jaco)
                links_jaco = self.robot_adaptor.backward_jacobian(
                    np.stack(links_jaco_list, axis=0)
                )  # shape (n_link, 3, n_joint_doa)
                # link pos gradient w.r.t. links pos; shape(n_link, 1, 3)
                grad_links_pos = links_pos_torch.grad.cpu().numpy()[:, None, :]
                # link pos gradient w.r.t. joint pos; (n_link, 1, 3) * (n_link, 3, n_joint_doa) = (n_link, 1, n_joint_doa)
                link_vec_grad = np.matmul(grad_links_pos, links_jaco)
                link_vec_grad = link_vec_grad.mean(1).sum(0)  # shape (n_joint_doa)

                # action gradient w.r.t. joint pos
                action_grad = qpos_doa_torch.grad.cpu().numpy().reshape(-1)

                # total gradient
                total_grad = link_vec_grad + action_grad

                # If fixed arm joints exist, set their gradients to zero
                if "qpos_arm_fixed" in ref_values:
                    # arm_indices = [0, 1, 2, 3, 4, 5, 6]
                    total_grad[arm_indices] = 0.0

                grad[:] = total_grad

            return total_cost.cpu().detach().item()

        return objective


# class DexPilotOptimizer(RetargetOptimizer):
#     retargeting_type = "DEXPILOT"

#     def __init__(self, robot_adaptor: RobotAdaptor, targets: Dict, params: Dict):
#         super().__init__(robot_adaptor)

#         # self.origin_links_name = targets["origin_links_name"]
#         # self.task_links_name = targets["task_links_name"]
#         self.fingertip_links_name = targets["fingertip_links_name"]  # determines the number of fingers
#         self.wrist_link_name = targets["wrist_link_name"]

#         self.num_fingers = len(self.fingertip_links_name)
#         origin_link_index, task_link_index = self.generate_link_indices(self.num_fingers)

#         links_name = [self.wrist_link_name] + self.fingertip_links_name
#         target_origin_links_name = [links_name[index] for index in origin_link_index]
#         target_task_links_name = [links_name[index] for index in task_link_index]

#         self.origin_links_name = target_origin_links_name
#         self.task_links_name = target_task_links_name

#         self.computed_links_name = list(set(target_origin_links_name).union(set(target_task_links_name)))
#         self.origin_links_indice = torch.tensor(
#             [self.computed_links_name.index(name) for name in target_origin_links_name]
#         )
#         self.task_links_indice = torch.tensor([self.computed_links_name.index(name) for name in target_task_links_name])

#         self.wrist_link_idx = self.computed_links_name.index(self.wrist_link_name)

#         # params for DexPilot
#         self.huber_loss = torch.nn.SmoothL1Loss(beta=params["huber_delta"])
#         self.project_dist = params["project_dist"]
#         self.escape_dist = params["escape_dist"]
#         self.eta1 = params["eta1"]
#         self.eta2 = params["eta2"]
#         self.opt.set_ftol_abs(params["opt_ftol_abs"])
#         self.opt.set_maxtime(params["opt_maxtime"])

#         self.projected, self.s2_project_index_origin, self.s2_project_index_task, self.projected_dist = (
#             self.set_dexpilot_cache(self.num_fingers, self.eta1, self.eta2)
#         )

#     @staticmethod
#     def generate_link_indices(num_fingers):
#         """
#         Example:
#         >>> generate_link_indices(4)
#         ([2, 3, 4, 3, 4, 4, 0, 0, 0, 0], [1, 1, 1, 2, 2, 3, 1, 2, 3, 4])
#         """
#         origin_link_index = []
#         task_link_index = []

#         # Add indices for connections between fingers
#         for i in range(1, num_fingers):
#             for j in range(i + 1, num_fingers + 1):
#                 origin_link_index.append(j)
#                 task_link_index.append(i)

#         # Add indices for connections to the base (0)
#         for i in range(1, num_fingers + 1):
#             origin_link_index.append(0)
#             task_link_index.append(i)

#         return origin_link_index, task_link_index

#     @staticmethod
#     def set_dexpilot_cache(num_fingers, eta1, eta2):
#         """
#         Example:
#         >>> set_dexpilot_cache(4, 0.1, 0.2)
#         (array([False, False, False, False, False, False]),
#         [1, 2, 2],
#         [0, 0, 1],
#         array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
#         """
#         projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)

#         s2_project_index_origin = []
#         s2_project_index_task = []
#         for i in range(0, num_fingers - 2):
#             for j in range(i + 1, num_fingers - 1):
#                 s2_project_index_origin.append(j)
#                 s2_project_index_task.append(i)

#         projected_dist = np.array([eta1] * (num_fingers - 1) + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2))

#         return projected, s2_project_index_origin, s2_project_index_task, projected_dist

#     def get_objective_function(self, ref_values: Dict[str, np.ndarray]):
#         # extract reference (target) values
#         target_vector = ref_values["target_vector"]
#         qpos_doa_init = ref_values["qpos_doa_last"]
#         wrist_link_pos_human = ref_values["wrist_link_pos"]
#         wrist_link_quat_human = ref_values["wrist_quat"]

#         # to torch (do not requires grad)
#         qpos_doa_init_torch = torch.as_tensor(qpos_doa_init).requires_grad_(False)
#         ref_wrist_quat_torch = torch.as_tensor(wrist_link_quat_human).requires_grad_(False)
#         wrist_link_pos_human_torch = torch.as_tensor(wrist_link_pos_human).requires_grad_(False)

#         len_proj = len(self.projected)
#         len_s2 = len(self.s2_project_index_task)
#         len_s1 = len_proj - len_s2

#         # Update projection indicator
#         target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
#         self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
#         self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False
#         self.projected[len_s1:len_proj] = np.logical_and(
#             self.projected[:len_s1][self.s2_project_index_origin], self.projected[:len_s1][self.s2_project_index_task]
#         )
#         self.projected[len_s1:len_proj] = np.logical_and(
#             self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
#         )

#         # Update weight vector
#         normal_weight = np.ones(len_proj, dtype=np.float32) * 1
#         high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
#         weight = np.where(self.projected, high_weight, normal_weight)

#         # We change the weight to 10 instead of 1 here, for vector originate from wrist to fingertips
#         # This ensures better intuitive mapping due wrong pose detection
#         weight = torch.from_numpy(
#             np.concatenate([weight, np.ones(self.num_fingers, dtype=np.float32) * len_proj + self.num_fingers])
#         )

#         # Compute reference distance vector
#         normal_vec = target_vector  # (10, 3)
#         dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (6, 3)
#         projected_vec = dir_vec * self.projected_dist[:, None]  # (6, 3)

#         # Compute final reference vector
#         reference_vec = np.where(self.projected[:, None], projected_vec, normal_vec[:len_proj])  # (6, 3)
#         reference_vec = np.concatenate([reference_vec, normal_vec[len_proj:]], axis=0)  # (10, 3)
#         torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
#         torch_target_vec.requires_grad_(False)

#         weight_action = torch.as_tensor(ref_values["weights"]["action"]).requires_grad_(False)

#         qpos_doa = np.zeros((self.robot_adaptor.doa))
#         qpos_dof = np.zeros((self.robot_model.dof))

#         # ---------------------- define the cost and gradient of the optimization ----------------------
#         def objective(x: np.ndarray, grad: np.ndarray) -> float:
#             qpos_doa[:] = x
#             qpos_dof[:] = self.robot_adaptor.forward_qpos(qpos_doa)
#             self.robot_model.compute_forward_kinematics(qpos_dof)

#             # ---------------------- variables ---------------------
#             links_pose_list = [self.robot_model.get_frame_pose(name) for name in self.computed_links_name]
#             links_pose = np.stack(links_pose_list, axis=0)  # shape (n, 4, 4)
#             links_pos = links_pose[:, 0:3, 3]  # shape (n, 3)

#             # to torch (requires grad)
#             links_pos_torch = torch.as_tensor(links_pos).requires_grad_(True)
#             qpos_doa_torch = torch.as_tensor(qpos_doa).requires_grad_(True)
#             wrist_pose_torch = torch.as_tensor(links_pose_list[self.wrist_link_idx])
#             wrist_quat_torch = utorch.matrix_to_quaternion(wrist_pose_torch[:3, :3])
#             wrist_quat_torch.requires_grad_(True)
#             wrist_pos_torch = wrist_pose_torch[:3, 3]
#             wrist_pos_torch.requires_grad_(True)

#             # ---------------------- costs ----------------------
#             # errors
#             origin_links_pos = links_pos_torch[self.origin_links_indice, :]
#             task_links_pos = links_pos_torch[self.task_links_indice, :]
#             links_vec = task_links_pos - origin_links_pos
#             links_vec_err = torch.norm(links_vec - torch_target_vec, dim=-1)

#             # costs with weights
#             huber_distance = (
#                 self.huber_loss(links_vec_err, torch.zeros_like(links_vec_err)) * weight / (links_vec.shape[0])
#             ).sum()
#             links_vec_cost = huber_distance.sum()
#             action_cost = (weight_action * (qpos_doa_torch - qpos_doa_init_torch) ** 2).sum()

#             wrist_pos_cost = 1 * torch.norm(wrist_pos_torch - wrist_link_pos_human_torch, dim=-1) ** 2
#             wrist_rot_err = utorch.quaternion_angular_error(
#                 ref_wrist_quat_torch.unsqueeze(0), wrist_quat_torch.unsqueeze(0)
#             ).squeeze()
#             wrist_rot_cost = 1 * wrist_rot_err**2

#             # total cost
#             total_cost = links_vec_cost + action_cost + wrist_pos_cost + wrist_rot_cost

#             # ---------------------- gradients ----------------------
#             if grad.size > 0:
#                 total_cost.backward()

#                 # finger gradient
#                 links_jaco_list = []
#                 self.robot_model.compute_jacobians(qpos_dof)
#                 for i, name in enumerate(self.computed_links_name):
#                     link_jaco = self.robot_model.get_frame_space_jacobian(name)
#                     links_jaco_list.append(link_jaco)
#                 links_jaco = self.robot_adaptor.backward_jacobian(
#                     np.stack(links_jaco_list, axis=0)
#                 )  # shape (n_link, 3, n_joint_doa)
#                 # link pos gradient w.r.t. links pos; shape(n_link, 1, 3)
#                 grad_links_pos = links_pos_torch.grad.cpu().numpy()[:, None, :]
#                 # link pos gradient w.r.t. joint pos; (n_link, 1, 3) * (n_link, 3, n_joint_doa) = (n_link, 1, n_joint_doa)
#                 link_vec_grad = np.matmul(grad_links_pos, links_jaco[:, :3, :])
#                 link_vec_grad = link_vec_grad.mean(1).sum(0)  # shape (n_joint_doa)

#                 # action gradient w.r.t. joint pos
#                 action_grad = qpos_doa_torch.grad.cpu().numpy().reshape(-1)

#                 wrist_pos_grad = wrist_pos_torch.grad.cpu().numpy().reshape(-1)
#                 wrist_jaco = self.robot_model.get_frame_space_jacobian(self.wrist_link_name)[
#                     :3, :
#                 ]  # 取线速度部分 (3, dof)
#                 wrist_jaco_doa = self.robot_adaptor.backward_jacobian(wrist_jaco)  # 转换到 DOA 维度 (3, doa)
#                 wrist_pos_grad = wrist_pos_grad @ wrist_jaco_doa  # (3,) @ (3, 23) -> (23,)

#                 wrist_jaco = links_jaco[self.wrist_link_idx]
#                 wrist_rot_grad_quat = wrist_quat_torch.grad.cpu().numpy().reshape(1, -1)
#                 wrist_quat = wrist_quat_torch.detach().numpy()
#                 wrist_rot_grad = (
#                     wrist_rot_grad_quat @ ucalc.mapping_from_space_avel_to_dquat(wrist_quat) @ wrist_jaco[3:, :]
#                 ).reshape(-1)

#                 # total gradient
#                 grad[:] = link_vec_grad[:] + action_grad[:] + wrist_pos_grad[:] + wrist_rot_grad[:]

#             return total_cost.cpu().detach().item()

#         return objective


class DexPilotOptimizer(RetargetOptimizer):
    retargeting_type = "DEXPILOT"

    def __init__(self, robot_adaptor: RobotAdaptor, targets: Dict, params: Dict):
        super().__init__(robot_adaptor)

        # self.origin_links_name = targets["origin_links_name"]
        # self.task_links_name = targets["task_links_name"]
        self.fingertip_links_name = targets["fingertip_links_name"]  # determines the number of fingers
        self.wrist_link_name = targets["wrist_link_name"]

        if "ee_link" in self.wrist_link_name:
            self.hand_type = "shadow"
        else:
            self.hand_type = "leap"

        self.num_fingers = len(self.fingertip_links_name)
        origin_link_index, task_link_index = self.generate_link_indices(self.num_fingers)

        links_name = [self.wrist_link_name] + self.fingertip_links_name
        target_origin_links_name = [links_name[index] for index in origin_link_index]
        target_origin_links_name.append("world")
        target_task_links_name = [links_name[index] for index in task_link_index]
        if self.hand_type == "leap":
            target_task_links_name.append("wrist")
        else:
            target_task_links_name.append("ee_link")

        self.origin_links_name = target_origin_links_name
        self.task_links_name = target_task_links_name

        self.computed_links_name = list(set(target_origin_links_name).union(set(target_task_links_name)))
        self.origin_links_indice = torch.tensor(
            [self.computed_links_name.index(name) for name in target_origin_links_name]
        )
        self.task_links_indice = torch.tensor([self.computed_links_name.index(name) for name in target_task_links_name])

        self.wrist_link_idx = self.computed_links_name.index(self.wrist_link_name)

        # params for DexPilot
        self.huber_loss = torch.nn.SmoothL1Loss(beta=params["huber_delta"])
        self.project_dist = params["project_dist"]
        self.escape_dist = params["escape_dist"]
        self.eta1 = params["eta1"]
        self.eta2 = params["eta2"]
        self.opt.set_ftol_abs(params["opt_ftol_abs"])
        self.opt.set_maxtime(params["opt_maxtime"])

        self.projected, self.s2_project_index_origin, self.s2_project_index_task, self.projected_dist = (
            self.set_dexpilot_cache(self.num_fingers, self.eta1, self.eta2)
        )

    @staticmethod
    def generate_link_indices(num_fingers):
        """
        Example:
        >>> generate_link_indices(4)
        ([2, 3, 4, 3, 4, 4, 0, 0, 0, 0], [1, 1, 1, 2, 2, 3, 1, 2, 3, 4])
        """
        origin_link_index = []
        task_link_index = []

        # Add indices for connections between fingers
        for i in range(1, num_fingers):
            for j in range(i + 1, num_fingers + 1):
                origin_link_index.append(j)
                task_link_index.append(i)

        # Add indices for connections to the base (0)
        for i in range(1, num_fingers + 1):
            origin_link_index.append(0)
            task_link_index.append(i)

        return origin_link_index, task_link_index

    @staticmethod
    def set_dexpilot_cache(num_fingers, eta1, eta2):
        """
        Example:
        >>> set_dexpilot_cache(4, 0.1, 0.2)
        (array([False, False, False, False, False, False]),
        [1, 2, 2],
        [0, 0, 1],
        array([0.1, 0.1, 0.1, 0.2, 0.2, 0.2]))
        """
        projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)

        s2_project_index_origin = []
        s2_project_index_task = []
        for i in range(0, num_fingers - 2):
            for j in range(i + 1, num_fingers - 1):
                s2_project_index_origin.append(j)
                s2_project_index_task.append(i)

        projected_dist = np.array([eta1] * (num_fingers - 1) + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2))

        return projected, s2_project_index_origin, s2_project_index_task, projected_dist

    def get_objective_function(self, ref_values: Dict[str, np.ndarray]):
        # extract reference (target) values
        target_vector = ref_values["target_vector"]
        qpos_doa_init = ref_values["qpos_doa_last"]
        wrist_link_pos_human = ref_values["wrist_link_pos"]
        wrist_link_quat_human = ref_values["wrist_quat"]

        # to torch (do not requires grad)
        qpos_doa_init_torch = torch.as_tensor(qpos_doa_init).requires_grad_(False)
        ref_wrist_quat_torch = torch.as_tensor(wrist_link_quat_human).requires_grad_(False)
        wrist_link_pos_human_torch = torch.as_tensor(wrist_link_pos_human).requires_grad_(False)

        len_proj = len(self.projected)
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2

        # Update projection indicator
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin], self.projected[:len_s1][self.s2_project_index_task]
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.03
        )

        # Update weight vector
        normal_weight = np.ones(len_proj, dtype=np.float32) * 1
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float32)
        weight = np.where(self.projected, high_weight, normal_weight)

        # We change the weight to 10 instead of 1 here, for vector originate from wrist to fingertips
        # This ensures better intuitive mapping due wrong pose detection
        weight = torch.from_numpy(
            np.concatenate([weight, np.ones(self.num_fingers, dtype=np.float32) * len_proj + self.num_fingers])
        )

        # Compute reference distance vector
        normal_vec = target_vector  # (10, 3)
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (6, 3)
        projected_vec = dir_vec * self.projected_dist[:, None]  # (6, 3)

        # Compute final reference vector
        reference_vec = np.where(self.projected[:, None], projected_vec, normal_vec[:len_proj])  # (6, 3)
        reference_vec = np.concatenate([reference_vec, normal_vec[len_proj:]], axis=0)  # (10, 3)
        reference_vec = np.concatenate([reference_vec, wrist_link_pos_human[None, :]], axis=0)  # (11, 3)

        torch_target_vec = torch.as_tensor(reference_vec, dtype=torch.float32)
        torch_target_vec.requires_grad_(False)

        weight_action = torch.as_tensor(ref_values["weights"]["action"]).requires_grad_(False)

        qpos_doa = np.zeros((self.robot_adaptor.doa))
        qpos_dof = np.zeros((self.robot_model.dof))

        # ---------------------- define the cost and gradient of the optimization ----------------------
        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos_doa[:] = x

            # 如果存在固定机械臂关节，则覆盖对应部分
            if "qpos_arm_fixed" in ref_values:
                if self.hand_type == "leap":
                    arm_indices = [0, 1, 2, 3, 4, 5, 6]
                else:
                    arm_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                fixed_arm = ref_values["qpos_arm_fixed"]
                qpos_doa[arm_indices] = fixed_arm

            qpos_dof[:] = self.robot_adaptor.forward_qpos(qpos_doa)
            self.robot_model.compute_forward_kinematics(qpos_dof)

            # ---------------------- variables ---------------------
            links_pose_list = [self.robot_model.get_frame_pose(name) for name in self.computed_links_name]
            links_pose = np.stack(links_pose_list, axis=0)  # shape (n, 4, 4)
            links_pos = links_pose[:, 0:3, 3]  # shape (n, 3)

            # to torch (requires grad)
            links_pos_torch = torch.as_tensor(links_pos).requires_grad_(True)
            qpos_doa_torch = torch.as_tensor(qpos_doa).requires_grad_(True)
            wrist_pose_torch = torch.as_tensor(links_pose_list[self.wrist_link_idx])
            wrist_quat_torch = utorch.matrix_to_quaternion(wrist_pose_torch[:3, :3])
            wrist_quat_torch.requires_grad_(True)
            wrist_pos_torch = wrist_pose_torch[:3, 3]
            wrist_pos_torch.requires_grad_(True)

            # ---------------------- costs ----------------------
            # errors
            origin_links_pos = links_pos_torch[self.origin_links_indice, :]
            task_links_pos = links_pos_torch[self.task_links_indice, :]
            links_vec = task_links_pos - origin_links_pos
            links_vec_err = torch.norm(links_vec - torch_target_vec, dim=-1)

            # costs with weights
            huber_distance = (
                self.huber_loss(links_vec_err, torch.zeros_like(links_vec_err)) * weight / (links_vec.shape[0])
            ).sum()
            links_vec_cost = huber_distance.sum()
            action_cost = (weight_action * (qpos_doa_torch - qpos_doa_init_torch) ** 2).sum()

            # wrist_pos_cost = 0.5 * torch.norm(wrist_pos_torch - wrist_link_pos_human_torch, dim=-1) ** 2
            wrist_rot_err = utorch.quaternion_angular_error(
                ref_wrist_quat_torch.unsqueeze(0), wrist_quat_torch.unsqueeze(0)
            ).squeeze()
            wrist_rot_cost = 0.1 * wrist_rot_err**2

            # total cost
            # total_cost = links_vec_cost + action_cost + wrist_pos_cost + wrist_rot_cost
            total_cost = links_vec_cost + action_cost + wrist_rot_cost

            # ---------------------- gradients ----------------------
            if grad.size > 0:
                total_cost.backward()

                # shadow finger gradient
                links_jaco_list = []
                self.robot_model.compute_jacobians(qpos_dof)
                for i, name in enumerate(self.computed_links_name):
                    link_jaco = self.robot_model.get_frame_space_jacobian(name)
                    links_jaco_list.append(link_jaco)
                links_jaco = self.robot_adaptor.backward_jacobian(
                    np.stack(links_jaco_list, axis=0)
                )  # shape (n_link, 3, n_joint_doa)
                # link pos gradient w.r.t. links pos; shape(n_link, 1, 3)
                grad_links_pos = links_pos_torch.grad.cpu().numpy()[:, None, :]
                # link pos gradient w.r.t. joint pos; (n_link, 1, 3) * (n_link, 3, n_joint_doa) = (n_link, 1, n_joint_doa)
                link_vec_grad = np.matmul(grad_links_pos, links_jaco[:, :3, :])
                link_vec_grad = link_vec_grad.mean(1).sum(0)  # shape (n_joint_doa)

                # action gradient w.r.t. joint pos
                action_grad = qpos_doa_torch.grad.cpu().numpy().reshape(-1)

                # wrist_pos_grad = wrist_pos_torch.grad.cpu().numpy().reshape(-1)
                # wrist_jaco = self.robot_model.get_frame_space_jacobian(self.wrist_link_name)[
                #     :3, :
                # ]  # 取线速度部分 (3, dof)
                # wrist_jaco_doa = self.robot_adaptor.backward_jacobian(wrist_jaco)  # 转换到 DOA 维度 (3, doa)
                # wrist_pos_grad = wrist_pos_grad @ wrist_jaco_doa  # (3,) @ (3, 23) -> (23,)

                wrist_jaco = links_jaco[self.wrist_link_idx]
                wrist_rot_grad_quat = wrist_quat_torch.grad.cpu().numpy().reshape(1, -1)
                wrist_quat = wrist_quat_torch.detach().numpy()
                wrist_rot_grad = (
                    wrist_rot_grad_quat @ ucalc.mapping_from_space_avel_to_dquat(wrist_quat) @ wrist_jaco[3:, :]
                ).reshape(-1)

                # total_grad = link_vec_grad + action_grad + wrist_pos_grad + wrist_rot_grad
                total_grad = link_vec_grad + action_grad + wrist_rot_grad

                # If fixed arm joints exist, set their gradients to zero
                if "qpos_arm_fixed" in ref_values:
                    # arm_indices = [0, 1, 2, 3, 4, 5, 6]
                    total_grad[arm_indices] = 0.0

                grad[:] = total_grad

                # total gradient
                # grad[:] = link_vec_grad[:] + action_grad[:] + wrist_pos_grad[:] + wrist_rot_grad[:]

            return total_cost.cpu().detach().item()

        return objective


class VectorWristOptimizer(RetargetOptimizer):
    retargeting_type = "VECTOR_WRIST"

    def __init__(self, robot_adaptor: RobotAdaptor, targets: Dict, params: Dict):
        super().__init__(robot_adaptor)

        self.origin_links_name = targets["origin_links_name"]
        self.task_links_name = targets["task_links_name"]
        self.wrist_link_name = targets["wrist_link_name"]
        self.computed_links_name = list(set(self.origin_links_name + self.task_links_name + [self.wrist_link_name]))

        self.origin_links_idx = torch.tensor([self.computed_links_name.index(name) for name in self.origin_links_name])
        self.task_links_idx = torch.tensor([self.computed_links_name.index(name) for name in self.task_links_name])
        self.wrist_link_idx = self.computed_links_name.index(self.wrist_link_name)

        self.huber_loss = torch.nn.SmoothL1Loss(beta=params["huber_delta"])
        self.opt.set_ftol_abs(params["opt_ftol_abs"])
        self.opt.set_maxtime(params["opt_maxtime"])

    def get_objective_function(self, ref_values: Dict[str, np.ndarray]):
        # extract reference (target) values
        ref_links_vec = ref_values["links_vec"]
        ref_wrist_quat = ref_values["wrist_quat"]  # (w, x, y, z)
        qpos_doa_init = ref_values["qpos_doa_last"]
        # to torch (do not requires grad)
        ref_links_vec_torch = torch.as_tensor(ref_links_vec).requires_grad_(False)
        ref_wrist_quat_torch = torch.as_tensor(ref_wrist_quat).requires_grad_(False)
        qpos_doa_init_torch = torch.as_tensor(qpos_doa_init).requires_grad_(False)

        # cost weights
        weight_links_vec = torch.as_tensor(ref_values["weights"]["links_vec"])
        weight_wrist_rot = ref_values["weights"]["wrist_rot"]
        weight_joint_vel = torch.as_tensor(ref_values["weights"]["action"]).requires_grad_(False)

        qpos_doa = np.zeros((self.robot_adaptor.doa))
        qpos_dof = np.zeros((self.robot_model.dof))

        # ---------------------- define the cost and gradient of the optimization ----------------------
        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos_doa[:] = x

            qpos_dof[:] = self.robot_adaptor.forward_qpos(qpos_doa)
            self.robot_model.compute_forward_kinematics(qpos_dof)

            # ---------------------- variables ---------------------
            links_pose_list = [self.robot_model.get_frame_pose(name) for name in self.computed_links_name]
            links_pose = np.stack(links_pose_list, axis=0)  # shape (n, 4, 4)
            links_pos = links_pose[:, 0:3, 3]  # shape (n, 3)

            # to torch (requires grad)
            links_pos_torch = torch.as_tensor(links_pos).requires_grad_(True)
            wrist_pose_torch = torch.as_tensor(links_pose_list[self.wrist_link_idx])
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
            qvel_doa_torch = qpos_doa_torch - qpos_doa_init_torch

            # costs with weights
            links_vec_cost = self.huber_loss(weight_links_vec * links_vec_err, torch.zeros_like(links_vec_err))
            wrist_rot_cost = weight_wrist_rot * wrist_rot_err**2
            action_cost = self.huber_loss(weight_joint_vel * qvel_doa_torch, torch.zeros_like(qvel_doa_torch))

            # total cost
            total_cost = links_vec_cost + wrist_rot_cost + action_cost

            # ---------------------- gradients ----------------------
            if grad.size > 0:
                total_cost.backward()

                # finger gradient
                links_jaco_list = []
                self.robot_model.compute_jacobians(qpos_dof)
                for i, name in enumerate(self.computed_links_name):
                    link_jaco = self.robot_model.get_frame_space_jacobian(name)
                    links_jaco_list.append(link_jaco)
                links_jaco = self.robot_adaptor.backward_jacobian(
                    np.stack(links_jaco_list, axis=0)
                )  # shape (n_link, 6, n_joint_doa)
                # link pos gradient w.r.t. links pos; shape(n_link, 1, 3)
                grad_links_pos = links_pos_torch.grad.cpu().numpy()[:, None, :]
                # link pos gradient w.r.t. joint pos; (n_link, 1, 3) * (n_link, 3, n_joint_doa) = (n_link, 1, n_joint_doa)
                link_vec_grad = np.matmul(grad_links_pos, links_jaco[:, :3, :])
                link_vec_grad = link_vec_grad.mean(1).sum(0)  # shape (n_joint_doa)

                wrist_jaco = links_jaco[self.wrist_link_idx]
                wrist_rot_grad_quat = wrist_quat_torch.grad.cpu().numpy().reshape(1, -1)
                wrist_quat = wrist_quat_torch.detach().numpy()
                wrist_rot_grad = (
                    wrist_rot_grad_quat @ ucalc.mapping_from_space_avel_to_dquat(wrist_quat) @ wrist_jaco[3:, :]
                ).reshape(-1)

                # action gradient w.r.t. joint pos
                action_grad = qpos_doa_torch.grad.cpu().numpy().reshape(-1)

                # total gradient
                grad[:] = link_vec_grad[:] + wrist_rot_grad[:] + action_grad[:]

            return total_cost.cpu().detach().item()

        return objective


def _mesh_bounding_capsule_radius(mesh_path: str) -> float:
    """Load a collision mesh and compute the bounding capsule radius via PCA.

    Finds the principal axis of the mesh (longest dimension via PCA),
    then returns the maximum perpendicular distance from any vertex to that axis.
    This gives the tightest cylindrical bound around the mesh's main axis.
    Returns 0.0 on failure (missing file, bad mesh, etc.).
    """
    try:
        import trimesh
        mesh = trimesh.load(mesh_path, force="mesh")
        vertices = np.array(mesh.vertices, dtype=np.float64)
    except Exception:
        return 0.0

    if len(vertices) < 2:
        return 0.0

    # PCA: eigenvector of largest eigenvalue = principal axis (longest dimension)
    centroid = vertices.mean(axis=0)
    d = vertices - centroid
    _, eigenvectors = np.linalg.eigh(d.T @ d)
    axis = eigenvectors[:, -1]  # (3,) unit vector

    # Max perpendicular distance from any vertex to the principal axis line
    d_par = (d @ axis)[:, None] * axis   # (N, 3) parallel component
    d_perp = d - d_par                   # (N, 3) perpendicular component
    return float(np.linalg.norm(d_perp, axis=1).max())


def _capsule_radii_from_urdf(urdf_path: str, capsule_defs: list) -> list:
    """Compute capsule radii from URDF collision geometry.

    Priority per link:
      1. Mesh collision geometry  → PCA-based bounding capsule radius (accurate)
      2. Box collision geometry   → sqrt(d2²+d3²)/2 heuristic (d1=length axis)
      3. No geometry found        → fall back to hardcoded default_r
    Looks up the distal link (link_b) first, then proximal (link_a).
    """
    import os
    import xml.etree.ElementTree as ET

    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    link_radii: dict = {}   # link_name → (radius, source_label)
    for link_el in root.findall("link"):
        name = link_el.get("name")
        max_r, src = 0.0, ""

        # 1. collision geometry (mesh > box)
        for col in link_el.findall("collision"):
            geom = col.find("geometry")
            if geom is None:
                continue
            mesh_el = geom.find("mesh")
            if mesh_el is not None:
                r = _mesh_bounding_capsule_radius(
                    os.path.join(urdf_dir, mesh_el.get("filename")))
                if r > max_r:
                    max_r, src = r, "col_mesh"
                continue
            box_el = geom.find("box")
            if box_el is not None:
                dims = sorted([float(x) for x in box_el.get("size").split()], reverse=True)
                r = np.sqrt(dims[1] ** 2 + dims[2] ** 2) / 2.0
                if r > max_r:
                    max_r, src = r, "box"

        # visual meshes intentionally NOT used as fallback:
        # they represent render geometry (with fillets, bolts, etc.)
        # and their PCA axis doesn't align with the capsule axis without
        # knowing the joint frame directions — gives incorrect radii

        if max_r > 0.0:
            link_radii[name] = (max_r, src)

    radii = []
    for link_a, link_b, default_r in capsule_defs:
        r_a, src_a = link_radii.get(link_a, (None, None))
        r_b, src_b = link_radii.get(link_b, (None, None))
        if r_a is not None and r_b is not None:
            # both links have geometry — take the max (bounding capsule covers both)
            if r_a >= r_b:
                r, label = r_a, f"{src_a}(proximal)"
            else:
                r, label = r_b, f"{src_b}(distal)"
        elif r_b is not None:
            r, label = r_b, f"{src_b}(distal)"
        elif r_a is not None:
            r, label = r_a, f"{src_a}(proximal)"
        else:
            r, label = default_r, "default"
        radii.append(r)
        print(f"  capsule {link_a}->{link_b}: r={r * 1000:.1f}mm ({label})")
    return radii


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

        self.huber_loss = torch.nn.SmoothL1Loss(beta=params["huber_delta"])
        self.opt.set_ftol_abs(params["opt_ftol_abs"])
        self.opt.set_maxtime(params["opt_maxtime"])

        # capsule collision: 2 capsules per finger + 1 palm capsule
        # finger order: 0-1=index, 2-3=middle, 4-5=ring, 6-7=thumb, 8=palm
        # default radii are fallbacks; real radii are computed from URDF collision boxes
        capsule_defs_raw = [
            ("mcp_joint",       "fingertip",       0.012),  # index proximal (0)
            ("fingertip",       "index_tip",       0.010),  # index distal  (1)
            ("mcp_joint_2",     "fingertip_2",     0.012),  # middle proximal (2)
            ("fingertip_2",     "middle_tip",      0.010),  # middle distal  (3)
            ("mcp_joint_3",     "fingertip_3",     0.012),  # ring proximal  (4)
            ("fingertip_3",     "ring_tip",        0.010),  # ring distal    (5)
            ("thumb_pip",       "thumb_fingertip", 0.013),  # thumb proximal (6)
            ("thumb_fingertip", "thumb_tip",       0.011),  # thumb distal   (7)
            ("palm_lower",      "mcp_joint_2",     0.023),  # palm           (8)
        ]
        urdf_path = params.get("urdf_path")
        if urdf_path:
            print("[VectorWristJointOptimizer] computing capsule radii from URDF...")
            radii = _capsule_radii_from_urdf(urdf_path, capsule_defs_raw)
            capsule_defs = [(a, b, r) for (a, b, _), r in zip(capsule_defs_raw, radii)]
        else:
            capsule_defs = capsule_defs_raw
        # add capsule frames not yet in computed_links_name
        extra = ["mcp_joint", "mcp_joint_2", "mcp_joint_3", "thumb_pip"]
        for f in extra:
            if f not in self.computed_links_name:
                self.computed_links_name.append(f)
        # pre-compute indices and radii
        self.capsule_idx = [
            (self.computed_links_name.index(a), self.computed_links_name.index(b), r)
            for a, b, r in capsule_defs
        ]
        # cross-finger pairs (different fingers, capsules 0-7)
        finger_pairs = [(i, j) for i in range(8) for j in range(i + 1, 8) if i // 2 != j // 2]
        # palm (8) vs thumb only — index/middle/ring MCP joints are ON the palm capsule,
        # so their proximal capsules share an endpoint with palm and would be immediately infeasible
        palm_pairs = [(6, 8), (7, 8)]
        self.capsule_collision_pairs = finger_pairs + palm_pairs
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
        # torch version for objective soft penalty (unused, kept for reference)
        self.cap_radii_sum_torch = torch.as_tensor(self.cap_radii_sum)
        self.collision_weight = 0.0   # soft penalty disabled — using hard constraints
        self.min_finger_dist  = 0.0

        # add hard inequality constraints: dist(capsule_i, capsule_j) >= r1+r2
        self._add_capsule_constraints()

    def _add_capsule_constraints(self):
        """Register capsule collision as hard NLopt inequality constraints.
        NLopt form: g(x) <= 0, so g_k = (r1+r2) - dist_k <= 0.
        Gradients are computed analytically in numpy — no PyTorch backward passes.
        """
        n_c = len(self.capsule_collision_pairs)
        tol = np.full(n_c, 1e-3)  # 1mm tolerance

        _qpos_doa = np.zeros(self.robot_adaptor.doa)
        _qpos_dof = np.zeros(self.robot_model.dof)

        def constraint_fn(result: np.ndarray, x: np.ndarray, grad: np.ndarray):
            _qpos_doa[:] = x
            _qpos_dof[:] = self.robot_adaptor.forward_qpos(_qpos_doa)
            self.robot_model.compute_forward_kinematics(_qpos_dof)

            links_pos = np.stack(
                [self.robot_model.get_frame_pose(name)[:3, 3]
                 for name in self.computed_links_name]
            )  # (n_links, 3)

            dists, ga0, ga1, gb0, gb1 = _seg_seg_dist_and_grad_np(
                links_pos[self.cap_ia],
                links_pos[self.cap_ib],
                links_pos[self.cap_ja],
                links_pos[self.cap_jb],
            )  # each (n_c,) or (n_c, 3)

            result[:] = self.cap_radii_sum - dists

            if grad.size > 0:
                self.robot_model.compute_jacobians(_qpos_dof)
                links_jaco = self.robot_adaptor.backward_jacobian(
                    np.stack([self.robot_model.get_frame_space_jacobian(name)
                              for name in self.computed_links_name])
                )[:, :3, :]  # (n_links, 3, n_doa)

                # d(g_k)/d(qpos) = -d(dist_k)/d(qpos)
                # = -(ga0[k] @ J[ia_k] + ga1[k] @ J[ib_k]
                #      + gb0[k] @ J[ja_k] + gb1[k] @ J[jb_k])
                # ga0 shape: (n_c, 3); links_jaco[cap_ia] shape: (n_c, 3, n_doa)
                # (n_c, 1, 3) @ (n_c, 3, n_doa) → (n_c, 1, n_doa) → (n_c, n_doa)
                grad[:] = -(
                    np.matmul(ga0[:, None, :], links_jaco[self.cap_ia]).squeeze(1)
                    + np.matmul(ga1[:, None, :], links_jaco[self.cap_ib]).squeeze(1)
                    + np.matmul(gb0[:, None, :], links_jaco[self.cap_ja]).squeeze(1)
                    + np.matmul(gb1[:, None, :], links_jaco[self.cap_jb]).squeeze(1)
                )

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
            qpos_doa[:] = x

            # 如果存在固定机械臂关节，则覆盖对应部分
            if "qpos_arm_fixed" in ref_values:
                if self.hand_type == "leap":
                    arm_indices = [0, 1, 2, 3, 4, 5, 6]
                else:
                    arm_indices = [0, 1, 2, 3, 4, 5, 6, 7, 8]
                fixed_arm = ref_values["qpos_arm_fixed"]
                qpos_doa[arm_indices] = fixed_arm

            qpos_dof[:] = self.robot_adaptor.forward_qpos(qpos_doa)
            self.robot_model.compute_forward_kinematics(qpos_dof)

            # ---------------------- variables ---------------------
            links_pose_list = [self.robot_model.get_frame_pose(name) for name in self.computed_links_name]
            links_pose = np.stack(links_pose_list, axis=0)  # shape (n, 4, 4)
            links_pos = links_pose[:, 0:3, 3]  # shape (n, 3)

            # to torch (requires grad)
            links_pos_torch = torch.as_tensor(links_pos).requires_grad_(True)
            wrist_pose_torch = torch.as_tensor(links_pose_list[self.wrist_link_idx])
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
            # print(qpos_doa_err.shape, weight_joint_pos.shape)
            joint_pos_cost = self.huber_loss(weight_joint_pos * qpos_doa_err, torch.zeros_like(qpos_doa_err))
            joint_vel_cost = self.huber_loss(weight_joint_vel * qvel_doa_torch, torch.zeros_like(qvel_doa_torch))

            # total cost (collision handled by hard constraints in NLopt)
            total_cost = links_vec_cost + wrist_rot_cost + joint_pos_cost + joint_vel_cost

            # ---------------------- gradients ----------------------
            if grad.size > 0:
                total_cost.backward()

                # finger gradient
                links_jaco_list = []
                self.robot_model.compute_jacobians(qpos_dof)
                for i, name in enumerate(self.computed_links_name):
                    link_jaco = self.robot_model.get_frame_space_jacobian(name)
                    links_jaco_list.append(link_jaco)
                links_jaco = self.robot_adaptor.backward_jacobian(
                    np.stack(links_jaco_list, axis=0)
                )  # shape (n_link, 6, n_joint_doa)
                # link pos gradient w.r.t. links pos; shape(n_link, 1, 3)
                grad_links_pos = links_pos_torch.grad.cpu().numpy()[:, None, :]
                # link pos gradient w.r.t. joint pos; (n_link, 1, 3) * (n_link, 3, n_joint_doa) = (n_link, 1, n_joint_doa)
                link_vec_grad = np.matmul(grad_links_pos, links_jaco[:, :3, :])
                link_vec_grad = link_vec_grad.mean(1).sum(0)  # shape (n_joint_doa)

                wrist_jaco = links_jaco[self.wrist_link_idx]
                wrist_rot_grad_quat = wrist_quat_torch.grad.cpu().numpy().reshape(1, -1)
                wrist_quat = wrist_quat_torch.detach().numpy()
                wrist_rot_grad = (
                    wrist_rot_grad_quat @ ucalc.mapping_from_space_avel_to_dquat(wrist_quat) @ wrist_jaco[3:, :]
                ).reshape(-1)

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
