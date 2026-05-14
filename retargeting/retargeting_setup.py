"""
RetargetingSetup — precomputed geometry metadata for a robot hand.

Converts link pair names (from YAML config) into Pinocchio frame indices
used by SelfSupervisedLoss during MLP training.
"""

import numpy as np
import torch

from .robot_model import RobotModel


class RetargetingSetup:
    def __init__(self, robot_model: RobotModel, targets: dict):
        origin_links = targets["origin_links_name"]
        task_links   = targets["task_links_name"]
        wrist_link   = targets["wrist_link_name"]

        self.computed_links_name = list(
            set(origin_links + task_links + [wrist_link])
        )

        valid_frames = set(robot_model.frame_names)
        missing = [n for n in self.computed_links_name if n not in valid_frames]
        if missing:
            raise ValueError(
                f"Frame names not found in Pinocchio model: {missing}\n"
                f"Available frames: {sorted(valid_frames)}"
            )

        self.origin_links_idx = torch.tensor(
            [self.computed_links_name.index(n) for n in origin_links]
        )
        self.task_links_idx = torch.tensor(
            [self.computed_links_name.index(n) for n in task_links]
        )
        self.wrist_link_idx = self.computed_links_name.index(wrist_link)
        self.joint_limits   = robot_model.joint_limits[robot_model._actuated_idx]
