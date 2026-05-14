from typing import List
import os
import numpy as np
import pinocchio as pin


class RobotModel:
    def __init__(
        self,
        robot_file_path: str,
        actuated_joints_name: List[str],
        touch_joints_name: List[str] = None,
    ):
        if not os.path.isfile(robot_file_path):
            raise FileNotFoundError(f"URDF not found: {robot_file_path}")

        touch_joints_name = touch_joints_name or []

        self._model = pin.buildModelFromUrdf(robot_file_path)
        self._data  = self._model.createData()

        _dof_names = self._dof_joint_names()
        if len(actuated_joints_name) + len(touch_joints_name) != self.dof:
            raise NotImplementedError("No support for coupled joints.")

        self.actuated_joints_name = actuated_joints_name
        self._actuated_idx = [_dof_names.index(n) for n in actuated_joints_name]

    def _dof_joint_names(self) -> List[str]:
        nqs = self._model.nqs
        return [name for i, name in enumerate(self._model.names) if nqs[i] > 0]

    @property
    def dof(self) -> int:
        return self._model.nq

    @property
    def doa(self) -> int:
        return len(self.actuated_joints_name)

    @property
    def frame_names(self) -> List[str]:
        return [frame.name for frame in self._model.frames]

    @property
    def joint_limits(self) -> np.ndarray:
        lower = self._model.lowerPositionLimit
        upper = self._model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

