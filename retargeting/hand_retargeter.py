import numpy as np

from .robot_model import RobotModel
from .retargeting_setup import RetargetingSetup
from .config_loader import load_retargeting_config


WEIGHTS_JOINT_POS = np.array([
    5.0, 0.0, 0.0, 0.0,
    5.0, 0.0, 0.0, 0.0,
    5.0, 0.0, 0.0, 0.0,
    0.5, 0.0, 0.0, 0.0,
], dtype=np.float64)

TARGET_LINK_PAIRS = [
    ["palm_lower", "thumb_tip"],
    ["palm_lower", "index_tip"],
    ["palm_lower", "middle_tip"],
    ["palm_lower", "ring_tip"],
    ["thumb_tip", "index_tip"],
    ["thumb_tip", "middle_tip"],
    ["thumb_tip", "ring_tip"],
    ["thumb_fingertip", "thumb_tip"],
    ["fingertip",        "index_tip"],
    ["fingertip_2",      "middle_tip"],
    ["fingertip_3",      "ring_tip"],
]
WRIST_LINK_NAME = "palm_lower"


class HandRetargeter:
    def __init__(self, urdf_path: str = None, hand_scale: float = 1.5,
                 yml_path: str = None, assets_path: str = None):
        if yml_path is not None:
            cfg = load_retargeting_config(yml_path, assets_path)
            urdf_path            = cfg["urdf_path"]
            link_pairs           = cfg["target_link_pairs"]
            wrist_link           = cfg["wrist_link"]
            hand_scale           = cfg["scaling_factor"]
            n_fingers            = cfg["n_fingers"]
            actuated_joints_name = cfg["actuated_joints_name"] or [str(i) for i in range(16)]
            touch_joints_name    = cfg["touch_joints_name"]
            weights_joint_pos    = np.array(cfg["weights_joint_pos"], dtype=np.float64) \
                                   if cfg["weights_joint_pos"] is not None \
                                   else WEIGHTS_JOINT_POS.copy()
        else:
            link_pairs           = TARGET_LINK_PAIRS
            wrist_link           = WRIST_LINK_NAME
            n_fingers            = 4
            actuated_joints_name = [str(i) for i in range(16)]
            touch_joints_name    = []
            weights_joint_pos    = WEIGHTS_JOINT_POS.copy()

        self.hand_scale    = hand_scale
        self.pinch_thres_1 = 0.1
        self.pinch_thres_2 = 0.01
        self.wrist_weight  = 1.0
        self.pinch_weight  = 10.0
        self.orient_weight = 10.0
        self.wjpos         = weights_joint_pos
        self.huber_delta   = 0.02

        self.robot_model = RobotModel(
            robot_file_path=urdf_path,
            actuated_joints_name=actuated_joints_name,
            touch_joints_name=touch_joints_name,
        )

        targets = {
            "origin_links_name": [pair[0] for pair in link_pairs],
            "task_links_name":   [pair[1] for pair in link_pairs],
            "wrist_link_name":   wrist_link,
        }

        self.setup = RetargetingSetup(
            robot_model=self.robot_model,
            targets=targets,
        )

        self.n_fingers = n_fingers
        self.actuated_joints_name = actuated_joints_name
