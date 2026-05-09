import numpy as np
from scipy.spatial.transform import Rotation as sciR
from sklearn.preprocessing import normalize

from robot_model import RobotPinocchio
from robot_adaptor import RobotAdaptor
from optimizer import VectorWristJointOptimizer
from utils.utils_mano import MANO_FINGERTIP_INDEX
from utils.utils_calc import quatXYZW2WXYZ
from config_loader import load_retargeting_config


def sigmoid(x, c=0, w=1):
    return 1 / (1 + np.exp(w * (x - c)))


# LEAP hand frame names in leap_hand_right.urdf
# Mapped from mingrui panda_leap frame names:
#   wrist               -> palm_lower
#   thumb_tip_center    -> thumb_tip
#   finger1_tip_center  -> index_tip
#   finger2_tip_center  -> middle_tip
#   finger3_tip_center  -> ring_tip
#   thumb_tip_center_lower  -> thumb_fingertip
#   finger1_tip_center_lower -> fingertip
#   finger2_tip_center_lower -> fingertip_2
#   finger3_tip_center_lower -> fingertip_3
TARGET_LINK_PAIRS = [
    # wrist -> fingertip vectors
    ["palm_lower", "thumb_tip"],
    ["palm_lower", "index_tip"],
    ["palm_lower", "middle_tip"],
    ["palm_lower", "ring_tip"],
    # thumb -> primary vectors (pinch detection)
    ["thumb_tip", "index_tip"],
    ["thumb_tip", "middle_tip"],
    ["thumb_tip", "ring_tip"],
    # fingertip orientation vectors (penultimate -> tip)
    ["thumb_fingertip", "thumb_tip"],
    ["fingertip",        "index_tip"],
    ["fingertip_2",      "middle_tip"],
    ["fingertip_3",      "ring_tip"],
]
WRIST_LINK_NAME = "palm_lower"

# joint weights: encourage non-zero pose at MCP joints (indices 0,4,8,12)
# matches mingrui logic: [0.5, 0, 0, 0] per finger
WEIGHTS_JOINT_POS = np.array([
    5.0, 0.0, 0.0, 0.0,   # index  joints 0..3  (high: abduction stays near 0)
    5.0, 0.0, 0.0, 0.0,   # middle joints 4..7  (high: abduction stays near 0)
    5.0, 0.0, 0.0, 0.0,   # ring   joints 8..11 (high: abduction stays near 0)
    0.5, 0.0, 0.0, 0.0,   # thumb  joints 12..15 (low: allow thumb spread)
], dtype=np.float64)


class HandRetargeter:
    def __init__(self, urdf_path: str = None, hand_scale: float = 1.5, yml_path: str = None, assets_path: str = None):
        """
        Args:
            urdf_path:   path to leap_hand_right.urdf (used when yml_path is None)
            hand_scale:  scale human hand keypoints to match robot size (mingrui default: 1.5)
            yml_path:    path to retargeting yml config (overrides urdf_path and link pairs)
        """
        # load from yml if provided, otherwise fall back to hardcoded defaults
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
            n_fingers            = 4  # LEAP Hand default
            actuated_joints_name = [str(i) for i in range(16)]
            touch_joints_name    = []
            weights_joint_pos    = WEIGHTS_JOINT_POS.copy()

        self.hand_scale    = hand_scale
        self.pinch_thres_1 = 0.1
        self.pinch_thres_2 = 0.01
        self.wrist_weight  = 1.0
        self.pinch_weight  = 10.0
        self.orient_weight = 10.0
        self.wjpos         = weights_joint_pos  # shape (n_doa,), ABD joints have high weight
        self.opt_maxtime  = 0.05
        self.opt_ftol_abs = 1e-5
        self.huber_delta  = 0.02

        robot_model = RobotPinocchio(robot_file_path=urdf_path)
        self.robot_adaptor = RobotAdaptor(
            robot_model=robot_model,
            actuated_joints_name=actuated_joints_name,
            touch_joints_name=touch_joints_name,
        )

        targets = {
            "origin_links_name": [pair[0] for pair in link_pairs],
            "task_links_name":   [pair[1] for pair in link_pairs],
            "wrist_link_name":   wrist_link,
        }
        params = {"huber_delta": self.huber_delta, "opt_ftol_abs": self.opt_ftol_abs, "opt_maxtime": self.opt_maxtime}

        self.optimizer = VectorWristJointOptimizer(
            robot_adaptor=self.robot_adaptor,
            targets=targets,
            params=params,
        )

        self.n_fingers = n_fingers
        self.actuated_joints_name = actuated_joints_name
        n_doa = self.robot_adaptor.doa

        self.qpos_init = np.zeros(n_doa, dtype=np.float64)
        self.qpos_last = np.zeros(n_doa, dtype=np.float64)
        self.ema_alpha = 0.3  # same as mingrui: 1.0 = no smoothing

    def _build_ref_values(self, hand_kps_in_wrist: np.ndarray) -> dict:
        """
        Build ref_values dict from raw (unscaled) keypoints.
        Used by retarget() and PhysicsLoss in train_mlp.py.
        """
        hand_kps  = hand_kps_in_wrist * self.hand_scale
        wrist_pos = hand_kps[0, :]
        wrist_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        N       = self.n_fingers
        n_total = 3 * N - 1

        thumb_tip = hand_kps[MANO_FINGERTIP_INDEX[0]]
        thumb_primary_dist = np.linalg.norm(
            hand_kps[MANO_FINGERTIP_INDEX[1:N]] - thumb_tip.reshape(1, 3), axis=1
        )
        pinch_thres_1 = self.pinch_thres_1
        pinch_thres_2 = self.pinch_thres_2

        sigmoid_weights_thumb_primary = sigmoid(thumb_primary_dist, c=pinch_thres_1, w=10)
        sigmoid_weights_wrist_fingertips = sigmoid(
            np.concatenate([[np.min(thumb_primary_dist)], thumb_primary_dist]),
            c=pinch_thres_1, w=-10,
        )

        ref_link_vec      = np.zeros((n_total, 3))
        weights_links_vec = np.zeros(n_total)

        # wrist -> fingertip vectors [0:N]
        ref_link_vec[0:N, :] = hand_kps[MANO_FINGERTIP_INDEX[:N]] - wrist_pos
        weights_links_vec[0:N] = self.wrist_weight * sigmoid_weights_wrist_fingertips

        # thumb -> primary vectors (pinch) [N:2N-1]
        rel_pos  = hand_kps[MANO_FINGERTIP_INDEX[1:N]] - thumb_tip.reshape(1, 3)
        rel_dist = np.linalg.norm(rel_pos, axis=1)
        k = pinch_thres_1 / (pinch_thres_1 - pinch_thres_2)
        rescaled_rel_dist = k * (rel_dist - pinch_thres_2)
        rescaled_rel_dist[rel_dist < pinch_thres_2] = 0
        rescaled_rel_dist[rel_dist > pinch_thres_1] = rel_dist[rel_dist > pinch_thres_1]
        rescaled_rel_pos = normalize(rel_pos) * rescaled_rel_dist.reshape(-1, 1)
        ref_link_vec[N:2*N-1, :] = rescaled_rel_pos
        weights_links_vec[N:2*N-1] = self.pinch_weight * sigmoid_weights_thumb_primary

        # fingertip orientation vectors [2N-1:3N-1]
        mano_idx = np.asarray(MANO_FINGERTIP_INDEX[:N])
        ref_link_vec[2*N-1:3*N-1, :] = hand_kps[mano_idx] - hand_kps[mano_idx - 1]
        weights_links_vec[2*N-1:3*N-1] = self.orient_weight

        return {
            "links_vec":     ref_link_vec,
            "wrist_quat":    wrist_quat,
            "qpos_doa":      self.qpos_init.copy(),
            "qpos_doa_last": self.qpos_last.copy(),
            "weights": {
                "links_vec":  weights_links_vec,
                "wrist_rot":  0.0,
                "joint_pos":  self.wjpos.copy(),
                "joint_vel":  np.full(len(self.qpos_init), 1e-2),
            },
            "params": {
                "huber_delta":   self.huber_delta,
                "opt_ftol_abs":  self.opt_ftol_abs,
                "opt_maxtime":   self.opt_maxtime,
            },
        }

    def retarget(self, hand_kps_in_wrist: np.ndarray) -> np.ndarray:
        """
        Args:
            hand_kps_in_wrist: (21, 3) keypoints in wrist frame
        Returns:
            qpos: (n_doa,) joint angles
        """
        ref_values = self._build_ref_values(hand_kps_in_wrist)
        qpos = self.optimizer.retarget(ref_values)

        # EMA smoothing (same as mingrui)
        qpos = self.ema_alpha * qpos + (1 - self.ema_alpha) * self.qpos_last
        self.qpos_last = qpos.copy()

        return qpos


if __name__ == "__main__":
    retargeter = HandRetargeter(urdf_path="assets/leap_hand/leap_hand_right.urdf")

    # test with random keypoints
    fake_kps = np.random.randn(21, 3) * 0.05
    qpos = retargeter.retarget(fake_kps)
    print("qpos:", qpos)
    print("qpos shape:", qpos.shape)
