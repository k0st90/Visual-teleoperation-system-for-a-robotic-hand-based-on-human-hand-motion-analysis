import numpy as np
from scipy.spatial.transform import Rotation as sciR
from sklearn.preprocessing import normalize

from robot_model import RobotPinocchio
from robot_adaptor import RobotAdaptor
from optimizer import VectorWristJointOptimizer
from utils.utils_mano import MANO_FINGERTIP_INDEX
from utils.utils_calc import quatXYZW2WXYZ


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
    def __init__(self, urdf_path: str, hand_scale: float = 1.5):
        """
        Args:
            urdf_path:   path to leap_hand_right.urdf
            hand_scale:  scale human hand keypoints to match robot size (mingrui default: 1.5)
        """
        self.hand_scale = hand_scale

        robot_model = RobotPinocchio(robot_file_path=urdf_path)
        actuated_joints_name = [str(i) for i in range(16)]
        self.robot_adaptor = RobotAdaptor(
            robot_model=robot_model,
            actuated_joints_name=actuated_joints_name,
            touch_joints_name=[],
        )

        targets = {
            "origin_links_name": [pair[0] for pair in TARGET_LINK_PAIRS],
            "task_links_name":   [pair[1] for pair in TARGET_LINK_PAIRS],
            "wrist_link_name":   WRIST_LINK_NAME,
        }
        params = {"huber_delta": 0.02, "opt_ftol_abs": 1e-5, "opt_maxtime": 0.05}

        self.optimizer = VectorWristJointOptimizer(
            robot_adaptor=self.robot_adaptor,
            targets=targets,
            params=params,
        )

        self.qpos_init = np.zeros(16, dtype=np.float64)
        self.qpos_last = np.zeros(16, dtype=np.float64)
        self.ema_alpha = 0.3  # same as mingrui: 1.0 = no smoothing

    def retarget(self, hand_kps_in_wrist: np.ndarray) -> np.ndarray:
        """
        Args:
            hand_kps_in_wrist: (21, 3) MediaPipe keypoints in wrist frame
        Returns:
            qpos: (16,) joint angles for LEAP hand
        """
        # scale human hand to robot hand size
        hand_kps = hand_kps_in_wrist * self.hand_scale

        # hand is fixed at base -> world frame == wrist frame
        wrist_pos = hand_kps[0, :]
        # identity quaternion (w, x, y, z) — no wrist rotation term for fixed hand
        wrist_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        # -------- pinch detection (same as mingrui) --------
        thumb_tip = hand_kps[MANO_FINGERTIP_INDEX[0]]
        thumb_primary_dist = np.linalg.norm(
            hand_kps[MANO_FINGERTIP_INDEX[1:4]] - thumb_tip.reshape(1, 3), axis=1
        )
        pinch_thres_1 = 0.1   # transition threshold
        pinch_thres_2 = 0.01  # contact threshold

        sigmoid_weights_thumb_primary = sigmoid(thumb_primary_dist, c=pinch_thres_1, w=10)
        sigmoid_weights_wrist_fingertips = sigmoid(
            np.concatenate([[np.min(thumb_primary_dist)], thumb_primary_dist]),
            c=pinch_thres_1, w=-10,
        )

        ref_link_vec = np.zeros((11, 3))
        weights_links_vec = np.zeros(11)

        # -------- wrist -> fingertip vectors [0:4] --------
        ref_link_vec[0:4, :] = hand_kps[MANO_FINGERTIP_INDEX[:4]] - wrist_pos
        weights_links_vec[0:4] = 1.0 * sigmoid_weights_wrist_fingertips

        # -------- thumb -> primary vectors (pinch) [4:7] --------
        # rescale distance: [pinch_thres_2, pinch_thres_1] -> [0, pinch_thres_1]
        rel_pos = hand_kps[MANO_FINGERTIP_INDEX[1:4]] - thumb_tip.reshape(1, 3)
        rel_dist = np.linalg.norm(rel_pos, axis=1)
        k = pinch_thres_1 / (pinch_thres_1 - pinch_thres_2)
        rescaled_rel_dist = k * (rel_dist - pinch_thres_2)
        rescaled_rel_dist[rel_dist < pinch_thres_2] = 0
        rescaled_rel_dist[rel_dist > pinch_thres_1] = rel_dist[rel_dist > pinch_thres_1]
        rescaled_rel_pos = normalize(rel_pos) * rescaled_rel_dist.reshape(-1, 1)
        ref_link_vec[4:7, :] = rescaled_rel_pos
        weights_links_vec[4:7] = 10.0 * sigmoid_weights_thumb_primary

        # -------- fingertip orientation vectors [7:11] --------
        mano_idx = np.asarray(MANO_FINGERTIP_INDEX[:4])
        ref_link_vec[7:11, :] = hand_kps[mano_idx] - hand_kps[mano_idx - 1]
        weights_links_vec[7:11] = 10.0

        # -------- build ref_values (same structure as mingrui) --------
        ref_values = {
            "links_vec":     ref_link_vec,
            "wrist_quat":    wrist_quat,
            "qpos_doa":      self.qpos_init.copy(),
            "qpos_doa_last": self.qpos_last.copy(),
            "weights": {
                "links_vec":  weights_links_vec,
                "wrist_rot":  0.0,          # hand fixed, no arm rotation term
                "joint_pos":  WEIGHTS_JOINT_POS.copy(),
                "joint_vel":  np.full(16, 1e-2),
            },
        }

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
