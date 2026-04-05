import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as sciR
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
from utils.utils_mano import OPERATOR2MANO_RIGHT, estimate_frame_from_hand_points

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


class _Landmark:
    """Simulates MediaPipe normalized landmark (x, y in [0,1])."""
    def __init__(self, x, y):
        self.x = x
        self.y = y


class WilorDetector:
    """Drop-in replacement for SingleHandDetector using WiLoR-mini."""

    def __init__(self, hand_type="Right"):
        self.hand_type = hand_type
        self.is_right_val = 1.0 if hand_type == "Right" else 0.0
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f"WilorDetector: using {device}")
        self.pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=torch.float16, verbose=False)

    @staticmethod
    def draw_skeleton_on_image(image, keypoint_2d, style="white"):
        if keypoint_2d is None:
            return image
        h, w = image.shape[:2]
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in keypoint_2d]
        color = (255, 48, 48) if style == "white" else (0, 255, 0)
        for a, b in HAND_CONNECTIONS:
            cv2.line(image, pts[a], pts[b], color, 2)
        for pt in pts:
            cv2.circle(image, pt, 4, color, -1)
        return image

    def detect(self, rgb, cam_K=None, input_size=(320, 240)):
        """
        Same interface as SingleHandDetector.detect().
        Args:
            input_size: (width, height) to resize frame before WiLoR inference.
                        Smaller = faster. Default 320x240 (~2x speedup vs 640x480).
                        Set to None to disable resize.
        Returns:
            num_box, hand_kps (21,3), keypoint_2d (list of _Landmark), wrist_pose_in_cam (4x4), wrist_rot (3x3)
        """
        h, w = rgb.shape[:2]
        if input_size is not None and (w, h) != input_size:
            rgb_small = cv2.resize(rgb, input_size, interpolation=cv2.INTER_LINEAR)
            scale_x = w / input_size[0]
            scale_y = h / input_size[1]
        else:
            rgb_small = rgb
            scale_x = scale_y = 1.0
        outputs = self.pipe.predict(rgb_small)

        for hand in outputs:
            if hand.get("is_right") != self.is_right_val:
                continue
            if "wilor_preds" not in hand:
                continue

            preds = hand["wilor_preds"]

            # (21, 3) — MANO joints в camera frame (global_orient вже застосовано)
            kps3d = preds["pred_keypoints_3d"][0].copy()  # (21, 3)
            kps3d -= kps3d[0:1, :]  # center at wrist

            # Застосовуємо ту саму трансформацію що й MediaPipe pipeline:
            # SVD wrist frame → MANO canonical frame (via OPERATOR2MANO_RIGHT)
            wrist_rot_svd = estimate_frame_from_hand_points(kps3d)
            kps3d = kps3d @ wrist_rot_svd @ OPERATOR2MANO_RIGHT

            # (21, 2) in pixel coordinates of rgb_small → scale back to original frame
            kps2d_px = preds["pred_keypoints_2d"][0]  # (21, 2)
            landmarks = [_Landmark(kps2d_px[i, 0] * scale_x / w, kps2d_px[i, 1] * scale_y / h) for i in range(21)]

            # Wrist rotation: SVD-based (consistent з трансформацією keypoints)
            # Це той самий підхід що й MediaPipe fallback
            wrist_rot_matrix = wrist_rot_svd @ OPERATOR2MANO_RIGHT  # (3, 3)

            # Не передаємо wrist_pose_in_cam — main.py використає SVD fallback
            wrist_pose_in_cam = None

            num_box = len(outputs)
            return num_box, kps3d, landmarks, wrist_pose_in_cam, wrist_rot_matrix

        return 0, None, None, None, None
