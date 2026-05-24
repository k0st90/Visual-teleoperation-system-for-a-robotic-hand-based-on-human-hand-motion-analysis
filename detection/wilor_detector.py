import numpy as np
import torch
import cv2
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline
from utils.mano import OPERATOR2MANO_RIGHT, estimate_frame_from_hand_points

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


class WilorDetector:
    """Drop-in replacement for SingleHandDetector using WiLoR-mini."""

    def __init__(self, is_right_val: float = 1.0, device: str = None):
        self.is_right_val = is_right_val
        if device is None:
            _device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        else:
            _device = torch.device(device)
        dtype = torch.float16 if _device.type == "cuda" else torch.float32
        print(f"WilorDetector: using {_device} ({dtype})")
        self.pipe = WiLorHandPose3dEstimationPipeline(device=_device, dtype=dtype, verbose=False)

    @staticmethod
    def draw_skeleton_on_image(image: np.ndarray, keypoint_2d: np.ndarray, style="white") -> np.ndarray:
        if keypoint_2d is None:
            return image
        h, w = image.shape[:2]
        pts = [(int(xy[0] * w), int(xy[1] * h)) for xy in keypoint_2d]
        color = (255, 48, 48) if style == "white" else (0, 255, 0)
        for a, b in HAND_CONNECTIONS:
            cv2.line(image, pts[a], pts[b], color, 2)
        for pt in pts:
            cv2.circle(image, pt, 4, color, -1)
        return image

    def detect(self, rgb: np.ndarray, input_size=(320, 240)) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Args:
            input_size: (width, height) to resize frame before WiLoR inference.
        Returns:
            num_box, hand_kps (21,3), keypoint_2d (21,2) normalized [0,1],
            wrist_pose_in_cam (4x4) or None, wrist_rot (3,3)
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

            kps3d = preds["pred_keypoints_3d"][0].copy()  # (21, 3)
            kps3d -= kps3d[0:1, :]

            wrist_rot_svd = estimate_frame_from_hand_points(kps3d)
            kps3d = kps3d @ wrist_rot_svd @ OPERATOR2MANO_RIGHT

            kps2d_px = preds["pred_keypoints_2d"][0]  # (21, 2)
            keypoint_2d = np.column_stack([
                kps2d_px[:, 0] * scale_x / w,
                kps2d_px[:, 1] * scale_y / h,
            ])  # (21, 2) normalized

            wrist_rot_matrix = wrist_rot_svd @ OPERATOR2MANO_RIGHT
            wrist_pose_in_cam = None

            return len(outputs), kps3d, keypoint_2d, wrist_pose_in_cam, wrist_rot_matrix

        return 0, None, None, None, None
