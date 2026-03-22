import os
import time
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import RunningMode

import numpy as np
import cv2
from scipy.spatial.transform import Rotation as sciR

from utils.utils_calc import batchPosRotVec2Isometry3d
from utils.utils_mano import (
    OPERATOR2MANO_RIGHT,
    OPERATOR2MANO_LEFT,
    estimate_frame_from_hand_points,
)

TASK_PATH = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]


class SingleHandDetector:
    def __init__(
        self,
        hand_type="Right",
        min_detection_confidence=0.8,
        min_tracking_confidence=0.8,
        selfie=False,
    ):
        self.hand_type = hand_type
        base_options = python.BaseOptions(model_asset_path=TASK_PATH)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)
        self.selfie = selfie
        self.operator2mano = (
            OPERATOR2MANO_RIGHT if hand_type == "Right" else OPERATOR2MANO_LEFT
        )
        # Tasks API returns physical handedness directly
        self.detected_hand_type = hand_type
        self._start_time = time.time()

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

    def detect(self, rgb, cam_K=None):
        timestamp_ms = int((time.time() - self._start_time) * 1000)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results = self.hand_detector.detect_for_video(mp_image, timestamp_ms)

        if not results.hand_landmarks:
            return 0, None, None, None, None

        desired_hand_num = -1
        for i, handedness in enumerate(results.handedness):
            label = handedness[0].display_name
            if label == self.detected_hand_type:
                desired_hand_num = i
                break
        if desired_hand_num < 0:
            return 0, None, None, None, None

        keypoint_3d = results.hand_world_landmarks[desired_hand_num]
        keypoint_2d = results.hand_landmarks[desired_hand_num]
        num_box = len(results.hand_landmarks)

        keypoint_3d_array = self.parse_keypoint_3d(keypoint_3d)
        keypoint_3d_array = keypoint_3d_array - keypoint_3d_array[0:1, :]
        mediapipe_wrist_rot = estimate_frame_from_hand_points(keypoint_3d_array)
        joint_pos = keypoint_3d_array @ mediapipe_wrist_rot @ self.operator2mano

        if cam_K is not None:
            wrist_pose_in_cam = self.estimate_wrist_frame_in_cam(
                points_3d_in_wrist=joint_pos,
                points_2d_in_img=self.parse_keypoint_2d(keypoint_2d, rgb.shape),
                cam_K=cam_K,
                rvec_init=sciR.from_matrix(
                    mediapipe_wrist_rot @ self.operator2mano
                ).as_rotvec(),
            )
        else:
            wrist_pose_in_cam = None

        # wrist rotation matrix (3x3) in camera frame — always available
        wrist_rot_matrix = mediapipe_wrist_rot @ self.operator2mano

        return num_box, joint_pos, keypoint_2d, wrist_pose_in_cam, wrist_rot_matrix

    @staticmethod
    def parse_keypoint_3d(keypoint_3d) -> np.ndarray:
        keypoint = np.empty([21, 3])
        for i in range(21):
            keypoint[i][0] = keypoint_3d[i].x
            keypoint[i][1] = keypoint_3d[i].y
            keypoint[i][2] = keypoint_3d[i].z
        return keypoint

    @staticmethod
    def parse_keypoint_2d(keypoint_2d, img_size) -> np.ndarray:
        keypoint = np.empty([21, 2])
        for i in range(21):
            keypoint[i][0] = keypoint_2d[i].x
            keypoint[i][1] = keypoint_2d[i].y
        keypoint = keypoint * np.array([img_size[1], img_size[0]])[None, :]
        return keypoint

    @staticmethod
    def estimate_wrist_frame_in_cam(
        points_3d_in_wrist, points_2d_in_img, cam_K, rvec_init, tvec_init=None
    ):
        if tvec_init is None:
            wrist_point_depth = 0.3
            wrist_point_3d = (
                wrist_point_depth
                * np.linalg.pinv(cam_K)
                @ np.array(
                    [points_2d_in_img[0][0], points_2d_in_img[0][1], 1.0]
                ).reshape(-1, 1)
            )
            tvec_init = wrist_point_3d

        points_2d_in_img = points_2d_in_img.reshape(-1, 1, 2)

        success, rvec, tvec = cv2.solvePnP(
            points_3d_in_wrist,
            points_2d_in_img,
            cam_K,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            useExtrinsicGuess=True,
            rvec=rvec_init,
            tvec=tvec_init,
        )

        wrist_pose_in_cam = batchPosRotVec2Isometry3d(tvec, rvec).reshape(4, 4)
        return wrist_pose_in_cam
