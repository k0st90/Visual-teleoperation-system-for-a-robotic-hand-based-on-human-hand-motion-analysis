import numpy as np

OPERATOR2MANO_RIGHT = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])

MANO_FINGERTIP_INDEX = [4, 8, 12, 16, 20]


def estimate_frame_from_hand_points(keypoint_3d_array: np.ndarray) -> np.ndarray:
    """
    Compute wrist orientation from 21 hand keypoints (MANO format).
    Returns: (3, 3) rotation matrix
    """
    assert keypoint_3d_array.shape == (21, 3)
    points = keypoint_3d_array[[0, 5, 9], :]

    x_vector = points[0] - points[2]

    points = points - np.mean(points, axis=0, keepdims=True)
    u, s, v = np.linalg.svd(points)
    normal = v[2, :]

    x = x_vector - np.sum(x_vector * normal) * normal
    x = x / np.linalg.norm(x)
    z = np.cross(x, normal)

    if np.sum(z * (points[1] - points[2])) < 0:
        normal *= -1
        z *= -1

    return np.stack([x, normal, z], axis=1)
