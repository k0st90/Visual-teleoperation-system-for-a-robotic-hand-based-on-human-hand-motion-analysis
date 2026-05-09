import numpy as np
from scipy.spatial.transform import Rotation as sciR


def quatWXYZ2XYZW(quat_wxyz):
    quat_wxyz = np.asarray(quat_wxyz).reshape(-1, 4)
    return quat_wxyz[:, [1, 2, 3, 0]].reshape(quat_wxyz.shape)


def mappingFromAvelToDquat(quat):
    """dq/dt = M(q) * avel_in_body_frame,  q = [w, x, y, z]"""
    w, x, y, z = quat
    return 0.5 * np.array([[-x, -y, -z],
                            [ w, -z,  y],
                            [ z,  w, -x],
                            [-y,  x,  w]])


def mapping_from_space_avel_to_dquat(quat):
    """dq/dt = M(q) * avel_in_space_frame,  q = [w, x, y, z]"""
    return mappingFromAvelToDquat(quat) @ sciR.from_quat(quatWXYZ2XYZW(quat)).as_matrix().T
