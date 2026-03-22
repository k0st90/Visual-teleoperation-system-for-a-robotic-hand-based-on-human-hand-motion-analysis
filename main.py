import time
import cv2
import numpy as np
import pybullet as pb
import pybullet_data
from scipy.spatial.transform import Rotation as sciR

from hand_detector import SingleHandDetector
from hand_retargeter import HandRetargeter

URDF_PATH = "assets/leap_hand/leap_hand_right.urdf"
HAND_BASE_POS = [0, 0, 0.1]
HAND_BASE_ORI = pb.getQuaternionFromEuler([0, 0, 0])


def setup_pybullet():
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.81)
    pb.loadURDF("plane.urdf")

    hand_id = pb.loadURDF(
        URDF_PATH,
        basePosition=HAND_BASE_POS,
        baseOrientation=HAND_BASE_ORI,
        useFixedBase=True,
    )

    pb.resetDebugVisualizerCamera(
        cameraDistance=0.4,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=HAND_BASE_POS,
    )

    return hand_id


DEFAULT_PARAMS = {
    "hand_scale":    1.5,
    "pinch_thres_1": 0.1,
    "pinch_thres_2": 0.01,
    "wrist_weight":  1.0,
    "pinch_weight":  10.0,
    "orient_weight": 10.0,
    "wjpos_index":   5.0,
    "wjpos_middle":  5.0,
    "wjpos_ring":    5.0,
    "wjpos_thumb":   0.5,
    "ema_alpha":     0.3,
    "opt_maxtime":   0.05,
    "opt_ftol_abs":  1e-5,
    "huber_delta":   0.02,
}


def apply_defaults(retargeter):
    retargeter.hand_scale    = DEFAULT_PARAMS["hand_scale"]
    retargeter.pinch_thres_1 = DEFAULT_PARAMS["pinch_thres_1"]
    retargeter.pinch_thres_2 = DEFAULT_PARAMS["pinch_thres_2"]
    retargeter.wrist_weight  = DEFAULT_PARAMS["wrist_weight"]
    retargeter.pinch_weight  = DEFAULT_PARAMS["pinch_weight"]
    retargeter.orient_weight = DEFAULT_PARAMS["orient_weight"]
    retargeter.wjpos[0]      = DEFAULT_PARAMS["wjpos_index"]
    retargeter.wjpos[4]      = DEFAULT_PARAMS["wjpos_middle"]
    retargeter.wjpos[8]      = DEFAULT_PARAMS["wjpos_ring"]
    retargeter.wjpos[12]     = DEFAULT_PARAMS["wjpos_thumb"]
    retargeter.ema_alpha     = DEFAULT_PARAMS["ema_alpha"]
    retargeter.opt_maxtime   = DEFAULT_PARAMS["opt_maxtime"]
    retargeter.opt_ftol_abs  = DEFAULT_PARAMS["opt_ftol_abs"]
    retargeter.huber_delta   = DEFAULT_PARAMS["huber_delta"]


def reinit_pybullet(retargeter):
    """Disconnect and reconnect PyBullet, reload URDF, recreate sliders with default values."""
    apply_defaults(retargeter)
    pb.disconnect()
    hand_id = setup_pybullet()
    joint_indices = get_joint_indices(hand_id)
    sliders = setup_sliders(retargeter)
    return hand_id, joint_indices, sliders


def setup_sliders(retargeter):
    """Add PyBullet GUI sliders and return dict of slider IDs."""
    s = {}
    s["hand_scale"]    = pb.addUserDebugParameter("hand_scale",    0.5,  3.0,  retargeter.hand_scale)
    s["pinch_thres_1"] = pb.addUserDebugParameter("pinch_thres_1", 0.03, 0.20, retargeter.pinch_thres_1)
    s["pinch_thres_2"] = pb.addUserDebugParameter("pinch_thres_2", 0.005,0.05, retargeter.pinch_thres_2)
    s["pinch_weight"]  = pb.addUserDebugParameter("pinch_weight",  1.0,  30.0, retargeter.pinch_weight)
    s["orient_weight"] = pb.addUserDebugParameter("orient_weight", 0.0,  20.0, retargeter.orient_weight)
    s["ema_alpha"]     = pb.addUserDebugParameter("ema_alpha",     0.1,  1.0,  retargeter.ema_alpha)
    s["wjpos_index"]   = pb.addUserDebugParameter("wjpos_index",   0.0,  10.0, retargeter.wjpos[0])
    s["wjpos_middle"]  = pb.addUserDebugParameter("wjpos_middle",  0.0,  10.0, retargeter.wjpos[4])
    s["wjpos_ring"]    = pb.addUserDebugParameter("wjpos_ring",    0.0,  10.0, retargeter.wjpos[8])
    s["wjpos_thumb"]   = pb.addUserDebugParameter("wjpos_thumb",   0.0,  10.0, retargeter.wjpos[12])
    s["wrist_weight"]  = pb.addUserDebugParameter("wrist_weight",  0.0,  5.0,  retargeter.wrist_weight)
    s["opt_maxtime"]   = pb.addUserDebugParameter("opt_maxtime",   0.01, 0.2,  retargeter.opt_maxtime)
    s["opt_ftol_abs"]  = pb.addUserDebugParameter("opt_ftol_abs",  1e-7, 1e-3, retargeter.opt_ftol_abs)
    s["huber_delta"]   = pb.addUserDebugParameter("huber_delta",   0.005,0.1,  retargeter.huber_delta)
    s["reset"]         = pb.addUserDebugParameter("[ Reset to defaults ]", 1, 0, 1)
    return s


def read_sliders(sliders, retargeter, last_reset):
    """Read slider values and update retargeter attributes.
    Returns current reset counter (compare with last_reset to detect click)."""
    retargeter.hand_scale    = pb.readUserDebugParameter(sliders["hand_scale"])
    retargeter.pinch_thres_1 = pb.readUserDebugParameter(sliders["pinch_thres_1"])
    retargeter.pinch_thres_2 = pb.readUserDebugParameter(sliders["pinch_thres_2"])
    retargeter.pinch_weight  = pb.readUserDebugParameter(sliders["pinch_weight"])
    retargeter.orient_weight = pb.readUserDebugParameter(sliders["orient_weight"])
    retargeter.ema_alpha     = pb.readUserDebugParameter(sliders["ema_alpha"])
    retargeter.wjpos[0]      = pb.readUserDebugParameter(sliders["wjpos_index"])
    retargeter.wjpos[4]      = pb.readUserDebugParameter(sliders["wjpos_middle"])
    retargeter.wjpos[8]      = pb.readUserDebugParameter(sliders["wjpos_ring"])
    retargeter.wjpos[12]     = pb.readUserDebugParameter(sliders["wjpos_thumb"])
    retargeter.wrist_weight  = pb.readUserDebugParameter(sliders["wrist_weight"])
    retargeter.opt_maxtime   = pb.readUserDebugParameter(sliders["opt_maxtime"])
    retargeter.opt_ftol_abs  = pb.readUserDebugParameter(sliders["opt_ftol_abs"])
    retargeter.huber_delta      = pb.readUserDebugParameter(sliders["huber_delta"])
    return int(pb.readUserDebugParameter(sliders["reset"]))


def get_joint_indices(hand_id):
    """Return list of actuated joint indices in pybullet (matching joint names '0'..'15')."""
    joint_indices = []
    for i in range(pb.getNumJoints(hand_id)):
        info = pb.getJointInfo(hand_id, i)
        joint_type = info[2]
        joint_name = info[1].decode("utf-8")
        if joint_type == pb.JOINT_REVOLUTE:
            joint_indices.append(i)
    return joint_indices


# Optimizer output (by joint name '0'-'15'): index(0-3), middle(4-7), ring(8-11), thumb(12-15)
# PyBullet revolute order within each finger: ABD, MCP, PIP, DIP = joints ['1','0','2','3'], ['5','4',...], ['9','8',...], ['12','13',...]
# → only need to swap ABD/MCP within each 3-finger group
PINOCCHIO_TO_PYBULLET = [1, 0, 2, 3, 5, 4, 6, 7, 9, 8, 10, 11, 12, 13, 14, 15]


def apply_qpos(hand_id, joint_indices, qpos):
    qpos_pb = qpos[PINOCCHIO_TO_PYBULLET]
    for i, joint_idx in enumerate(joint_indices):
        pb.resetJointState(hand_id, joint_idx, qpos_pb[i])


def main():
    # -------- setup PyBullet --------
    hand_id = setup_pybullet()
    joint_indices = get_joint_indices(hand_id)
    print(f"PyBullet: {len(joint_indices)} actuated joints")

    # -------- setup detector + retargeter --------
    detector = SingleHandDetector(hand_type="Right")
    retargeter = HandRetargeter(urdf_path=URDF_PATH)
    sliders = setup_sliders(retargeter)

    # -------- setup camera --------
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # Estimated intrinsics for a typical 60° FOV webcam at 640x480
    fx = 554.0
    cam_K = np.array([[fx, 0, 320.0], [0, fx, 240.0], [0, 0, 1.0]])

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Running. Press Q to quit, R to recalibrate wrist reference.")
    frame_idx = 0
    t_prev = time.time()
    last_reset = int(pb.readUserDebugParameter(sliders["reset"]))
    wrist_rot_ref = None   # set on first detection; relative rotations from this neutral pose
    wrist_quat_smooth = np.array([0.0, 0.0, 0.0, 1.0])  # smoothed quaternion (xyzw), start = identity
    WRIST_EMA = 0.15       # fraction of new value per frame; lower = smoother but more lag

    while True:
        ret, bgr = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        num_box, hand_kps, keypoint_2d, wrist_pose_in_cam, wrist_rot = detector.detect(rgb, cam_K)

        current_reset = read_sliders(sliders, retargeter, last_reset)
        if current_reset != last_reset:
            hand_id, joint_indices, sliders = reinit_pybullet(retargeter)
            last_reset = int(pb.readUserDebugParameter(sliders["reset"]))
            wrist_rot_ref = None  # recalibrate on next detection
            wrist_quat_smooth = np.array([0.0, 0.0, 0.0, 1.0])

        if hand_kps is not None:
            # Use PnP rotation (more accurate than SVD approx) if available.
            using_pnp = wrist_pose_in_cam is not None
            R_cur = wrist_pose_in_cam[:3, :3] if using_pnp else wrist_rot
            if frame_idx % 60 == 0:
                print(f"  wrist rot source: {'PnP' if using_pnp else 'SVD fallback'}")
            # On first detection, store reference orientation (= PyBullet identity).
            if wrist_rot_ref is None:
                wrist_rot_ref = R_cur.copy()
                print("Wrist reference set.")
            # Relative rotation expressed in the neutral MANO/PyBullet world frame:
            # R_rel = R_ref.T @ R_cur  (identity at calibration, correct axis directions)
            R_rel = wrist_rot_ref.T @ R_cur
            q_new = sciR.from_matrix(R_rel).as_quat()  # (x,y,z,w)
            # EMA smoothing on quaternion; fix sign to avoid q / -q flipping
            if np.dot(q_new, wrist_quat_smooth) < 0:
                q_new = -q_new
            wrist_quat_smooth = WRIST_EMA * q_new + (1.0 - WRIST_EMA) * wrist_quat_smooth
            wrist_quat_smooth /= np.linalg.norm(wrist_quat_smooth)
            pb.resetBasePositionAndOrientation(hand_id, HAND_BASE_POS, wrist_quat_smooth)

            qpos = retargeter.retarget(hand_kps)
            apply_qpos(hand_id, joint_indices, qpos)
            if frame_idx % 30 == 0:
                pin_names = ['1','0','2','3','12','13','14','15','5','4','6','7','9','8','10','11']
                for i, (n, v) in enumerate(zip(pin_names, qpos)):
                    print(f"  pin[{i:2d}] joint'{n}' = {v:+.3f}")
            annotated = detector.draw_skeleton_on_image(bgr, keypoint_2d)
            rot_label = "rot: PnP" if using_pnp else "rot: SVD"
            cv2.putText(annotated, rot_label, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        else:
            annotated = bgr

        t_now = time.time()
        fps = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev = t_now
        cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Hand Tracking", annotated)

        pb.stepSimulation()

        frame_idx += 1
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            wrist_rot_ref = None
            wrist_quat_smooth = np.array([0.0, 0.0, 0.0, 1.0])
            print("Wrist reference cleared — will recalibrate on next detection.")

    cap.release()
    cv2.destroyAllWindows()
    pb.disconnect()


if __name__ == "__main__":
    main()
