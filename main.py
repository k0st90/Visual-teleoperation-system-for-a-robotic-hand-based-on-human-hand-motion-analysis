import argparse
import queue
import threading
import time
import cv2
import numpy as np
import pybullet as pb
import pybullet_data
from scipy.spatial.transform import Rotation as sciR

from hand_retargeter import HandRetargeter

HAND_CONFIGS = {
    "leap":    ("assets/leap_hand/leap_hand_right.urdf",    "configs/leap_hand_right.yml"),
    "allegro": ("assets/allegro_hand/allegro_hand_right.urdf", "configs/allegro_hand_right.yml"),
    "shadow":  ("assets/shadow_hand/shadow_hand_right.urdf",   "configs/shadow_hand_right.yml"),
}

HAND_BASE_POS = [0, 0, 0.1]
HAND_BASE_ORI = pb.getQuaternionFromEuler([0, 0, 0])


def setup_pybullet(urdf_path):
    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.81)
    pb.loadURDF("plane.urdf")

    hand_id = pb.loadURDF(
        urdf_path,
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


def reinit_pybullet(retargeter, urdf_path):
    """Disconnect and reconnect PyBullet, reload URDF, recreate sliders with default values."""
    apply_defaults(retargeter)
    pb.disconnect()
    hand_id = setup_pybullet(urdf_path)
    joint_indices, pb_names = get_joint_indices(hand_id)
    joint_indices, mapping = build_joint_mapping(retargeter.actuated_joints_name, pb_names, joint_indices)
    sliders = setup_sliders(retargeter)
    return hand_id, joint_indices, mapping, sliders


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
    """Return (joint_indices, joint_names) for revolute joints in PyBullet URDF order."""
    joint_indices = []
    joint_names = []
    for i in range(pb.getNumJoints(hand_id)):
        info = pb.getJointInfo(hand_id, i)
        if info[2] == pb.JOINT_REVOLUTE:
            joint_indices.append(i)
            joint_names.append(info[1].decode("utf-8"))
    return joint_indices, joint_names


def build_joint_mapping(pino_names, pb_names, all_joint_indices):
    """Returns (actuated_joint_indices, mapping) where mapping permutes optimizer
    output to match PyBullet joint order. Touch joints (e.g. WRJ1/WRJ2) are
    excluded — they stay at their default PyBullet position (0).
    """
    pino_pos = {name: i for i, name in enumerate(pino_names)}
    actuated_indices = []
    mapping = []
    for idx, name in zip(all_joint_indices, pb_names):
        if name in pino_pos:
            actuated_indices.append(idx)
            mapping.append(pino_pos[name])
    return actuated_indices, np.array(mapping, dtype=np.int32)


def apply_qpos(hand_id, joint_indices, qpos, mapping):
    qpos_pb = qpos[mapping]
    for i, joint_idx in enumerate(joint_indices):
        pb.resetJointState(hand_id, joint_idx, qpos_pb[i])


def detection_loop(detector, cap, cam_K, det_queue, stop_event):
    """Runs in a separate thread: captures frames and runs hand detection.
    Puts (bgr, hand_kps, keypoint_2d, wrist_pose_in_cam, wrist_rot) into det_queue.
    Queue size=1: old results are dropped so main thread always gets the latest frame.
    """
    while not stop_event.is_set():
        ret, bgr = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        num_box, hand_kps, keypoint_2d, wrist_pose_in_cam, wrist_rot = detector.detect(rgb, cam_K)
        result = (bgr, hand_kps, keypoint_2d, wrist_pose_in_cam, wrist_rot)
        # Drop stale result if main thread hasn't consumed it yet
        try:
            det_queue.get_nowait()
        except queue.Empty:
            pass
        det_queue.put(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--detector", choices=["wilor", "mediapipe"], default="wilor",
                        help="Hand detector backend (default: wilor)")
    parser.add_argument("--hand", choices=list(HAND_CONFIGS.keys()), default="leap",
                        help="Robot hand to use (default: leap)")
    args = parser.parse_args()

    urdf_path, yml_path = HAND_CONFIGS[args.hand]
    print(f"Hand: {args.hand}  |  URDF: {urdf_path}  |  Config: {yml_path}")

    # -------- setup PyBullet --------
    hand_id = setup_pybullet(urdf_path)

    # -------- setup detector + retargeter --------
    if args.detector == "wilor":
        from wilor_detector import WilorDetector
        detector = WilorDetector(hand_type="Right")
    else:
        from hand_detector import SingleHandDetector
        detector = SingleHandDetector(hand_type="Right")
    retargeter = HandRetargeter(yml_path=yml_path)

    joint_indices, pb_names = get_joint_indices(hand_id)
    joint_indices, mapping = build_joint_mapping(retargeter.actuated_joints_name, pb_names, joint_indices)
    print(f"PyBullet: {len(joint_indices)} actuated joints")
    sliders = setup_sliders(retargeter)

    # -------- setup camera --------
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    fx = 554.0
    cam_K = np.array([[fx, 0, 320.0], [0, fx, 240.0], [0, 0, 1.0]])

    if not cap.isOpened():
        print("Cannot open camera")
        return

    # -------- start detector thread --------
    det_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    det_thread = threading.Thread(target=detection_loop,
                                  args=(detector, cap, cam_K, det_queue, stop_event),
                                  daemon=True)
    det_thread.start()
    print("Running. Press Q to quit, R to recalibrate wrist reference.")

    frame_idx = 0
    t_prev = time.time()
    last_reset = int(pb.readUserDebugParameter(sliders["reset"]))
    wrist_rot_ref = None
    wrist_quat_smooth = np.array([0.0, 0.0, 0.0, 1.0])
    WRIST_EMA = 0.15
    last_result = None  # last valid detection result (reused when queue is empty)

    while True:
        # Get latest detection — non-blocking, reuse last if nothing new yet
        try:
            last_result = det_queue.get_nowait()
        except queue.Empty:
            pass

        current_reset = read_sliders(sliders, retargeter, last_reset)
        if current_reset != last_reset:
            hand_id, joint_indices, mapping, sliders = reinit_pybullet(retargeter, urdf_path)
            last_reset = int(pb.readUserDebugParameter(sliders["reset"]))
            wrist_rot_ref = None
            wrist_quat_smooth = np.array([0.0, 0.0, 0.0, 1.0])
            last_result = None

        if last_result is not None:
            bgr, hand_kps, keypoint_2d, wrist_pose_in_cam, wrist_rot = last_result

            if hand_kps is not None:
                using_pnp = wrist_pose_in_cam is not None
                R_cur = wrist_pose_in_cam[:3, :3] if using_pnp else wrist_rot
                if frame_idx % 60 == 0:
                    print(f"  wrist rot source: {'PnP' if using_pnp else 'SVD fallback'}")
                if wrist_rot_ref is None:
                    wrist_rot_ref = R_cur.copy()
                    print("Wrist reference set.")
                R_rel = wrist_rot_ref.T @ R_cur
                q_new = sciR.from_matrix(R_rel).as_quat()
                if np.dot(q_new, wrist_quat_smooth) < 0:
                    q_new = -q_new
                wrist_quat_smooth = WRIST_EMA * q_new + (1.0 - WRIST_EMA) * wrist_quat_smooth
                wrist_quat_smooth /= np.linalg.norm(wrist_quat_smooth)
                pb.resetBasePositionAndOrientation(hand_id, HAND_BASE_POS, wrist_quat_smooth)

                qpos = retargeter.retarget(hand_kps)
                apply_qpos(hand_id, joint_indices, qpos, mapping)
                annotated = detector.draw_skeleton_on_image(bgr, keypoint_2d)
                rot_label = "rot: PnP" if using_pnp else "rot: SVD"
                cv2.putText(annotated, rot_label, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            else:
                annotated = bgr
        else:
            annotated = np.zeros((480, 640, 3), dtype=np.uint8)

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

    stop_event.set()
    det_thread.join(timeout=2.0)
    cap.release()
    cv2.destroyAllWindows()
    pb.disconnect()


if __name__ == "__main__":
    main()
