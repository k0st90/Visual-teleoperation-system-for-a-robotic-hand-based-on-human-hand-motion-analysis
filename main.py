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


def get_joint_indices(hand_id):
    joint_indices = []
    joint_names = []
    for i in range(pb.getNumJoints(hand_id)):
        info = pb.getJointInfo(hand_id, i)
        if info[2] == pb.JOINT_REVOLUTE:
            joint_indices.append(i)
            joint_names.append(info[1].decode("utf-8"))
    return joint_indices, joint_names


def build_joint_mapping(pino_names, pb_names, all_joint_indices):
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
    while not stop_event.is_set():
        ret, bgr = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        num_box, hand_kps, keypoint_2d, wrist_pose_in_cam, wrist_rot = detector.detect(rgb, cam_K)
        result = (bgr, hand_kps, keypoint_2d, wrist_pose_in_cam, wrist_rot)
        try:
            det_queue.get_nowait()
        except queue.Empty:
            pass
        det_queue.put(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/leap_hand_right.yml",
                        help="Path to retargeting config YAML")
    parser.add_argument("--checkpoint", default=None,
                        help="MLP checkpoint path (default: checkpoints/mlp_ss_<config_stem>_best.pt)")
    args = parser.parse_args()

    import pathlib
    from config_loader import load_retargeting_config
    hand_name = pathlib.Path(args.config).stem
    urdf_path  = load_retargeting_config(args.config)["urdf_path"]
    print(f"Config: {args.config}  |  URDF: {urdf_path}")

    hand_id = setup_pybullet(urdf_path)

    from wilor_detector import WilorDetector
    from mlp_selfsupervised.infer import MLPRetargeter

    detector = WilorDetector(hand_type="Right")
    retargeter = HandRetargeter(yml_path=args.config)

    ckpt = args.checkpoint or f"checkpoints/mlp_ss_{hand_name}_best.pt"
    mlp = MLPRetargeter(ckpt)
    print(f"Checkpoint: {ckpt}")

    joint_indices, pb_names = get_joint_indices(hand_id)
    joint_indices, mapping = build_joint_mapping(retargeter.actuated_joints_name, pb_names, joint_indices)
    print(f"PyBullet: {len(joint_indices)} actuated joints")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    fx = 554.0
    cam_K = np.array([[fx, 0, 320.0], [0, fx, 240.0], [0, 0, 1.0]])

    if not cap.isOpened():
        print("Cannot open camera")
        return

    det_queue = queue.Queue(maxsize=1)
    stop_event = threading.Event()
    det_thread = threading.Thread(target=detection_loop,
                                  args=(detector, cap, cam_K, det_queue, stop_event),
                                  daemon=True)
    det_thread.start()
    print("Running. Press Q to quit, R to recalibrate wrist reference.")

    frame_idx = 0
    t_prev = time.time()
    wrist_rot_ref = None
    wrist_quat_smooth = np.array([0.0, 0.0, 0.0, 1.0])
    WRIST_EMA = 0.15
    last_result = None

    while True:
        try:
            last_result = det_queue.get_nowait()
        except queue.Empty:
            pass

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

                qpos = mlp.retarget(hand_kps)
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
