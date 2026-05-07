"""
Video retargeting: extract hand movements from a video and replay in PyBullet.
Produces a side-by-side video: original | PyBullet simulation.

Usage:
    python video_retarget.py --input path/to/video.mp4 --config configs/leap_hand_right.yml
    python video_retarget.py --input video.mp4 --config configs/shadow_hand_right.yml --out result.mp4
"""

import argparse
import os
import sys
import pathlib
import queue
import threading

import cv2
import numpy as np
import pybullet as pb
import pybullet_data
from scipy.spatial.transform import Rotation as sciR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wilor_detector import WilorDetector
from hand_retargeter import HandRetargeter
from config_loader import load_retargeting_config
from mlp_selfsupervised.infer import MLPRetargeter


HAND_BASE_POS = [0, 0, 0.35]
RENDER_W = 640
RENDER_H = 480
R_CAM2WORLD = sciR.from_euler("x", -90, degrees=True).as_matrix()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input",      required=True, help="Input video file")
    p.add_argument("--config",     default="configs/leap_hand_right.yml")
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--out",        default=None, help="Output video path")
    p.add_argument("--fps",        type=float, default=None,
                   help="Output FPS (default: same as input)")
    return p.parse_args()


def setup_pybullet(urdf_path):
    pb.connect(pb.DIRECT)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.81)
    hand_id = pb.loadURDF(
        urdf_path,
        basePosition=HAND_BASE_POS,
        baseOrientation=pb.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
    )
    return hand_id


def get_joint_indices(hand_id):
    joint_indices, joint_names = [], []
    for i in range(pb.getNumJoints(hand_id)):
        info = pb.getJointInfo(hand_id, i)
        if info[2] == pb.JOINT_REVOLUTE:
            joint_indices.append(i)
            joint_names.append(info[1].decode("utf-8"))
    return joint_indices, joint_names


def build_joint_mapping(pino_names, pb_names, all_joint_indices):
    pino_pos = {name: i for i, name in enumerate(pino_names)}
    actuated_indices, mapping = [], []
    for idx, name in zip(all_joint_indices, pb_names):
        if name in pino_pos:
            actuated_indices.append(idx)
            mapping.append(pino_pos[name])
    return actuated_indices, np.array(mapping, dtype=np.int32)


def apply_qpos(hand_id, joint_indices, qpos, mapping):
    qpos_pb = qpos[mapping]
    for i, joint_idx in enumerate(joint_indices):
        pb.resetJointState(hand_id, joint_idx, qpos_pb[i])


def render_pybullet(view_matrix, proj_matrix):
    _, _, rgba, _, _ = pb.getCameraImage(
        width=RENDER_W, height=RENDER_H,
        viewMatrix=view_matrix,
        projectionMatrix=proj_matrix,
        renderer=pb.ER_TINY_RENDERER,
    )
    rgb = np.array(rgba, dtype=np.uint8).reshape(RENDER_H, RENDER_W, 4)[:, :, :3]
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def main():
    args = parse_args()

    if not os.path.isfile(args.input):
        print(f"ERROR: input video not found: {args.input}")
        sys.exit(1)

    hand_name = pathlib.Path(args.config).stem
    cfg = load_retargeting_config(args.config)
    urdf_path = cfg["urdf_path"]

    ckpt = args.checkpoint or f"checkpoints/mlp_ss_{hand_name}_best.pt"
    if not os.path.isfile(ckpt):
        print(f"ERROR: checkpoint not found: {ckpt}")
        sys.exit(1)

    print(f"Config:     {args.config}")
    print(f"Checkpoint: {ckpt}")
    print(f"Input:      {args.input}")

    detector  = WilorDetector(hand_type="Right")
    retargeter = HandRetargeter(yml_path=args.config)
    mlp       = MLPRetargeter(ckpt)

    hand_id = setup_pybullet(urdf_path)
    joint_indices, pb_names = get_joint_indices(hand_id)
    joint_indices, mapping  = build_joint_mapping(
        retargeter.actuated_joints_name, pb_names, joint_indices
    )

    view_matrix = pb.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=HAND_BASE_POS,
        distance=1.1,
        yaw=45, pitch=-30, roll=0,
        upAxisIndex=2,
    )
    proj_matrix = pb.computeProjectionMatrixFOV(
        fov=60, aspect=RENDER_W / RENDER_H,
        nearVal=0.01, farVal=10.0,
    )

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {args.input}")
        sys.exit(1)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_fps = args.fps or src_fps
    total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = args.out or (os.path.splitext(args.input)[0] + "_retarget.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    writer   = cv2.VideoWriter(out_path, fourcc, out_fps, (RENDER_W * 2, RENDER_H))

    cam_K = np.array([[554.0, 0, 320.0], [0, 554.0, 240.0], [0, 0, 1.0]])

    wrist_quat_smooth = np.array([0.0, 0.0, 0.0, 1.0])
    kps_smooth = None
    WRIST_EMA = 0.08
    KPS_EMA   = 0.15
    frame_idx = 0

    # detector runs in a background thread — overlaps with PyBullet render
    frame_q  = queue.Queue(maxsize=2)
    result_q = queue.Queue(maxsize=2)
    stop_ev  = threading.Event()

    def detector_thread():
        while not stop_ev.is_set():
            try:
                item = frame_q.get(timeout=0.5)
            except queue.Empty:
                continue
            if item is None:
                result_q.put(None)
                break
            src_frame, bgr_orig = item
            rgb = cv2.cvtColor(src_frame, cv2.COLOR_BGR2RGB)
            det = detector.detect(rgb, cam_K)
            result_q.put((src_frame, bgr_orig, det))

    t = threading.Thread(target=detector_thread, daemon=True)
    t.start()

    # feed first frames into queue
    def read_and_enqueue():
        while True:
            ret, bgr = cap.read()
            if not ret:
                frame_q.put(None)
                break
            src = cv2.resize(bgr, (RENDER_W, RENDER_H))
            frame_q.put((src, bgr))

    reader = threading.Thread(target=read_and_enqueue, daemon=True)
    reader.start()

    print(f"Processing {total} frames → {out_path}")

    while True:
        item = result_q.get()
        if item is None:
            break
        src_frame, _, det = item
        frame_idx += 1

        num_box, hand_kps, keypoint_2d, wrist_pose_in_cam, wrist_rot = det

        if hand_kps is not None:
            # smooth keypoints to reduce WiLoR jitter
            if kps_smooth is None:
                kps_smooth = hand_kps.copy()
            else:
                kps_smooth = KPS_EMA * hand_kps + (1.0 - KPS_EMA) * kps_smooth

            R_cur   = wrist_pose_in_cam[:3, :3] if wrist_pose_in_cam is not None else wrist_rot
            R_world = R_CAM2WORLD @ R_cur
            q_new   = sciR.from_matrix(R_world).as_quat()
            if np.dot(q_new, wrist_quat_smooth) < 0:
                q_new = -q_new
            wrist_quat_smooth = WRIST_EMA * q_new + (1.0 - WRIST_EMA) * wrist_quat_smooth
            wrist_quat_smooth /= np.linalg.norm(wrist_quat_smooth)
            pb.resetBasePositionAndOrientation(hand_id, HAND_BASE_POS, wrist_quat_smooth)

            qpos = mlp.retarget(kps_smooth)
            apply_qpos(hand_id, joint_indices, qpos, mapping)

            src_frame = detector.draw_skeleton_on_image(src_frame, keypoint_2d)

        pb.stepSimulation()
        sim_frame = render_pybullet(view_matrix, proj_matrix)

        side_by_side = np.concatenate([src_frame, sim_frame], axis=1)

        label_orig = "Original"
        label_sim  = "PyBullet"
        cv2.putText(side_by_side, label_orig, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(side_by_side, label_sim, (RENDER_W + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        writer.write(side_by_side)

        if frame_idx % 30 == 0:
            print(f"  frame {frame_idx}/{total}")

    stop_ev.set()
    cap.release()
    writer.release()
    pb.disconnect()
    print(f"\nDone. Saved → {out_path}")


if __name__ == "__main__":
    main()
