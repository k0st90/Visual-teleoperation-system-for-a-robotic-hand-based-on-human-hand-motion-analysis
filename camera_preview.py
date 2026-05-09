"""
Camera preview — opens PyBullet GUI with the selected hand.
User adjusts camera, presses S to save position.
Prints camera params to stdout on exit.

Usage (internal, called from app.py):
    python camera_preview.py --config configs/leap_hand_right.yml
"""

import argparse
import sys
import os

import pybullet as pb
import pybullet_data

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config_loader import load_retargeting_config

HAND_BASE_POS = [0, 0, 0.35]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--cam-distance", type=float, default=0.7)
    p.add_argument("--cam-yaw",      type=float, default=45.0)
    p.add_argument("--cam-pitch",    type=float, default=-30.0)
    p.add_argument("--out",          default=None, help="File to write camera params")
    args = p.parse_args()

    cfg      = load_retargeting_config(args.config)
    urdf     = cfg["urdf_path"]

    pb.connect(pb.GUI)
    pb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pb.setGravity(0, 0, -9.81)
    pb.loadURDF("plane.urdf")
    pb.loadURDF(urdf, basePosition=HAND_BASE_POS, useFixedBase=True)

    pb.resetDebugVisualizerCamera(
        cameraDistance=args.cam_distance,
        cameraYaw=args.cam_yaw,
        cameraPitch=args.cam_pitch,
        cameraTargetPosition=HAND_BASE_POS,
    )

    pb.configureDebugVisualizer(pb.COV_ENABLE_GUI, 0)

    while True:
        pb.stepSimulation()
        keys = pb.getKeyboardEvents()
        if ord('s') in keys and keys[ord('s')] & pb.KEY_WAS_TRIGGERED:
            cam = pb.getDebugVisualizerCamera()
            yaw   = cam[8]
            pitch = cam[9]
            # compute real distance from view matrix
            import numpy as np
            vm = np.array(cam[2]).reshape(4, 4, order='F')
            cam_pos = -vm[:3, :3].T @ vm[:3, 3]
            dist = float(np.linalg.norm(cam_pos - np.array(HAND_BASE_POS)))
            if args.out:
                with open(args.out, "w") as f:
                    f.write(f"{dist:.4f} {yaw:.4f} {pitch:.4f}")
            break

    pb.disconnect()


if __name__ == "__main__":
    main()
