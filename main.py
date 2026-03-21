import time
import cv2
import numpy as np
import pybullet as pb
import pybullet_data

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

    # -------- setup camera --------
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        print("Cannot open camera")
        return

    print("Running. Press Q to quit.")
    frame_idx = 0
    t_prev = time.time()

    while True:
        ret, bgr = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        num_box, hand_kps, keypoint_2d, _ = detector.detect(rgb)

        if hand_kps is not None:
            qpos = retargeter.retarget(hand_kps)
            apply_qpos(hand_id, joint_indices, qpos)
            if frame_idx % 30 == 0:
                pin_names = ['1','0','2','3','12','13','14','15','5','4','6','7','9','8','10','11']
                for i, (n, v) in enumerate(zip(pin_names, qpos)):
                    print(f"  pin[{i:2d}] joint'{n}' = {v:+.3f}")
            annotated = detector.draw_skeleton_on_image(bgr, keypoint_2d)
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
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    pb.disconnect()


if __name__ == "__main__":
    main()
