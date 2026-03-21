"""
Відкриває PyBullet і по черзі гне кожен палець, щоб зрозуміти
реальний порядок пальців у LEAP URDF.
"""
import time
import pybullet as pb
import pybullet_data

pb.connect(pb.GUI)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
pb.loadURDF("plane.urdf")
hand_id = pb.loadURDF("assets/leap_hand/leap_hand_right.urdf",
                      basePosition=[0,0,0.1], useFixedBase=True)
pb.resetDebugVisualizerCamera(0.4, 45, -30, [0,0,0.1])

# joint groups: (назва_пальця, [pb_індекси])
# з debug_joints.py: pb 1,2,3,4 = joints 1,0,2,3
#                    pb 6,7,8,9 = joints 5,4,6,7
#                    pb 11,12,13,14 = joints 9,8,10,11
#                    pb 16,17,18,19 = joints 12,13,14,15
GROUPS = [
    ("joints 1,0,2,3  (URDF chain від palm_lower)",  [1,2,3,4]),
    ("joints 5,4,6,7  (URDF chain від palm_lower)",  [6,7,8,9]),
    ("joints 9,8,10,11 (URDF chain від palm_lower)", [11,12,13,14]),
    ("joints 12,13,14,15 (URDF chain від palm_lower)", [16,17,18,19]),
]

for label, indices in GROUPS:
    print(f"\nГнемо: {label}")
    print("Подивись який палець рухається, потім Enter...")

    # reset all
    for i in range(pb.getNumJoints(hand_id)):
        if pb.getJointInfo(hand_id,i)[2] == pb.JOINT_REVOLUTE:
            pb.resetJointState(hand_id, i, 0.0)

    # bend this group
    for idx in indices:
        pb.resetJointState(hand_id, idx, 0.8)

    for _ in range(120):
        pb.stepSimulation()
        time.sleep(1/60)

    input(">>> ")

pb.disconnect()
