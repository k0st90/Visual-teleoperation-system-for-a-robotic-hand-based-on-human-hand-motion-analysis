import pybullet as pb
import pybullet_data
from robot_model import RobotPinocchio

pb.connect(pb.DIRECT)
pb.setAdditionalSearchPath(pybullet_data.getDataPath())
hand_id = pb.loadURDF("assets/leap_hand/leap_hand_right.urdf", useFixedBase=True)

print("PyBullet joints:")
pb_joints = []
for i in range(pb.getNumJoints(hand_id)):
    info = pb.getJointInfo(hand_id, i)
    if info[2] == pb.JOINT_REVOLUTE:
        print(f"  pb_idx={i}  name={info[1].decode()}")
        pb_joints.append((i, info[1].decode()))
pb.disconnect()

print()
robot = RobotPinocchio("assets/leap_hand/leap_hand_right.urdf")
print("Pinocchio joints:", robot.dof_joint_names)
