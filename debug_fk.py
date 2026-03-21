"""
Перевіряємо чи joints 12,13,14,15 дійсно рухають thumb_tip у pinocchio FK.
"""
import numpy as np
from robot_model import RobotPinocchio

robot = RobotPinocchio("assets/leap_hand/leap_hand_right.urdf")

# Baseline: qpos = 0
q0 = np.zeros(16)
robot.compute_forward_kinematics(q0)
tips_zero = {
    "thumb_tip":  robot.get_frame_pose("thumb_tip")[:3, 3],
    "index_tip":  robot.get_frame_pose("index_tip")[:3, 3],
    "middle_tip": robot.get_frame_pose("middle_tip")[:3, 3],
    "ring_tip":   robot.get_frame_pose("ring_tip")[:3, 3],
}
print("=== Baseline (all zeros) ===")
for k, v in tips_zero.items():
    print(f"  {k}: {np.round(v, 4)}")

print()
print("=== Рух кожного joint окремо (qpos[i]=1.0) ===")

# Pinocchio joint names
pin_names = ['1','0','2','3','12','13','14','15','5','4','6','7','9','8','10','11']
finger_label = ['IDX','IDX','IDX','IDX','THB','THB','THB','THB',
                'MID','MID','MID','MID','RNG','RNG','RNG','RNG']

for i in range(16):
    q = np.zeros(16)
    q[i] = 1.0
    robot.compute_forward_kinematics(q)
    thumb_pos  = robot.get_frame_pose("thumb_tip")[:3, 3]
    index_pos  = robot.get_frame_pose("index_tip")[:3, 3]
    middle_pos = robot.get_frame_pose("middle_tip")[:3, 3]
    ring_pos   = robot.get_frame_pose("ring_tip")[:3, 3]

    delta_thumb  = np.linalg.norm(thumb_pos  - tips_zero["thumb_tip"])
    delta_index  = np.linalg.norm(index_pos  - tips_zero["index_tip"])
    delta_middle = np.linalg.norm(middle_pos - tips_zero["middle_tip"])
    delta_ring   = np.linalg.norm(ring_pos   - tips_zero["ring_tip"])

    moved = []
    if delta_thumb  > 1e-4: moved.append(f"thumb(Δ={delta_thumb:.4f})")
    if delta_index  > 1e-4: moved.append(f"index(Δ={delta_index:.4f})")
    if delta_middle > 1e-4: moved.append(f"middle(Δ={delta_middle:.4f})")
    if delta_ring   > 1e-4: moved.append(f"ring(Δ={delta_ring:.4f})")

    print(f"  pin[{i:2d}] joint'{pin_names[i]}' ({finger_label[i]}): {', '.join(moved) if moved else 'no movement'}")
