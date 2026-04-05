"""
config_loader.py — loads retargeting YAML configs (extended dex-retargeting format).

YAML format (configs/leap_hand_right.yml):
  retargeting:
    type: vector
    urdf_path: ...
    wrist_link: ...
    scaling_factor: ...
    target_origin_link_names: [...]   # wrist vectors origin (dex-compatible)
    target_task_link_names:   [...]   # wrist vectors task   (dex-compatible)
    pinch_origin_link_names:  [...]   # pinch vectors origin (mingrui extension)
    pinch_task_link_names:    [...]   # pinch vectors task   (mingrui extension)
    orient_link_pairs: [[o,t], ...]   # orient vectors       (mingrui extension)
    capsule_defs: [[a,b,r], ...]      # capsule definitions
    capsule_collision_pairs: [[i,j]]  # which capsule pairs to check
"""

import yaml


def load_retargeting_config(yml_path: str) -> dict:
    """Load and parse a retargeting YAML config file.

    Returns a dict with keys:
      urdf_path         str
      wrist_link        str
      scaling_factor    float
      target_link_pairs list of (origin_link, task_link)  — all 11 in correct order:
                          [0:4]  wrist vectors
                          [4:7]  pinch vectors
                          [7:11] orient vectors
      capsule_defs      list of (link_a, link_b, default_radius)
      capsule_collision_pairs  list of (i, j)
    """
    with open(yml_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = raw["retargeting"]

    urdf_path      = cfg["urdf_path"]
    wrist_link     = cfg["wrist_link"]
    scaling_factor = float(cfg.get("scaling_factor", 1.5))

    # wrist vectors (4)
    wrist_origins = cfg["target_origin_link_names"]
    wrist_tasks   = cfg["target_task_link_names"]
    wrist_pairs   = list(zip(wrist_origins, wrist_tasks))

    # pinch vectors (3)
    pinch_origins = cfg["pinch_origin_link_names"]
    pinch_tasks   = cfg["pinch_task_link_names"]
    pinch_pairs   = list(zip(pinch_origins, pinch_tasks))

    # orient vectors (4)
    orient_pairs = [tuple(p) for p in cfg["orient_link_pairs"]]

    n_fingers = len(wrist_pairs)  # number of robot fingers (3–5)

    # all pairs in order: wrist[0:N] + pinch[N:2N-1] + orient[2N-1:3N-1]
    target_link_pairs = wrist_pairs + pinch_pairs + orient_pairs

    # actuated joint names — optional, None means use hand_retargeter defaults
    actuated_joints_name = cfg.get("actuated_joints_name", None)
    if actuated_joints_name is not None:
        actuated_joints_name = [str(j) for j in actuated_joints_name]

    # touch (fixed) joint names — joints present in URDF but not controlled (e.g. wrist)
    touch_joints_name_raw = cfg.get("touch_joints_name", None)
    touch_joints_name = [str(j) for j in touch_joints_name_raw] if touch_joints_name_raw else []

    # per-joint position weights — optional, None means use hand_retargeter defaults
    weights_joint_pos_raw = cfg.get("weights_joint_pos", None)
    weights_joint_pos = list(map(float, weights_joint_pos_raw)) if weights_joint_pos_raw is not None else None

    # capsule defs and collision pairs are optional — None means no collision avoidance
    if "capsule_defs" in cfg and "capsule_collision_pairs" in cfg:
        capsule_defs            = [(str(a), str(b), float(r)) for a, b, r in cfg["capsule_defs"]]
        capsule_collision_pairs = [(int(i), int(j)) for i, j in cfg["capsule_collision_pairs"]]
    else:
        capsule_defs            = None
        capsule_collision_pairs = None

    return {
        "urdf_path":               urdf_path,
        "wrist_link":              wrist_link,
        "scaling_factor":          scaling_factor,
        "n_fingers":               n_fingers,
        "actuated_joints_name":    actuated_joints_name,
        "touch_joints_name":       touch_joints_name,
        "weights_joint_pos":       weights_joint_pos,
        "target_link_pairs":       target_link_pairs,
        "capsule_defs":            capsule_defs,
        "capsule_collision_pairs": capsule_collision_pairs,
    }
