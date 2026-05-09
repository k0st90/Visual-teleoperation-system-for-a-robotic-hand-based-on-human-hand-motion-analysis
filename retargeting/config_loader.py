"""
config_loader.py — loads retargeting YAML configs.

YAML format (configs/leap_hand_right.yml):
  retargeting:
    urdf_path: ...
    wrist_link: ...
    scaling_factor: ...
    target_origin_link_names: [...]   # wrist vectors origin
    target_task_link_names:   [...]   # wrist vectors task
    pinch_origin_link_names:  [...]   # pinch vectors origin
    pinch_task_link_names:    [...]   # pinch vectors task
    orient_link_pairs: [[o,t], ...]   # orient vectors
"""

import os
import xml.etree.ElementTree as ET

import yaml


def _urdf_parent_map(urdf_path: str) -> dict:
    """Returns {frame_name: parent_link} covering both child-link names and joint names.
    Pinocchio creates frames for both, so we index by both to handle any naming convention.
    """
    root = ET.parse(urdf_path).getroot()
    pmap = {}
    for j in root.findall("joint"):
        jname = j.get("name")
        parent = j.find("parent")
        child  = j.find("child")
        if parent is None or child is None:
            continue
        parent_link = parent.get("link")
        pmap[child.get("link")] = parent_link  # frame named after child link
        pmap[jname]             = parent_link  # frame named after joint (fixed joints)
    return pmap


def _auto_orient_pairs(tip_links: list, urdf_path: str) -> list:
    """Derive (penultimate, tip) orient pairs from the URDF kinematic tree."""
    parent_map = _urdf_parent_map(urdf_path)
    pairs = []
    for tip in tip_links:
        parent = parent_map.get(tip)
        if parent is None:
            raise ValueError(
                f"Cannot auto-detect orient pair for '{tip}': "
                f"frame not found in URDF joint tree of {urdf_path}"
            )
        pairs.append((parent, tip))
    return pairs


def load_retargeting_config(yml_path: str, assets_path: str = None) -> dict:
    """Load and parse a retargeting YAML config file.

    Returns a dict with keys:
      urdf_path         str
      wrist_link        str
      scaling_factor    float
      target_link_pairs list of (origin_link, task_link)  — all 11 in correct order:
                          [0:4]  wrist vectors
                          [4:7]  pinch vectors
                          [7:11] orient vectors
    """
    if not os.path.isfile(yml_path):
        raise FileNotFoundError(f"Config not found: {yml_path}")

    with open(yml_path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = raw["retargeting"]

    urdf_path = cfg["urdf_path"]
    if not os.path.isabs(urdf_path):
        base = assets_path if assets_path else os.path.dirname(os.path.abspath(yml_path))
        urdf_path = os.path.join(base, urdf_path)

    if not os.path.isfile(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
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

    # orient vectors — auto-detected from URDF if not specified in config
    if "orient_link_pairs" in cfg:
        orient_pairs = [tuple(p) for p in cfg["orient_link_pairs"]]
    else:
        orient_pairs = _auto_orient_pairs(wrist_tasks, urdf_path)

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

    return {
        "urdf_path":            urdf_path,
        "wrist_link":           wrist_link,
        "scaling_factor":       scaling_factor,
        "n_fingers":            n_fingers,
        "actuated_joints_name": actuated_joints_name,
        "touch_joints_name":    touch_joints_name,
        "weights_joint_pos":    weights_joint_pos,
        "target_link_pairs":    target_link_pairs,
    }
