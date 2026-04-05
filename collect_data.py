"""
Data collection script for MLP retargeter training.

Runs the full pipeline (WiLoR → HandRetargeter → NLopt optimizer) and logs:
  - human keypoints in wrist frame (21, 3)
  - qpos output from optimizer        (n_doa,)

Usage:
    python collect_data.py --hand allegro --out data/allegro_session1.npz
    python collect_data.py --hand leap    --out data/leap_session1.npz

Collect multiple sessions (different hand poses, speeds) and concatenate.
Target: 50k+ frames per hand model.
"""

import argparse
import sys
import os
import time
import threading
import queue

import cv2
import numpy as np

from wilor_detector import WilorDetector
from hand_detector import SingleHandDetector
from hand_retargeter import HandRetargeter


HAND_CONFIGS = {
    "leap":    "configs/leap_hand_right.yml",
    "allegro": "configs/allegro_hand_right.yml",
    "shadow":  "configs/shadow_hand_right.yml",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--hand",     default="allegro", choices=list(HAND_CONFIGS))
    p.add_argument("--detector", default="wilor",   choices=["wilor", "mediapipe"])
    p.add_argument("--out",      default=None,      help="Output .npz path (default: data/<hand>_<timestamp>.npz)")
    p.add_argument("--show",     action="store_true", help="Show camera window")
    return p.parse_args()


def main():
    args = parse_args()
    yml_path = HAND_CONFIGS[args.hand]

    if args.out is None:
        os.makedirs("data", exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = f"data/{args.hand}_{ts}.npz"

    # ── detector (runs in background thread) ──────────────────────────────
    if args.detector == "wilor":
        detector = WilorDetector(hand_type="Right")
    else:
        detector = SingleHandDetector(hand_type="Right")

    # ── retargeter (NLopt optimizer inside) ───────────────────────────────
    print(f"Loading retargeter: {yml_path}")
    retargeter = HandRetargeter(yml_path=yml_path)
    n_doa = retargeter.robot_adaptor.doa
    print(f"DOA: {n_doa}")

    # ── camera ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        sys.exit(1)

    # ── detector thread (same pattern as main.py) ─────────────────────────
    frame_queue  = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=1)
    stop_event   = threading.Event()

    def detector_thread():
        while not stop_event.is_set():
            try:
                bgr = frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = detector.detect(rgb)
            # replace stale result
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
            result_queue.put(result)

    t = threading.Thread(target=detector_thread, daemon=True)
    t.start()

    # ── collection buffers ────────────────────────────────────────────────
    buf_kps  = []   # (21, 3) each
    buf_qpos = []   # (n_doa,) each

    frame_count  = 0
    saved_count  = 0
    t0           = time.time()

    print(f"Collecting data → {args.out}")
    print("Move your hand in front of the camera. Press Q to stop.")

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frame_count += 1

        # push to detector (drop if busy)
        try:
            frame_queue.put_nowait(bgr)
        except queue.Full:
            pass

        # get latest detection result
        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            result = None

        if result is not None:
            _, joint_pos, _, _ = result   # joint_pos: (21, 3) in wrist frame
            if joint_pos is not None:
                qpos = retargeter.retarget(joint_pos)
                # store raw kps (before hand_scale) — retarget() applies scale internally
                buf_kps.append(joint_pos.astype(np.float32))
                buf_qpos.append(qpos.astype(np.float32))
                saved_count += 1

        # HUD
        elapsed = time.time() - t0
        fps_cam  = frame_count / max(elapsed, 1e-3)
        fps_save = saved_count / max(elapsed, 1e-3)
        hud = f"frames:{frame_count}  saved:{saved_count}  cam:{fps_cam:.1f}fps  save:{fps_save:.1f}fps  Q=quit"
        if args.show:
            cv2.putText(bgr, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("collect_data", bgr)
        else:
            if frame_count % 60 == 0:
                print(hud)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    # ── save ──────────────────────────────────────────────────────────────
    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()

    if not buf_kps:
        print("No data collected.")
        return

    kps_arr  = np.stack(buf_kps,  axis=0)   # (N, 21, 3)
    qpos_arr = np.stack(buf_qpos, axis=0)   # (N, n_doa)

    np.savez_compressed(args.out, kps=kps_arr, qpos=qpos_arr)
    print(f"\nSaved {saved_count} samples → {args.out}")
    print(f"  kps  shape: {kps_arr.shape}")
    print(f"  qpos shape: {qpos_arr.shape}")


if __name__ == "__main__":
    main()
