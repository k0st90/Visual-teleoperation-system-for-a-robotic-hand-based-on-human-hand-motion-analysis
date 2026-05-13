"""
Data collection for self-supervised MLP training.

Runs only the detector (WiLoR/MediaPipe) — NO optimizer.
Saves raw keypoints in wrist frame: (21, 3) per frame.

Usage:
    python mlp_selfsupervised/collect_data.py --out data/kps_session1.npz
    python mlp_selfsupervised/collect_data.py --detector mediapipe --out data/kps_session2.npz

Target: 50k+ frames of diverse hand poses.
"""

import argparse
import os
import sys
import time
import threading
import queue

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from detection import WilorDetector


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=None)
    p.add_argument("--show", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    if args.out is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        args.out = f"data/kps_{ts}.npz"
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    detector = WilorDetector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: cannot open camera")
        sys.exit(1)

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
            try:
                result_queue.get_nowait()
            except queue.Empty:
                pass
            result_queue.put(result)

    t = threading.Thread(target=detector_thread, daemon=True)
    t.start()

    buf_kps = []
    frame_count = saved_count = 0
    t0 = time.time()

    print(f"Collecting keypoints → {args.out}")
    print("Move your hand. Press Q to stop.")

    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        frame_count += 1

        try:
            frame_queue.put_nowait(bgr)
        except queue.Full:
            pass

        try:
            result = result_queue.get_nowait()
        except queue.Empty:
            result = None

        if result is not None:
            _, joint_pos, *_ = result
            if joint_pos is not None:
                buf_kps.append(joint_pos.astype(np.float32))
                saved_count += 1

        elapsed = time.time() - t0
        if args.show:
            hud = f"saved:{saved_count}  fps:{frame_count/max(elapsed,1e-3):.1f}  Q=quit"
            cv2.putText(bgr, hud, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("collect", bgr)
        elif frame_count % 60 == 0:
            print(f"frames:{frame_count}  saved:{saved_count}  fps:{frame_count/max(elapsed,1e-3):.1f}")

        if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
            break

    stop_event.set()
    cap.release()
    cv2.destroyAllWindows()

    if not buf_kps:
        print("No data collected.")
        return

    kps_arr = np.stack(buf_kps, axis=0)  # (N, 21, 3)
    np.savez_compressed(args.out, kps=kps_arr)
    print(f"\nSaved {saved_count} samples → {args.out}")
    print(f"  kps shape: {kps_arr.shape}")


if __name__ == "__main__":
    main()
