"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "brainco-retargeting @ ..",
#   "mediapipe>=0.10",
#   "opencv-python>=4.8",
#   "numpy>=1.24",
# ]
# ///

Record hand landmarks (no retargeting) from a video file or live webcam.

Outputs a .npz file with keys:
    landmarks  – np.ndarray of shape (N, 25, 3), float64
    side       – str, "left" or "right"

Frames where no hand is detected are dropped (not included in the output).

Usage:
    uv run demos/record_video.py --input hand.mp4 --hand right --output-npz out.npz
    uv run demos/record_video.py --live --camera-id 0 --hand right --output-npz out.npz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import mediapipe as mp
import numpy as np

from _utils import draw_hand_skeleton_mp21, mp21_to_xr25


def parse_args():
    p = argparse.ArgumentParser(description="Record hand landmarks from video or webcam")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--input", type=str, help="Input video file path")
    src.add_argument("--live", action="store_true", help="Use live webcam input")
    p.add_argument("--camera-id", type=int, default=0, help="Camera device index (with --live)")
    p.add_argument(
        "--hand",
        choices=["left", "right", "auto"],
        default="auto",
        help="Which hand to track. 'auto' uses MediaPipe handedness (mirrors front-cam convention).",
    )
    p.add_argument("--output-npz", type=str, required=True, help="Output .npz file path")
    p.add_argument("--preview", action="store_true", help="Show live preview window")
    return p.parse_args()


def main():
    args = parse_args()

    if args.live:
        cap = cv2.VideoCapture(args.camera_id)
        print(f"Recording from camera {args.camera_id}. Press 'q' to stop and save.")
    else:
        if not Path(args.input).exists():
            sys.exit(f"Input file not found: {args.input}")
        cap = cv2.VideoCapture(args.input)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Processing {args.input}  ({total} frames)")

    if not cap.isOpened():
        sys.exit("Cannot open video source")

    mp_hands_mod = mp.solutions.hands
    hands = mp_hands_mod.Hands(
        static_image_mode=not args.live,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    accumulated: list[np.ndarray] = []
    detected_side: str | None = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_world_landmarks and results.multi_handedness:
            wl = results.multi_hand_world_landmarks[0]
            mp21 = np.array([[lm.x, lm.y, lm.z] for lm in wl.landmark], dtype=np.float64)
            xr25 = mp21_to_xr25(mp21)

            mp_side_raw = results.multi_handedness[0].classification[0].label
            mp_side = "right" if mp_side_raw == "Left" else "left"
            if args.hand != "auto":
                mp_side = args.hand
            detected_side = mp_side

            accumulated.append(xr25)

            if args.preview and results.multi_hand_landmarks:
                draw_hand_skeleton_mp21(frame, results.multi_hand_landmarks[0])

        if args.preview:
            cv2.putText(frame, f"Frames: {len(accumulated)}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("Record – press q to stop", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        if not args.live and frame_idx % 100 == 0:
            print(f"  frame {frame_idx}, detected {len(accumulated)} hands")

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    if not accumulated:
        sys.exit("No hand landmarks detected. Nothing to save.")

    out_path = Path(args.output_npz)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.stack(accumulated)
    side_str = detected_side or "unknown"
    np.savez(str(out_path), landmarks=arr, side=side_str)
    print(f"Saved {len(accumulated)} frames → {out_path}  (side={side_str}, shape={arr.shape})")


if __name__ == "__main__":
    main()
