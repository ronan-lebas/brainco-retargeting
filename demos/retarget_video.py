# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mediapipe>=0.10",
#   "sapien>=3.0.0",
#   "opencv-python>=4.8",
#   "numpy>=1.24",
#   "pyyaml",
#   "dex-retargeting @ git+https://github.com/dexsuite/dex-retargeting",
#   "torch",
# ]
#
# [[tool.uv.indexes]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Run end-to-end retargeting from a video file.

MediaPipe extracts hand landmarks from each frame; BrainCo retargeting maps them
to 6 motor commands. At least one output must be specified.

    --output-npz    saves motor commands, shape (N, 6), values in [0, 1]
    --output-video  side-by-side video: source frame with skeleton | Sapien hand

Usage:
    uv run demos/retarget_video.py --input clip.mp4 --hand right --output-npz motors.npz
    uv run demos/retarget_video.py --input clip.mp4 --hand right --output-video out.mp4
    uv run demos/retarget_video.py --input clip.mp4 --hand right \\
        --output-npz motors.npz --output-video out.mp4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import mediapipe as mp
import numpy as np

from _utils import SapienHandRenderer, VideoWriter, draw_hand_skeleton_mp21, draw_motor_bars, mp21_to_xr25
from brainco_retargeting import BrainCoRetargeter


def parse_args():
    p = argparse.ArgumentParser(description="Retarget hand motion from a video file")
    p.add_argument("--input", type=str, required=True, help="Input video file path")
    p.add_argument(
        "--hand",
        choices=["left", "right", "auto"],
        default="auto",
        help="Which hand to retarget. 'auto' mirrors MediaPipe's front-cam convention.",
    )
    p.add_argument("--output-npz", type=str, default=None, help="Output .npz for motor commands")
    p.add_argument("--output-video", type=str, default=None,
                   help="Output side-by-side video (camera | Sapien hand)")
    p.add_argument("--panel-width", type=int, default=640, help="Width of each panel in output video")
    p.add_argument("--panel-height", type=int, default=480, help="Height of each panel in output video")
    args = p.parse_args()
    if not args.output_npz and not args.output_video:
        p.error("At least one of --output-npz or --output-video must be specified")
    return args


def main():
    args = parse_args()

    if not Path(args.input).exists():
        sys.exit(f"Input file not found: {args.input}")

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.input}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"Input: {args.input}  ({total} frames @ {src_fps:.1f} fps)")

    retargeter = BrainCoRetargeter()
    mp_hands_mod = mp.solutions.hands
    hands = mp_hands_mod.Hands(
        static_image_mode=True,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    renderer: SapienHandRenderer | None = None
    video_writer: VideoWriter | None = None
    pw, ph = args.panel_width, args.panel_height

    all_motors: list[np.ndarray] = []
    detected_side: str | None = None
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        motors = np.zeros(6)
        cam_panel = cv2.resize(frame, (pw, ph))

        if results.multi_hand_world_landmarks and results.multi_handedness:
            wl = results.multi_hand_world_landmarks[0]
            mp21 = np.array([[lm.x, lm.y, lm.z] for lm in wl.landmark], dtype=np.float64)
            xr25 = mp21_to_xr25(mp21)

            mp_side_raw = results.multi_handedness[0].classification[0].label
            mp_side = "right" if mp_side_raw == "Left" else "left"
            if args.hand != "auto":
                mp_side = args.hand

            if detected_side != mp_side:
                detected_side = mp_side
                if args.output_video:
                    # Re-initialise renderer if side changes
                    renderer = SapienHandRenderer(detected_side, pw, ph)

            retarget_fn = (retargeter.retarget_left if detected_side == "left"
                           else retargeter.retarget_right)
            motors = retarget_fn(xr25)
            all_motors.append(motors.copy())

            if results.multi_hand_landmarks:
                draw_hand_skeleton_mp21(cam_panel, results.multi_hand_landmarks[0])

        if args.output_video:
            # Lazily create renderer and writer on first side detection
            if renderer is None and detected_side:
                renderer = SapienHandRenderer(detected_side, pw, ph)

            if video_writer is None and detected_side:
                out_path = Path(args.output_video)
                out_path.parent.mkdir(parents=True, exist_ok=True)
                video_writer = VideoWriter(str(out_path), src_fps, pw * 2, ph)

            if renderer is not None and video_writer is not None:
                robot_panel = renderer.render(motors)
                robot_panel = draw_motor_bars(robot_panel, motors)
                combined = np.hstack([cam_panel, robot_panel])
                video_writer.write(combined)

        if (frame_idx) % 100 == 0 or frame_idx == total:
            print(f"  {frame_idx}/{total} frames, detected={len(all_motors)}")

    cap.release()
    hands.close()
    if video_writer is not None:
        video_writer.release()
        print(f"Video saved → {args.output_video}")

    if args.output_npz:
        if not all_motors:
            print("Warning: no hand detected, output npz will be empty")
            motors_arr = np.zeros((0, 6), dtype=np.float64)
        else:
            motors_arr = np.stack(all_motors)
        out_path = Path(args.output_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out_path), motors=motors_arr)
        print(f"Motors saved → {args.output_npz}  shape={motors_arr.shape}")


if __name__ == "__main__":
    main()
