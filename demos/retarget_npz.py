# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "sapien>=3.0.0",
#   "opencv-python>=4.8",
#   "numpy>=1.24",
#   "pyyaml",
#   "dex-retargeting @ git+https://github.com/dexsuite/dex-retargeting",
#   "torch",
# ]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Retarget hand landmarks from a .npz file.

Input .npz must contain:
    landmarks  – np.ndarray of shape (N, 25, 3)
    side       – str, "left" or "right"

Outputs (at least one must be specified):
    --output-npz    saves motor commands, shape (N, 6), values in [0, 1]
    --output-video  renders each frame as a Sapien hand visualisation

Usage:
    uv run demos/retarget_npz.py --input landmarks.npz --output-npz motors.npz
    uv run demos/retarget_npz.py --input landmarks.npz --output-video hand.mp4 --fps 30
    uv run demos/retarget_npz.py --input landmarks.npz --output-npz motors.npz --output-video hand.mp4
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import numpy as np

from _utils import SapienHandRenderer, VideoWriter, draw_motor_bars
from brainco_retargeting import BrainCoRetargeter


def parse_args():
    p = argparse.ArgumentParser(description="Retarget landmarks NPZ to motor commands / video")
    p.add_argument("--input", type=str, required=True, help="Input .npz with landmarks + side")
    p.add_argument("--output-npz", type=str, default=None, help="Output .npz for motor commands")
    p.add_argument("--output-video", type=str, default=None,
                   help="Output video path showing Sapien hand render")
    p.add_argument("--fps", type=float, default=30.0, help="Output video frame rate")
    p.add_argument("--width", type=int, default=640, help="Output video frame width")
    p.add_argument("--height", type=int, default=480, help="Output video frame height")
    args = p.parse_args()
    if not args.output_npz and not args.output_video:
        p.error("At least one of --output-npz or --output-video must be specified")
    return args


def main():
    args = parse_args()

    if not Path(args.input).exists():
        sys.exit(f"Input file not found: {args.input}")

    data = np.load(args.input, allow_pickle=True)
    landmarks = data["landmarks"]   # (N, 25, 3)
    side = str(data["side"])

    if landmarks.ndim != 3 or landmarks.shape[1:] != (25, 3):
        sys.exit(f"Expected landmarks shape (N, 25, 3), got {landmarks.shape}")

    if side not in ("left", "right"):
        sys.exit(f"Expected side 'left' or 'right', got '{side}'")

    n_frames = len(landmarks)
    print(f"Input: {args.input}  ({n_frames} frames, side={side})")

    retargeter = BrainCoRetargeter()
    retarget_fn = retargeter.retarget_left if side == "left" else retargeter.retarget_right

    renderer: SapienHandRenderer | None = None
    video_writer: VideoWriter | None = None

    if args.output_video:
        renderer = SapienHandRenderer(side, args.width, args.height)
        out_path = Path(args.output_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        video_writer = VideoWriter(str(out_path), args.fps, args.width, args.height)

    all_motors: list[np.ndarray] = []

    for i, lm in enumerate(landmarks):
        motors = retarget_fn(lm)
        all_motors.append(motors)

        if renderer is not None and video_writer is not None:
            frame = renderer.render(motors)
            frame = draw_motor_bars(frame, motors)
            video_writer.write(frame)

        if (i + 1) % 100 == 0 or (i + 1) == n_frames:
            print(f"  {i + 1}/{n_frames} frames")

    if video_writer is not None:
        video_writer.release()
        print(f"Video saved → {args.output_video}")

    if args.output_npz:
        motors_arr = np.stack(all_motors)
        out_path = Path(args.output_npz)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out_path), motors=motors_arr)
        print(f"Motors saved → {args.output_npz}  shape={motors_arr.shape}")


if __name__ == "__main__":
    main()
