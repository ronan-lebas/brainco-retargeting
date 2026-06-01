# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "mediapipe==0.10.21",
#   "sapien>=3.0.0",
#   "opencv-python>=4.8",
#   "numpy>=1.24",
#   "pyyaml",
#   "dex-retargeting @ git+https://github.com/dexsuite/dex-retargeting",
#   "torch",
# ]
#
# [tool.uv]
# override-dependencies = ["numpy>=1.24,<2"]
#
# [[tool.uv.index]]
# name = "pytorch-cpu"
# url = "https://download.pytorch.org/whl/cpu"
# explicit = true
#
# [tool.uv.sources]
# torch = { index = "pytorch-cpu" }
# ///
"""Real-time hand retargeting demo using a webcam.

Opens a camera, runs MediaPipe Hands on each frame, converts the 21-pt
MediaPipe landmarks to the 25-pt XR format, runs BrainCo retargeting, and
displays a side-by-side window:
    Left panel  – camera feed with hand skeleton overlay
    Right panel – Sapien render of the BrainCo hand model

Usage:
    uv run demos/live_camera.py
    uv run demos/live_camera.py --hand right --camera-id 1 --output-npz /tmp/session.npz
"""

import argparse
import os
import sys
from pathlib import Path

# Must be set before cv2/Qt initialises to prevent black windows on Linux+NVIDIA
os.environ.setdefault("QT_X11_NO_MITSHM", "1")

sys.path.insert(0, str(Path(__file__).parent))

import cv2
import mediapipe as mp
import numpy as np

from _utils import SapienHandRenderer, draw_hand_skeleton_mp21, draw_motor_bars, mp21_to_xr25
from brainco_retargeting import BrainCoRetargeter


def parse_args():
    p = argparse.ArgumentParser(description="Live camera retargeting demo")
    p.add_argument("--camera-id", type=int, default=0, help="OpenCV camera device index")
    p.add_argument(
        "--hand",
        choices=["left", "right", "auto"],
        default="auto",
        help="Which hand to retarget. 'auto' uses MediaPipe's handedness detection.",
    )
    p.add_argument(
        "--output-npz",
        type=str,
        default=None,
        help="If set, save accumulated (25,3) landmarks to this .npz on exit.",
    )
    p.add_argument("--width", type=int, default=640, help="Camera capture width")
    p.add_argument("--height", type=int, default=480, help="Camera capture height")
    return p.parse_args()


def main():
    args = parse_args()

    panel_w, panel_h = 640, 480

    # Create the display window FIRST to lock in Qt's OpenGL context before any
    # EGL initialisation (MediaPipe and Sapien both trigger EGL on NVIDIA GPUs,
    # which corrupts the Qt GL widget if it hasn't been created yet).
    placeholder = np.zeros((panel_h, panel_w * 2, 3), dtype=np.uint8)
    cv2.putText(placeholder, "Initializing...", (panel_w // 2 + 160, panel_h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
    cv2.imshow("BrainCo Retargeting - press q to quit", placeholder)
    cv2.waitKey(1)

    print("Loading retargeter...")
    retargeter = BrainCoRetargeter()

    print("Loading MediaPipe...")
    mp_hands_mod = mp.solutions.hands
    hands = mp_hands_mod.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    active_side = args.hand if args.hand != "auto" else "right"
    print(f"Loading Sapien renderer ({active_side} hand)...")
    renderer = SapienHandRenderer(active_side, panel_w, panel_h)

    # Open camera after all EGL/GPU init has settled
    backend = cv2.CAP_V4L2 if sys.platform == "linux" else cv2.CAP_ANY
    cap = cv2.VideoCapture(args.camera_id, backend)
    if not cap.isOpened():
        sys.exit(f"Cannot open camera {args.camera_id}")
    for _ in range(15):
        cap.read()

    accumulated_landmarks: list[np.ndarray] = []
    motors = np.zeros(6)

    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        cam_panel = cv2.resize(frame, (panel_w, panel_h))

        if results.multi_hand_world_landmarks and results.multi_handedness:
            wl = results.multi_hand_world_landmarks[0]
            mp21 = np.array([[lm.x, lm.y, lm.z] for lm in wl.landmark], dtype=np.float64)
            xr25 = mp21_to_xr25(mp21)

            mp_side_raw = results.multi_handedness[0].classification[0].label
            mp_side = "right" if mp_side_raw == "Left" else "left"
            if args.hand != "auto":
                mp_side = args.hand

            if active_side != mp_side:
                active_side = mp_side
                print(f"Side changed to {active_side}, reloading Sapien renderer...")
                renderer = SapienHandRenderer(active_side, panel_w, panel_h)

            retarget_fn = retargeter.retarget_left if active_side == "left" else retargeter.retarget_right
            motors = retarget_fn(xr25)

            if args.output_npz:
                accumulated_landmarks.append(xr25.copy())

            if results.multi_hand_landmarks:
                draw_hand_skeleton_mp21(cam_panel, results.multi_hand_landmarks[0])

        robot_panel = renderer.render(motors)
        robot_panel = draw_motor_bars(robot_panel, motors)

        combined = np.hstack([cam_panel, robot_panel])
        cv2.putText(combined, "Camera", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(combined, "BrainCo Hand", (panel_w + 10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(combined, active_side.upper(), (panel_w + 10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 100), 2)

        cv2.imshow("BrainCo Retargeting - press q to quit", combined)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()

    if args.output_npz and accumulated_landmarks:
        out = Path(args.output_npz)
        out.parent.mkdir(parents=True, exist_ok=True)
        arr = np.stack(accumulated_landmarks)
        np.savez(str(out), landmarks=arr, side=active_side or "unknown")
        print(f"Saved {len(accumulated_landmarks)} frames of landmarks to {out}")


if __name__ == "__main__":
    main()
