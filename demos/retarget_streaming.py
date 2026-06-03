"""Real-time hand retargeting as an importable module.

Runs MediaPipe hand detection + BrainCo retargeting in a background daemon
thread, updating a cv2 GUI window continuously.  The latest 6 motor values
(normalised to [0, 1]) are always available via `get_joints()`.

Quickstart
----------
    import retarget_streaming

    retarget_streaming.start()          # opens GUI window, starts camera

    while True:
        joints = retarget_streaming.get_joints()   # (6,) float array, [0, 1]
        ...

    retarget_streaming.stop()           # or just let the process exit

Class API (for explicit lifecycle management)
---------------------------------------------
    with retarget_streaming.VideoRetargeter(hand="right") as vr:
        joints = vr.joints
"""

import os
import threading

# Must be set before cv2/Qt initialises to avoid black windows on Linux+NVIDIA
os.environ.setdefault("QT_X11_NO_MITSHM", "1")

import sys
import cv2
import mediapipe as mp
import numpy as np

from brainco_retargeting import BrainCoRetargeter
from brainco_retargeting._utils import (
    SapienHandRenderer,
    _MOTOR_RANGES,
    draw_hand_skeleton_mp21,
    draw_motor_bars,
    mp21_to_xr25,
)

from brainco_retargeting import np_retargeting as _np_retargeting

_NP_JOINT_ORDER = [
    'thumb_metacarpal', 'thumb_proximal',
    'index_proximal', 'middle_proximal', 'ring_proximal', 'pinky_proximal',
]


class VideoRetargeter:
    """Camera + MediaPipe + BrainCo retargeting running in a background thread.

    Parameters
    ----------
    hand:        "left" | "right" | "auto"  (default "auto")
    camera_id:   OpenCV device index (default 0)
    np_retarget: use pure-numpy retargeter instead of dex-retargeting
    width/height: capture resolution (default 640×480)
    """

    def __init__(
        self,
        hand: str = "auto",
        camera_id: int = 0,
        np_retarget: bool = False,
        width: int = 640,
        height: int = 480,
    ):
        self._hand = hand
        self._camera_id = camera_id
        self._np_retarget = np_retarget
        self._width = width
        self._height = height

        self._joints = np.zeros(6, dtype=np.float64)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> "VideoRetargeter":
        """Start the background camera+GUI thread and return self."""
        if self._thread is not None and self._thread.is_alive():
            return self
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="video-retarget")
        self._thread.start()
        return self

    def stop(self) -> None:
        """Signal the background thread to stop and wait for it."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
        self._thread = None

    @property
    def joints(self) -> np.ndarray:
        """Latest retargeted motor values, shape (6,), values in [0, 1]."""
        with self._lock:
            return self._joints.copy()

    def __enter__(self) -> "VideoRetargeter":
        return self.start()

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Background loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        panel_w, panel_h = self._width, self._height

        # Create window first to lock in the Qt/GL context before EGL inits
        placeholder = np.zeros((panel_h, panel_w * 2, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Initializing...", (panel_w // 2 + 160, panel_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (200, 200, 200), 2)
        cv2.imshow("BrainCo Retargeting - press q to quit", placeholder)
        cv2.waitKey(1)

        retargeter = None if self._np_retarget else BrainCoRetargeter()

        mp_hands_mod = mp.solutions.hands
        hands = mp_hands_mod.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        active_side = self._hand if self._hand != "auto" else "right"
        renderer = SapienHandRenderer(active_side, panel_w, panel_h)

        backend = cv2.CAP_V4L2 if sys.platform == "linux" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._camera_id, backend)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open camera {self._camera_id}")
        for _ in range(15):
            cap.read()

        motors = np.zeros(6)

        try:
            while not self._stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb.flags.writeable = False
                results = hands.process(frame_rgb)
                frame_rgb.flags.writeable = True

                cam_panel = cv2.resize(frame, (panel_w, panel_h))

                if results.multi_hand_landmarks and results.multi_handedness:
                    target_idx = None
                    if self._hand != "auto":
                        mp_label_wanted = "Left" if self._hand == "right" else "Right"
                        for hidx, handedness in enumerate(results.multi_handedness):
                            if handedness.classification[0].label == mp_label_wanted:
                                target_idx = hidx
                                break
                    if target_idx is None:
                        target_idx = 0

                    if self._np_retarget:
                        src = results.multi_hand_landmarks[target_idx]
                    else:
                        if not results.multi_hand_world_landmarks:
                            continue
                        src = results.multi_hand_world_landmarks[target_idx]

                    mp21 = np.array([[lm.x, lm.y, lm.z] for lm in src.landmark], dtype=np.float64)

                    mp_side_raw = results.multi_handedness[target_idx].classification[0].label
                    mp_side = "right" if mp_side_raw == "Left" else "left"
                    if self._hand != "auto":
                        mp_side = self._hand

                    if active_side != mp_side:
                        active_side = mp_side
                        renderer = SapienHandRenderer(active_side, panel_w, panel_h)

                    if self._np_retarget:
                        angles = _np_retargeting.retarget(mp21, active_side)
                        raw = np.array([angles[f'{active_side}_{k}_joint'] for k in _NP_JOINT_ORDER])
                        motors = np.array([(r - lo) / (hi - lo) for r, (lo, hi) in zip(raw, _MOTOR_RANGES)])
                    else:
                        xr25 = mp21_to_xr25(mp21)
                        retarget_fn = retargeter.retarget_left if active_side == "left" else retargeter.retarget_right
                        motors = retarget_fn(xr25)

                    with self._lock:
                        self._joints[:] = motors

                    draw_hand_skeleton_mp21(cam_panel, results.multi_hand_landmarks[target_idx])

                robot_panel = renderer.render(motors)
                robot_panel = draw_motor_bars(robot_panel, motors)

                combined = np.hstack([cam_panel, robot_panel])
                cv2.putText(combined, "Camera", (10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(combined, "BrainCo Hand", (panel_w + 10, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                cv2.putText(combined, active_side.upper(), (panel_w + 10, 54),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 220, 100), 2)

                cv2.imshow("BrainCo Retargeting - press q to quit", combined)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    self._stop_event.set()
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            hands.close()


# ---------------------------------------------------------------------------
# Module-level singleton API  (import video_retarget; video_retarget.start())
# ---------------------------------------------------------------------------

_instance: VideoRetargeter | None = None


def start(
    hand: str = "auto",
    camera_id: int = 0,
    np_retarget: bool = False,
    width: int = 640,
    height: int = 480,
) -> VideoRetargeter:
    """Start the global VideoRetargeter instance and return it."""
    global _instance
    _instance = VideoRetargeter(
        hand=hand, camera_id=camera_id, np_retarget=np_retarget,
        width=width, height=height,
    )
    _instance.start()
    return _instance


def get_joints() -> np.ndarray:
    """Return the latest retargeted joint values from the global instance.

    Returns zeros (6,) until a hand is detected.
    Raises RuntimeError if start() has not been called.
    """
    if _instance is None:
        raise RuntimeError("Call retarget_streaming.start() before get_joints()")
    return _instance.joints


def stop() -> None:
    """Stop the global VideoRetargeter instance."""
    global _instance
    if _instance is not None:
        _instance.stop()
        _instance = None
