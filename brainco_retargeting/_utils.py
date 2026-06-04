"""Shared utilities for BrainCo retargeting demos."""

import cv2
import numpy as np
from pathlib import Path

from brainco_retargeting._geometry import MIRRORED_INPUT, detect_hand_side

_MOTOR_RANGES = [
    (0.0, 1.52),  # thumb_metacarpal
    (0.0, 1.05),  # thumb_proximal
    (0.0, 1.47),  # index_proximal
    (0.0, 1.47),  # middle_proximal
    (0.0, 1.47),  # ring_proximal
    (0.0, 1.47),  # pinky_proximal
]

_MOTOR_LABELS = [
    "Thumb Meta",
    "Thumb Prox",
    "Index",
    "Middle",
    "Ring",
    "Pinky",
]

_HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),              # thumb
    (0, 5), (5, 6), (6, 7), (7, 8), (8, 9),       # index  (25-pt: 0→5 metacarpal, 5-9)
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14),  # middle
    (0, 15), (15, 16), (16, 17), (17, 18), (18, 19),  # ring
    (0, 20), (20, 21), (21, 22), (22, 23), (23, 24),  # pinky
    (5, 10), (10, 15), (15, 20),                   # palm knuckle bar
]

# MediaPipe's 21-pt connections (for drawing on camera feed directly)
_MP_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17),
]


def select_hand(results, want: str = "auto"):
    """Pick which detected hand to use, with its true (geometry-based) side.

    The side is read from 3D
    landmark geometry (see ``detect_hand_side``) rather than MediaPipe's
    unreliable handedness label.

    Args:
        results: a MediaPipe ``Hands`` result (``multi_hand_landmarks`` etc.).
        want:    ``"left"`` / ``"right"`` (the physical hand to track) or
                 ``"auto"`` (use whichever hand is detected).

    Returns:
        ``(index, side)`` — the index into ``results.multi_hand_landmarks`` and
        the detected side (``"left"``/``"right"``) of that hand — or ``None`` if
        no hand was detected. When ``want`` is a specific hand but it is not on
        screen, falls back to the first detected hand (so something is always
        shown) while still reporting that hand's true side.
    """
    normalized = results.multi_hand_landmarks
    if not normalized:
        return None
    # Prefer metric world landmarks for the chirality test; fall back to the
    # normalized landmarks if (rarely) world landmarks are unavailable.
    source = results.multi_hand_world_landmarks or normalized
    sides = [
        detect_hand_side(np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]))
        for hand in source
    ]
    if want in ("left", "right"):
        for i, side in enumerate(sides):
            if side == want:
                return i, side
    return 0, sides[0]


def mp21_to_xr25(mp_landmarks: np.ndarray) -> np.ndarray:
    """Convert MediaPipe 21-point world landmarks to 25-point XR format.

    The 4 extra points are non-thumb metacarpal joints, synthesized by
    placing them 1/3 of the way from the wrist to each finger's MCP joint.
    """
    assert mp_landmarks.shape == (21, 3), f"Expected (21,3), got {mp_landmarks.shape}"
    xr = np.zeros((25, 3), dtype=np.float64)
    wrist = mp_landmarks[0]
    xr[0] = wrist
    xr[1:5] = mp_landmarks[1:5]                                      # thumb
    xr[5] = wrist + 0.33 * (mp_landmarks[5] - wrist)                 # index metacarpal
    xr[6:10] = mp_landmarks[5:9]                                      # index MCP..tip
    xr[10] = wrist + 0.33 * (mp_landmarks[9] - wrist)                # middle metacarpal
    xr[11:15] = mp_landmarks[9:13]                                    # middle MCP..tip
    xr[15] = wrist + 0.33 * (mp_landmarks[13] - wrist)               # ring metacarpal
    xr[16:20] = mp_landmarks[13:17]                                   # ring MCP..tip
    xr[20] = wrist + 0.33 * (mp_landmarks[17] - wrist)               # pinky metacarpal
    xr[21:25] = mp_landmarks[17:21]                                   # pinky MCP..tip
    return xr


def draw_hand_skeleton_mp21(
    image: np.ndarray,
    mp_result_landmarks,  # mediapipe NormalizedLandmarkList
    color: tuple = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Draw MediaPipe 21-point skeleton onto an image in-place."""
    h, w = image.shape[:2]
    pts = np.array([[int(lm.x * w), int(lm.y * h)] for lm in mp_result_landmarks.landmark])
    for a, b in _MP_CONNECTIONS:
        cv2.line(image, tuple(pts[a]), tuple(pts[b]), color, thickness)
    for pt in pts:
        cv2.circle(image, tuple(pt), 4, (255, 255, 255), -1)
        cv2.circle(image, tuple(pt), 4, color, 1)
    return image


def draw_motor_bars(image: np.ndarray, motors: np.ndarray, x0: int = 10, y0: int = None) -> np.ndarray:
    """Overlay 6 motor value bars (0–1) on the image."""
    h, w = image.shape[:2]
    if y0 is None:
        y0 = h - 120
    bar_h = 12
    bar_max_w = 180
    label_w = 80
    pad = 4
    bg = image.copy()
    for i, (val, label) in enumerate(zip(motors, _MOTOR_LABELS)):
        y = y0 + i * (bar_h + pad)
        # Background bar
        cv2.rectangle(bg, (x0 + label_w, y), (x0 + label_w + bar_max_w, y + bar_h), (50, 50, 50), -1)
        # Value bar
        fill_w = int(np.clip(val, 0, 1) * bar_max_w)
        cv2.rectangle(bg, (x0 + label_w, y), (x0 + label_w + fill_w, y + bar_h), (0, 200, 100), -1)
        # Label
        cv2.putText(bg, f"{label}: {val:.2f}", (x0, y + bar_h - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (220, 220, 220), 1, cv2.LINE_AA)
    return cv2.addWeighted(bg, 0.85, image, 0.15, 0)


class SapienHandRenderer:
    """Offscreen Sapien renderer for the BrainCo hand URDF."""

    # Active joint order from get_active_joints() (verified from URDF):
    #   0: thumb_metacarpal  1: index_proximal   2: middle_proximal
    #   3: ring_proximal     4: pinky_proximal   5: thumb_proximal
    #   6: index_distal      7: middle_distal    8: ring_distal
    #   9: pinky_distal     10: thumb_distal
    _SIDE_API_ORDER = {
        # motor_idx → active_joint_idx
        "left": {
            "left_thumb_metacarpal_joint": (0, 0),   # motor_idx, joint_idx
            "left_thumb_proximal_joint":   (1, 5),
            "left_index_proximal_joint":   (2, 1),
            "left_middle_proximal_joint":  (3, 2),
            "left_ring_proximal_joint":    (4, 3),
            "left_pinky_proximal_joint":   (5, 4),
        },
        "right": {
            "right_thumb_metacarpal_joint": (0, 0),
            "right_thumb_proximal_joint":   (1, 5),
            "right_index_proximal_joint":   (2, 1),
            "right_middle_proximal_joint":  (3, 2),
            "right_ring_proximal_joint":    (4, 3),
            "right_pinky_proximal_joint":   (5, 4),
        },
    }
    # passive distal joint: (joint_idx_in_active, coupled_motor_idx)
    _DISTAL_COUPLING = {
        "left":  [(6, 2), (7, 3), (8, 4), (9, 5), (10, 1)],  # idx,mid,ring,pinky,thumb distal
        "right": [(6, 2), (7, 3), (8, 4), (9, 5), (10, 1)],
    }

    def __init__(self, side: str, width: int = 640, height: int = 480):
        import sapien
        import sapien.render as sr
        sr.set_log_level("error")

        self._sapien = sapien
        self.side = side
        self._width = width
        self._height = height

        self._scene = sapien.Scene()
        self._scene.set_ambient_light([0.10, 0.10, 0.10])
        self._scene.add_directional_light([-1, 0.2, -0.5], [1.3, 1.3, 1.25])   # key light
        self._scene.add_directional_light([-0.4, -0.9, 0.4], [0.35, 0.35, 0.40])  # fill light

        assets_dir = Path(__file__).parent / "assets"
        urdf_path = assets_dir / "brainco_hand" / f"brainco_{side}.urdf"

        loader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self._robot = loader.load(str(urdf_path))

        # Hide coordinate-axis visual on base_link
        for link in self._robot.get_links():
            if link.name == "base_link":
                rb = link.entity.find_component_by_type(sr.RenderBodyComponent)
                if rb:
                    rb.disable()

        # Build joint-index lookup from active_joints list
        active_joints = self._robot.get_active_joints()
        jname_to_idx = {j.name: i for i, j in enumerate(active_joints)}

        api_map = self._SIDE_API_ORDER[side]
        self._motor_to_joint = {}   # motor_idx → joint_idx_in_qpos
        for jname, (motor_idx, _) in api_map.items():
            if jname in jname_to_idx:
                self._motor_to_joint[motor_idx] = jname_to_idx[jname]

        # Distal joints are mimic joints (from URDF):
        #   thumb_distal  = 1.000 × thumb_proximal
        #   *_distal      = 1.155 × *_proximal   (index / middle / ring / pinky)
        self._distal = []  # list of (joint_idx_in_qpos, motor_idx, multiplier)
        pfx = side
        distal_names = [
            (f"{pfx}_thumb_distal_joint",  1, 1.000),
            (f"{pfx}_index_distal_joint",  2, 1.155),
            (f"{pfx}_middle_distal_joint", 3, 1.155),
            (f"{pfx}_ring_distal_joint",   4, 1.155),
            (f"{pfx}_pinky_distal_joint",  5, 1.155),
        ]
        for jname, motor_idx, multiplier in distal_names:
            if jname in jname_to_idx:
                self._distal.append((jname_to_idx[jname], motor_idx, multiplier))

        # Camera: look along +X (SAPIEN principal axis), centred on hand
        cam_actor = self._scene.create_actor_builder().build_kinematic()
        cam_actor.set_pose(sapien.Pose(p=[-0.35, -0.065, 0.025]))
        self._cam = self._scene.add_mounted_camera(
            "renderer_cam", cam_actor, sapien.Pose(), width, height, 1.0, 0.01, 10.0
        )

    def render(self, motors: np.ndarray) -> np.ndarray:
        """Render the hand with given motor values (6,) in [0, 1].

        Returns a BGR image of shape (height, width, 3).
        """
        n = self._robot.dof
        qpos = np.zeros(n)

        # Denormalise: normalized = 1 - (max - raw) / (max - min)  →  raw = max - (1 - normalized) * (max - min)
        raw = np.array([
            lo + m * (hi - lo) for m, (lo, hi) in zip(motors, _MOTOR_RANGES)
        ])

        for motor_idx, joint_idx in self._motor_to_joint.items():
            qpos[joint_idx] = raw[motor_idx]

        for joint_idx, motor_idx, multiplier in self._distal:
            qpos[joint_idx] = multiplier * raw[motor_idx]

        self._robot.set_qpos(qpos)
        self._scene.update_render()
        self._cam.take_picture()
        color = self._cam.get_picture("Color")
        rgb = (np.clip(color[:, :, :3], 0, 1) * 255).astype(np.uint8)
        return rgb[:, :, ::-1]  # RGB → BGR


class VideoWriter:
    """Thin wrapper around cv2.VideoWriter."""

    def __init__(self, path: str, fps: float, width: int, height: int):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        if not self._writer.isOpened():
            raise RuntimeError(f"Cannot open VideoWriter for {path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()
