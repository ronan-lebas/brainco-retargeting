from pathlib import Path

import numpy as np
import yaml
from dex_retargeting.retargeting_config import RetargetingConfig

from brainco_retargeting._geometry import hand_local_basis

_ASSETS_DIR = Path(__file__).parent / "assets"

_MOTOR_RANGES = [
    (0.0, 1.52),  # thumb metacarpal
    (0.0, 1.05),  # thumb proximal
    (0.0, 1.47),  # index proximal
    (0.0, 1.47),  # middle proximal
    (0.0, 1.47),  # ring proximal
    (0.0, 1.47),  # pinky proximal
]

_LEFT_API_JOINTS = [
    "left_thumb_metacarpal_joint",
    "left_thumb_proximal_joint",
    "left_index_proximal_joint",
    "left_middle_proximal_joint",
    "left_ring_proximal_joint",
    "left_pinky_proximal_joint",
]

_RIGHT_API_JOINTS = [
    "right_thumb_metacarpal_joint",
    "right_thumb_proximal_joint",
    "right_index_proximal_joint",
    "right_middle_proximal_joint",
    "right_ring_proximal_joint",
    "right_pinky_proximal_joint",
]


def _prepare_cfg(cfg: dict) -> dict:
    """Strip xr_teleoperate-specific keys and promote the DexPilot index array."""
    cfg = dict(cfg)
    # The YAML stores both DexPilot and vector indices under custom keys;
    # RetargetingConfig expects a single 'target_link_human_indices' field.
    if "target_link_human_indices_dexpilot" in cfg:
        cfg["target_link_human_indices"] = cfg.pop("target_link_human_indices_dexpilot")
    cfg.pop("target_link_human_indices_vector", None)
    return cfg


def _normalize(val: float, min_val: float, max_val: float) -> float:
    return 1.0 - float(np.clip((max_val - val) / (max_val - min_val), 0.0, 1.0))


class BrainCoRetargeter:
    """
    Retargets 25-point XR/MediaPipe hand landmarks to 6 BrainCo motor commands.

    Input:  np.ndarray of shape (25, 3) — hand landmarks in any consistent coordinate frame.
    Output: np.ndarray of shape (6,)   — normalized motor angles in [0.0, 1.0],
            where 0.0 = fully open and 1.0 = fully closed.

    Motor order: thumb_metacarpal, thumb_proximal, index, middle, ring, pinky.
    """

    def __init__(self) -> None:
        RetargetingConfig.set_default_urdf_dir(str(_ASSETS_DIR))

        config_path = _ASSETS_DIR / "brainco_hand" / "brainco.yml"
        with config_path.open("r") as f:
            cfg = yaml.safe_load(f)

        left_cfg = RetargetingConfig.from_dict(_prepare_cfg(cfg["left"]))
        right_cfg = RetargetingConfig.from_dict(_prepare_cfg(cfg["right"]))
        self._left = left_cfg.build()
        self._right = right_cfg.build()

        self._left_hw_idx = [self._left.joint_names.index(n) for n in _LEFT_API_JOINTS]
        self._right_hw_idx = [self._right.joint_names.index(n) for n in _RIGHT_API_JOINTS]

        self._left_indices = self._left.optimizer.target_link_human_indices
        self._right_indices = self._right.optimizer.target_link_human_indices

        # Hand-local frame of each robot hand (open pose, in its base_link frame),
        # used to rotate camera-frame landmarks into the robot frame — see canonicalize().
        self._canon_basis = {
            "left": self._robot_hand_basis(self._left, "left"),
            "right": self._robot_hand_basis(self._right, "right"),
        }

    @staticmethod
    def _robot_hand_basis(retargeting, side: str) -> np.ndarray:
        """Hand-local basis of the robot hand at the open (qpos=0) pose."""
        robot = retargeting.optimizer.robot
        robot.compute_forward_kinematics(np.zeros(robot.dof))
        names = {n.lower(): n for n in robot.link_names}
        def pos(link: str) -> np.ndarray:
            return robot.get_link_pose(robot.get_link_index(names[link.lower()]))[:3, 3]
        # XR-25 layout: 0 wrist, 6 index_mcp, 11 middle_mcp, 21 pinky_mcp.
        ref = np.zeros((25, 3))
        ref[0] = pos("base_link")
        ref[6] = pos(f"{side}_index_proximal_link")
        ref[11] = pos(f"{side}_middle_proximal_link")
        ref[21] = pos(f"{side}_pinky_proximal_link")
        return hand_local_basis(ref)

    def canonicalize(self, landmarks: np.ndarray, side: str) -> np.ndarray:
        """Rotate (25, 3) camera-frame landmarks into the robot base frame.

        MediaPipe world landmarks are expressed in the camera frame, so they
        only align with the robot's ``base_link`` frame when the palm faces the
        camera. This builds the hand-local frame from the input landmarks and
        rotates it onto the robot's hand-local frame, making retargeting
        independent of how the hand is oriented in front of the camera.

        The transform is a pure rotation (it preserves chirality), so a left
        hand stays left and a right hand stays right — pass the matching ``side``
        as returned by ``brainco_retargeting._geometry.detect_hand_side``.

        Input/output: (25, 3) XR landmarks, output has the wrist at the origin.
        """
        L = np.asarray(landmarks, dtype=np.float64)
        if L.shape != (25, 3):
            raise ValueError(f"Expected landmarks shape (25, 3), got {L.shape}")
        wrist = L[0]
        rotation = self._canon_basis[side] @ hand_local_basis(L).T
        return (L - wrist) @ rotation.T

    def retarget_left(self, landmarks: np.ndarray) -> np.ndarray:
        """landmarks: (25, 3) → motor angles: (6,) in [0, 1]"""
        return self._retarget(landmarks, self._left, self._left_indices, self._left_hw_idx)

    def retarget_right(self, landmarks: np.ndarray) -> np.ndarray:
        """landmarks: (25, 3) → motor angles: (6,) in [0, 1]"""
        return self._retarget(landmarks, self._right, self._right_indices, self._right_hw_idx)

    @staticmethod
    def _retarget(landmarks, retargeting, indices, hw_idx):
        landmarks = np.asarray(landmarks, dtype=np.float64)
        if landmarks.shape != (25, 3):
            raise ValueError(f"Expected landmarks shape (25, 3), got {landmarks.shape}")

        bone_vectors = landmarks[indices[1, :]] - landmarks[indices[0, :]]
        raw_angles = retargeting.retarget(bone_vectors)[hw_idx]

        normalized = np.array(
            [_normalize(raw_angles[i], *_MOTOR_RANGES[i]) for i in range(6)],
            dtype=np.float64,
        )
        return normalized
