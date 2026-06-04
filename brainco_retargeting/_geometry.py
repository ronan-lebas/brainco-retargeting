"""Pure-geometry helpers for MediaPipe hand post-processing (numpy only).

These solve the two problems that make raw MediaPipe output unusable for
retargeting:

1. **Handedness.** MediaPipe's reported ``"Left"``/``"Right"`` label assumes a
   *mirrored* (selfie) image and is unreliable for a non-mirrored feed (phone
   back camera, raw video). Instead we read the hand's chirality straight from
   the 3D landmark geometry via a rotation-invariant triple product, which is
   independent of how the hand happens to be oriented in front of the camera.

2. **Orientation.** MediaPipe world landmarks live in the *camera* frame, but
   the dex-retargeting optimizer matches 3D bone vectors against the robot's
   ``base_link`` frame. They only line up when the palm happens to face the
   camera. Expressing the landmarks in a hand-local frame (built from the wrist
   and the knuckles) and rotating that into the robot frame removes the
   dependence on hand orientation entirely — see ``BrainCoRetargeter``.

See README "Handedness & Camera Convention" for the user-facing summary.
"""

import numpy as np

# Is the image handed to MediaPipe horizontally mirrored (front/selfie camera)?
#   - Phone BACK camera / raw video  -> False  (apparent hand == physical hand)
#   - Front/selfie camera            -> True   (demos flip the frame so that the
#                                               on-screen hand matches reality)
# If "left"/"right" come out swapped for your setup, flip this constant.
MIRRORED_INPUT = False

# MediaPipe-21 vs XR-25 indices for the four points that define hand chirality
# and the hand-local frame (wrist, index knuckle, pinky knuckle, middle knuckle).
_KEY_IDX = {
    21: dict(wrist=0, index_mcp=5, pinky_mcp=17, middle_mcp=9, thumb=1),
    25: dict(wrist=0, index_mcp=6, pinky_mcp=21, middle_mcp=11, thumb=1),
}


def _key_points(landmarks: np.ndarray):
    """Return (wrist, index_mcp, pinky_mcp, middle_mcp, thumb) for a 21- or 25-pt set."""
    L = np.asarray(landmarks, dtype=np.float64)
    idx = _KEY_IDX.get(L.shape[0])
    if idx is None:
        raise ValueError(f"Expected 21 or 25 landmarks, got shape {L.shape}")
    return (L[idx["wrist"]], L[idx["index_mcp"]], L[idx["pinky_mcp"]],
            L[idx["middle_mcp"]], L[idx["thumb"]])


def detect_hand_side(landmarks: np.ndarray) -> str:
    """Determine whether landmarks describe a left or right hand from geometry.

    Uses the signed volume (scalar triple product) of the vectors wrist→index,
    wrist→pinky, wrist→thumb. This pseudoscalar flips sign between a left and a
    right hand and is invariant to rotation, so it does not care how the hand is
    posed. Calibrated against the BrainCo URDF: ``> 0`` is a right hand.

    Accepts MediaPipe-21 or XR-25 landmarks of shape (N, 3). Returns
    ``"left"`` or ``"right"``.
    """
    wrist, index_mcp, pinky_mcp, _middle, thumb = _key_points(landmarks)
    triple = np.dot(np.cross(index_mcp - wrist, pinky_mcp - wrist), thumb - wrist)
    return "right" if triple > 0 else "left"


def hand_local_basis(landmarks: np.ndarray) -> np.ndarray:
    """Build an orthonormal hand-local frame from wrist + knuckles.

    Columns are [across, finger, normal]:
        finger  – wrist → middle knuckle (along the hand)
        across  – index knuckle → pinky knuckle (across the palm)
        normal  – palm normal (across × finger)

    Returns a (3, 3) matrix mapping hand-local coordinates to the input frame.
    Built only from the wrist and the knuckles, so it is stable regardless of
    how much the fingers are curled.
    """
    wrist, index_mcp, pinky_mcp, middle_mcp, _thumb = _key_points(landmarks)
    finger = middle_mcp - wrist
    finger = finger / (np.linalg.norm(finger) + 1e-9)
    across = pinky_mcp - index_mcp
    normal = np.cross(finger, across)
    normal = normal / (np.linalg.norm(normal) + 1e-9)
    across = np.cross(normal, finger)
    across = across / (np.linalg.norm(across) + 1e-9)
    return np.column_stack([across, finger, normal])
