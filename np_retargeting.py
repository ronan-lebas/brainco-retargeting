"""
BrainCo Revo2 hand retargeting — pure numpy, no dependencies.

Usage:
    from np_retargeting import retarget

    # landmarks: MediaPipe landmark list (21 items with .x .y .z)
    #         OR numpy array of shape (21, 3)
    angles = retarget(landmarks)           # dict, 6 controllable joints, radians
    angles = retarget(landmarks, 'left')   # left hand
"""

import numpy as np

# Joint limits (radians) from brainco_hand/brainco_{side}.urdf
_JOINT_LIMITS = {
    'thumb_metacarpal': (0.0,    1.5184),
    'thumb_proximal':   (0.0,    1.0472),
    'index_proximal':   (0.0,    1.4661),
    'middle_proximal':  (0.0,    1.4661),
    'ring_proximal':    (0.0,    1.4661),
    'pinky_proximal':   (0.0,    1.4661),
}

# MediaPipe landmark indices
_W   = 0
_T1, _T2, _T3, _T4           = 1,  2,  3,  4   # thumb  CMC MCP IP TIP
_I1, _I2, _I3, _I4           = 5,  6,  7,  8   # index  MCP PIP DIP TIP
_M1, _M2, _M3, _M4           = 9,  10, 11, 12  # middle MCP PIP DIP TIP
_R1, _R2, _R3, _R4           = 13, 14, 15, 16  # ring   MCP PIP DIP TIP
_P1, _P2, _P3, _P4           = 17, 18, 19, 20  # pinky  MCP PIP DIP TIP


def _pt(lm, i):
    """Extract point i as a (3,) float64 array."""
    if isinstance(lm, np.ndarray):
        return lm[i].astype(np.float64)
    return np.array([lm[i].x, lm[i].y, lm[i].z], dtype=np.float64)


def _angle(a, b, c):
    """Angle at vertex b formed by a-b-c, in radians."""
    v1, v2 = a - b, c - b
    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
    return np.arccos(np.clip(cos, -1.0, 1.0))


def _bend(a, b, c):
    """Flexion/bend angle at b: 0 = straight, positive = curled."""
    return np.pi - _angle(a, b, c)


def retarget(landmarks, hand_side: str = 'right') -> dict:
    """
    Map 21 MediaPipe hand landmarks to 6 Revo2 controllable joint angles.

    Args:
        landmarks: MediaPipe landmark list OR numpy array (21, 3)
        hand_side:  'right' or 'left'

    Returns:
        dict  {joint_name: angle_radians}  — 6 independently actuated joints.
        Distal joints are mimic joints driven by the robot firmware:
            distal_angle = 1.155 * proximal_angle  (fingers)
            distal_angle = 1.0   * thumb_proximal  (thumb)
    """
    p = lambda i: _pt(landmarks, i)  # noqa: E731

    # --- thumb ---
    meta = _angle(p(_W), p(_T1), p(_T2)) - np.pi / 2   # CMC abduction
    meta = float(np.clip(meta, 0.0, np.pi / 2))
    prox_t = float(_bend(p(_T1), p(_T2), p(_T3)))
    # (thumb distal is mimic: 1.0 * thumb_proximal)

    # --- four fingers: proximal only (distal mimics proximal * 1.155) ---
    prox_i = float(_bend(p(_I1), p(_I2), p(_I3)))
    prox_m = float(_bend(p(_M1), p(_M2), p(_M3)))
    prox_r = float(_bend(p(_R1), p(_R2), p(_R3)))
    prox_p = float(_bend(p(_P1), p(_P2), p(_P3)))

    s = hand_side
    raw = {
        f'{s}_thumb_metacarpal_joint': meta,
        f'{s}_thumb_proximal_joint':   prox_t,
        f'{s}_index_proximal_joint':   prox_i,
        f'{s}_middle_proximal_joint':  prox_m,
        f'{s}_ring_proximal_joint':    prox_r,
        f'{s}_pinky_proximal_joint':   prox_p,
    }

    keys = [
        'thumb_metacarpal', 'thumb_proximal',
        'index_proximal', 'middle_proximal', 'ring_proximal', 'pinky_proximal',
    ]
    return {
        f'{s}_{k}_joint': float(np.clip(raw[f'{s}_{k}_joint'], *_JOINT_LIMITS[k]))
        for k in keys
    }
