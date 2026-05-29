# BrainCo Retargeting

Retargets 25-point XR/MediaPipe hand landmarks to 6 BrainCo prosthetic hand motor commands.

## Installation

Requires Python ≥ 3.9 and [uv](https://github.com/astral-sh/uv).

Two dependencies require special handling, both configured in `pyproject.toml` so they install automatically:

- **`dex-retargeting`** — pulled from [dexsuite/dex-retargeting](https://github.com/dexsuite/dex-retargeting) on GitHub (requires `git`)
- **`torch`** — pulled from the PyTorch CPU index (`https://download.pytorch.org/whl/cpu`)

```bash
uv pip install -e .
```

## Core API

```python
from brainco_retargeting import BrainCoRetargeter
import numpy as np

retargeter = BrainCoRetargeter()

# landmarks: np.ndarray of shape (25, 3) — XR hand landmarks in any consistent frame
motors_left  = retargeter.retarget_left(landmarks)   # → (6,) float64 in [0.0, 1.0]
motors_right = retargeter.retarget_right(landmarks)  # → (6,) float64 in [0.0, 1.0]
```

Motor values are normalised: `0.0 = fully open`, `1.0 = fully closed`.

**Motor order:**

| Index | Joint |
|-------|-------|
| 0 | thumb_metacarpal |
| 1 | thumb_proximal |
| 2 | index_proximal |
| 3 | middle_proximal |
| 4 | ring_proximal |
| 5 | pinky_proximal |

## Landmark Format (25 points)

The retargeter uses XR-style 25-point landmarks, which differ from the standard MediaPipe 21-point format by adding a metacarpal joint for each non-thumb finger:

| Index | Joint |
|-------|-------|
| 0 | Wrist |
| 1–4 | Thumb (CMC, MCP, IP, tip) |
| **5** | **Index metacarpal** |
| 6–9 | Index (MCP, PIP, DIP, tip) |
| **10** | **Middle metacarpal** |
| 11–14 | Middle (MCP, PIP, DIP, tip) |
| **15** | **Ring metacarpal** |
| 16–19 | Ring (MCP, PIP, DIP, tip) |
| **20** | **Pinky metacarpal** |
| 21–24 | Pinky (MCP, PIP, DIP, tip) |

---

## Demo Scripts

The `demos/` directory contains ready-to-run scripts covering all common usage patterns. All scripts use [uv](https://github.com/astral-sh/uv) with PEP 723 inline metadata — dependencies are installed automatically on first run.

Run from the **repo root**:

```bash
uv run demos/<script>.py [args]
```

### `demos/live_camera.py` — Real-time webcam demo

Opens a camera, tracks the hand with MediaPipe, and shows a side-by-side window:
- **Left** – camera feed with skeleton overlay
- **Right** – Sapien render of the BrainCo hand + motor value bars

```bash
uv run demos/live_camera.py
uv run demos/live_camera.py --hand right --camera-id 1
uv run demos/live_camera.py --hand left --output-npz session.npz   # save landmarks on quit (q)
```

| Flag | Default | Description |
|------|---------|-------------|
| `--camera-id` | `0` | OpenCV device index |
| `--hand` | `auto` | `left`, `right`, or `auto` (MediaPipe handedness) |
| `--output-npz` | — | Save accumulated landmarks on exit |

---

### `demos/record_video.py` — Record hand landmarks to NPZ

Extracts 25-pt hand landmarks from a video file or live webcam and saves them to `.npz`. No retargeting is performed — useful for capturing raw motion data.

```bash
uv run demos/record_video.py --input clip.mp4 --hand right --output-npz landmarks.npz
uv run demos/record_video.py --live --hand right --output-npz landmarks.npz --preview
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | — | Input video file (mutually exclusive with `--live`) |
| `--live` | — | Capture from webcam |
| `--camera-id` | `0` | Camera device index (with `--live`) |
| `--hand` | `auto` | `left`, `right`, or `auto` |
| `--output-npz` | required | Output `.npz` path |
| `--preview` | off | Show preview window |

**Output NPZ:**
```
landmarks  (N, 25, 3)  float64
side       str                   "left" or "right"
```

---

### `demos/retarget_npz.py` — Retarget from a landmarks NPZ

Takes a landmarks `.npz` and produces motor commands and/or a visualisation video.

```bash
uv run demos/retarget_npz.py --input landmarks.npz --output-npz motors.npz
uv run demos/retarget_npz.py --input landmarks.npz --output-video hand.mp4 --fps 30
uv run demos/retarget_npz.py --input landmarks.npz --output-npz motors.npz --output-video hand.mp4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Landmarks `.npz` |
| `--output-npz` | — | Save motor commands (at least one output required) |
| `--output-video` | — | Render Sapien hand video |
| `--fps` | `30` | Output video frame rate |

**Output NPZ:**
```
motors  (N, 6)  float64   values in [0, 1]
```

---

### `demos/retarget_video.py` — End-to-end retargeting from a video

Runs MediaPipe + retargeting on a video file in a single step.

```bash
uv run demos/retarget_video.py --input clip.mp4 --hand right --output-npz motors.npz
uv run demos/retarget_video.py --input clip.mp4 --hand right --output-video out.mp4
uv run demos/retarget_video.py --input clip.mp4 --hand right \
    --output-npz motors.npz --output-video out.mp4
```

| Flag | Default | Description |
|------|---------|-------------|
| `--input` | required | Input video file |
| `--hand` | `auto` | `left`, `right`, or `auto` |
| `--output-npz` | — | Save motor commands (at least one output required) |
| `--output-video` | — | Side-by-side video: source frame + Sapien render |
| `--panel-width/height` | `640`/`480` | Size of each panel in the output video |

---

## MediaPipe → 25-point Conversion

Standard MediaPipe gives 21 landmarks. The 4 missing metacarpal joints (indices 5, 10, 15, 20) are synthesised by placing them at 1/3 of the wrist→MCP vector, consistent with palm anatomy. This conversion is applied automatically in all demo scripts via `demos/_utils.py:mp21_to_xr25()`.

## Handedness Convention

MediaPipe uses a **mirrored** convention for front-facing cameras: its `"Left"` label corresponds to the person's **right** hand. All demo scripts apply this correction automatically when `--hand auto` is used.

## Typical Workflow

```
# 1. Record raw hand motion from a video
uv run demos/record_video.py --input clip.mp4 --hand right --output-npz landmarks.npz

# 2a. Retarget to motor commands
uv run demos/retarget_npz.py --input landmarks.npz --output-npz motors.npz

# 2b. Or retarget + render visualisation video
uv run demos/retarget_npz.py --input landmarks.npz --output-video hand.mp4

# — or do steps 1+2 in one shot from a video —
uv run demos/retarget_video.py --input clip.mp4 --hand right \
    --output-npz motors.npz --output-video out.mp4
```
