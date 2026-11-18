# MediaPipe in this repo

What the pipeline gives your exercise code each frame, and the facts about
MediaPipe you'll need. How you use it is up to you.

## What you get

Your exercise's `step(pose)` is called once per frame with a `Pose`
(`physiopal/pose/types.py`). It holds MediaPipe's output for one person:
**33 landmarks**, each with:

| Field | Meaning |
|---|---|
| `x`, `y` | Position in the image, **normalized 0–1**. `(0, 0)` is the top-left corner; **y points down**. |
| `z` | Depth relative to the hips (smaller = closer to camera). Rough. |
| `visibility` | 0–1: how likely the point is visible (not hidden behind the body). |
| `presence` | 0–1: how likely the point is in the frame at all. |

Plus **world landmarks**: the same 33 points in 3D **metres**, with the origin
between the hips.

```python
from physiopal.pose import L

pose[L.LEFT_KNEE]            # Landmark(x, y, z, visibility, presence), normalized
pose.px(L.LEFT_KNEE)         # np.array([x, y]) in pixels
pose.xyz(L.LEFT_KNEE)        # np.array([x, y, z]) in metres (world landmarks)
pose.landmarks / pose.world  # all 33 of each
pose.width, pose.height      # frame size in pixels
pose.timestamp_ms            # time of this frame
```

## The 33 landmarks

Use them by name with `L.<NAME>` (`physiopal/pose/landmarks.py`) instead of raw
numbers. **LEFT means the person's left**, not the left of the image.

| # | Name | # | Name | # | Name |
|---|---|---|---|---|---|
| 0 | NOSE | 11 | LEFT_SHOULDER | 23 | LEFT_HIP |
| 1 | LEFT_EYE_INNER | 12 | RIGHT_SHOULDER | 24 | RIGHT_HIP |
| 2 | LEFT_EYE | 13 | LEFT_ELBOW | 25 | LEFT_KNEE |
| 3 | LEFT_EYE_OUTER | 14 | RIGHT_ELBOW | 26 | RIGHT_KNEE |
| 4 | RIGHT_EYE_INNER | 15 | LEFT_WRIST | 27 | LEFT_ANKLE |
| 5 | RIGHT_EYE | 16 | RIGHT_WRIST | 28 | RIGHT_ANKLE |
| 6 | RIGHT_EYE_OUTER | 17 | LEFT_PINKY | 29 | LEFT_HEEL |
| 7 | LEFT_EAR | 18 | RIGHT_PINKY | 30 | RIGHT_HEEL |
| 8 | RIGHT_EAR | 19 | LEFT_INDEX | 31 | LEFT_FOOT_INDEX |
| 9 | MOUTH_LEFT | 20 | RIGHT_INDEX | 32 | RIGHT_FOOT_INDEX |
| 10 | MOUTH_RIGHT | 21 | LEFT_THUMB | | |
| | | 22 | RIGHT_THUMB | | |

Official diagram: [pose landmarker guide](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker).

## Good to know

- **Normalized x and y use different scales** (the frame is wider than it is
  tall), so a distance or angle computed from raw `.x/.y` is skewed.
  `pose.px()` gives true pixel positions.
- **Hidden points still get a position.** MediaPipe guesses where an occluded
  landmark is and gives it a low `visibility`.
- **Coordinates jitter** a little from frame to frame, even when standing still.
- **Frame rate varies** between laptops, so `pose.timestamp_ms` is the reliable clock.
- **Mirroring:** the runner flips the *display* like a mirror, but MediaPipe
  sees the un-flipped frame, so `LEFT_*` is always the person's real left.
- **Old tutorials won't work.** Most examples online use `mp.solutions.pose`,
  which was removed in MediaPipe 1.x. This repo uses the newer Tasks API; only
  `physiopal/pose/estimator.py` talks to MediaPipe.

## Models

`python scripts/download_models.py --all` downloads three models into `models/`.
Choose one with `--model`:

| Model | Speed | Accuracy |
|---|---|---|
| `lite` | fastest | lowest |
| `full` | ~30 fps live on a MacBook (camera-limited) | good, **default** |
| `heavy` | slowest | best |
