"""OpenCV drawing for the dev runner. Not used by exercise logic.

Split in two because of mirroring: draw the skeleton on the raw frame, flip it
for a selfie view, THEN draw text (so the text isn't mirrored).
"""

from __future__ import annotations

import cv2
import numpy as np

from ..exercises import ExerciseResult
from ..pose import CONNECTIONS, Pose

GREEN = (80, 200, 80)
YELLOW = (0, 210, 255)
RED = (60, 60, 230)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _vis_color(v: float) -> tuple[int, int, int]:
    return GREEN if v >= 0.7 else YELLOW if v >= 0.4 else RED


def draw_skeleton(frame: np.ndarray, pose: Pose | None) -> None:
    """Joints coloured by visibility: green = confident, yellow = unsure, red = guessed."""
    if pose is None:
        return
    pts = {i: tuple(int(c) for c in pose.px(i)) for i in range(len(pose.landmarks))}
    for a, b in CONNECTIONS:
        v = min(pose.visibility(a), pose.visibility(b))
        cv2.line(frame, pts[a], pts[b], _vis_color(v), 2, cv2.LINE_AA)
    for i in range(11, len(pose.landmarks)):  # skip face points 0-10 for clarity
        cv2.circle(frame, pts[i], 4, _vis_color(pose.visibility(i)), -1, cv2.LINE_AA)


def _text(frame, s, org, scale=0.6, color=WHITE, thick=1) -> None:
    cv2.putText(frame, s, org, FONT, scale, BLACK, thick + 3, cv2.LINE_AA)
    cv2.putText(frame, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_hud(frame: np.ndarray, title: str, result: ExerciseResult | None, fps: float) -> None:
    """Top-left: title, reps, state, then every value in result.debug. Bottom: feedback."""
    h, w = frame.shape[:2]
    _text(frame, title, (12, 30), 0.8, WHITE, 2)
    _text(frame, f"{fps:4.1f} fps", (w - 110, 30))
    if result is None:
        return

    _text(frame, f"Reps: {result.reps}", (12, 65), 0.9, GREEN, 2)
    if result.state:
        _text(frame, f"State: {result.state}", (12, 95))

    y = 130
    for name, value in result.debug.items():
        shown = f"{value:.1f}" if isinstance(value, float) else str(value)
        _text(frame, f"{name}: {shown}", (12, y), 0.5)
        y += 22

    for i, msg in enumerate(result.feedback[:3]):
        _text(frame, msg, (12, h - 20 - 32 * i), 0.8, YELLOW, 2)
