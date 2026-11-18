"""Data types that flow out of the pose estimator.

Exercise code only ever sees a ``Pose``; it never imports MediaPipe.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .landmarks import L


@dataclass(frozen=True)
class Landmark:
    """One landmark.

    In ``Pose.landmarks``: x, y are normalized to [0, 1] of the image
    (origin top-left, y points DOWN); z is relative depth (smaller = closer).
    In ``Pose.world``: x, y, z are metres, origin between the hips, y points DOWN.
    """

    x: float
    y: float
    z: float
    visibility: float = 1.0  # how likely the point is visible (not occluded), 0..1
    presence: float = 1.0  # how likely the point is in frame at all, 0..1


@dataclass(frozen=True)
class Pose:
    landmarks: tuple[Landmark, ...]  # 33 normalized image landmarks
    world: tuple[Landmark, ...]  # 33 world landmarks (metres)
    width: int  # image size in pixels, needed to undo normalization
    height: int
    timestamp_ms: int

    def __getitem__(self, lm: L) -> Landmark:
        return self.landmarks[lm]

    def px(self, lm: L) -> np.ndarray:
        """2D pixel coordinates (x, y), i.e. normalized x/y scaled by the frame size."""
        p = self.landmarks[lm]
        return np.array([p.x * self.width, p.y * self.height])

    def xyz(self, lm: L) -> np.ndarray:
        """3D world coordinates in metres (origin between the hips)."""
        p = self.world[lm]
        return np.array([p.x, p.y, p.z])

    def visibility(self, lm: L) -> float:
        return self.landmarks[lm].visibility

    def visible(self, lms: Iterable[L], threshold: float = 0.5) -> bool:
        """True if every landmark in ``lms`` is at least ``threshold`` visible."""
        return all(self.landmarks[lm].visibility >= threshold for lm in lms)
