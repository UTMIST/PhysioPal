"""Shared test helpers: build fake Poses without a camera or MediaPipe."""

from __future__ import annotations

import pytest

from physiopal.pose import NUM_LANDMARKS, L, Landmark, Pose

W, H = 1000, 1000


def make_pose(points: dict[L, tuple[float, float]], visibility: float = 0.99,
              timestamp_ms: int = 0) -> Pose:
    """Build a Pose from PIXEL coords (x, y) on a 1000x1000 image.

    Landmarks you don't specify sit at the image centre with visibility 0.
    """
    lms = []
    for i in range(NUM_LANDMARKS):
        if i in points:
            x, y = points[L(i)]
            lms.append(Landmark(x / W, y / H, 0.0, visibility, visibility))
        else:
            lms.append(Landmark(0.5, 0.5, 0.0, 0.0, 0.0))
    world = tuple(Landmark(lm.x - 0.5, lm.y - 0.5, 0.0, lm.visibility) for lm in lms)
    return Pose(tuple(lms), world, W, H, timestamp_ms)


def standing_pose(**kw) -> Pose:
    """A person standing straight, arms at sides (both sides overlap, like a side view)."""
    pts = {}
    for s in ("LEFT", "RIGHT"):
        dx = -40 if s == "LEFT" else 40
        pts[L[f"{s}_SHOULDER"]] = (500 + dx, 300)
        pts[L[f"{s}_ELBOW"]] = (500 + dx, 420)
        pts[L[f"{s}_WRIST"]] = (500 + dx, 530)
        pts[L[f"{s}_HIP"]] = (500 + dx, 550)
        pts[L[f"{s}_KNEE"]] = (500 + dx, 720)
        pts[L[f"{s}_ANKLE"]] = (500 + dx, 890)
    return make_pose(pts, **kw)


@pytest.fixture
def standing():
    return standing_pose()
