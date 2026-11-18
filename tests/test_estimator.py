"""Integration checks for the MediaPipe wrapper. Skipped if models aren't downloaded."""

import numpy as np
import pytest

from physiopal.pose.estimator import MODELS_DIR, PoseEstimator

MODEL = MODELS_DIR / "pose_landmarker_lite.task"
pytestmark = pytest.mark.skipif(
    not MODEL.exists(), reason="run: python scripts/download_models.py lite"
)


def test_blank_frame_has_no_pose():
    with PoseEstimator(str(MODEL)) as est:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert est.process(frame, 0) is None


def test_timestamps_forced_monotonic():
    with PoseEstimator(str(MODEL)) as est:
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        est.process(frame, 100)
        est.process(frame, 100)  # same timestamp would crash MediaPipe; wrapper bumps it
