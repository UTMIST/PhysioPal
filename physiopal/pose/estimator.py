"""The ONLY file that talks to MediaPipe.

Turns BGR frames (what OpenCV gives you) into ``Pose`` objects. If we ever swap
pose models, or move pose estimation into the browser, only this file changes.
"""

from __future__ import annotations

import contextlib
import os
import sys
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from .types import Landmark, Pose

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


@contextlib.contextmanager
def _quiet_native_stderr(enabled: bool = True):
    """Mute MediaPipe's C++ startup log spam (it writes straight to fd 2, so
    env vars and Python logging can't silence it). Python exceptions still propagate.
    Set PHYSIOPAL_VERBOSE=1 to see the logs when debugging MediaPipe itself."""
    if not enabled or os.environ.get("PHYSIOPAL_VERBOSE"):
        yield
        return
    sys.stderr.flush()
    saved = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


class PoseEstimator:
    """Wraps MediaPipe's PoseLandmarker in VIDEO mode (uses tracking between frames).

    Usage::

        with PoseEstimator("full") as est:
            pose = est.process(frame_bgr, timestamp_ms)   # Pose | None
    """

    def __init__(
        self,
        model: str = "full",
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        path = Path(model)
        if not path.suffix:  # "lite" | "full" | "heavy"
            path = MODELS_DIR / f"pose_landmarker_{model}.task"
        if not path.exists():
            raise FileNotFoundError(
                f"Pose model not found at {path}. Run: python scripts/download_models.py"
            )
        options = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        with _quiet_native_stderr():
            self._landmarker = vision.PoseLandmarker.create_from_options(options)
        self._last_ts = -1
        self._warmed_up = False  # first inference also logs a lot

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> Pose | None:
        """Run pose estimation on one frame. Returns None if no person is found.

        ``timestamp_ms`` must strictly increase between calls (MediaPipe requirement).
        """
        timestamp_ms = max(int(timestamp_ms), self._last_ts + 1)
        self._last_ts = timestamp_ms

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        with _quiet_native_stderr(not self._warmed_up):
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        self._warmed_up = True
        if not result.pose_landmarks:
            return None

        def convert(points) -> tuple[Landmark, ...]:
            return tuple(
                Landmark(p.x, p.y, p.z, p.visibility or 0.0, p.presence or 0.0) for p in points
            )

        h, w = frame_bgr.shape[:2]
        return Pose(
            landmarks=convert(result.pose_landmarks[0]),
            world=convert(result.pose_world_landmarks[0]),
            width=w,
            height=h,
            timestamp_ms=timestamp_ms,
        )

    def close(self) -> None:
        with _quiet_native_stderr():
            self._landmarker.close()

    def __enter__(self) -> PoseEstimator:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
