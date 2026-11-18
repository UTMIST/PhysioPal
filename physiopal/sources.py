"""Frame sources: webcam or video file, with the timestamps MediaPipe needs."""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray  # BGR, as OpenCV loads it
    timestamp_ms: int
    index: int


class VideoSource:
    """``VideoSource("0")`` opens webcam 0; ``VideoSource("clip.mp4")`` opens a file.

    Webcam timestamps come from the wall clock; file timestamps come from the
    file's fps, so a recorded clip gives the same results every run.
    """

    def __init__(self, source: str | int, width: int | None = 1280, height: int | None = 720):
        self.is_camera = isinstance(source, int) or str(source).isdigit()
        self.cap = cv2.VideoCapture(int(source) if self.is_camera else str(source))
        if not self.cap.isOpened():
            hint = (
                " On macOS, allow camera access for your terminal in System Settings ->"
                " Privacy & Security -> Camera." if self.is_camera else ""
            )
            raise RuntimeError(f"Could not open video source {source!r}.{hint}")
        if self.is_camera and width and height:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def __iter__(self) -> Iterator[Frame]:
        start = time.monotonic()
        i = 0
        while True:
            ok, image = self.cap.read()
            if not ok:
                return
            if self.is_camera:
                ts = int((time.monotonic() - start) * 1000)
            else:
                ts = int(i * 1000 / self.fps)
            yield Frame(image, ts, i)
            i += 1

    def close(self) -> None:
        self.cap.release()
