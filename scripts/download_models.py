"""Download MediaPipe pose landmarker models into models/.

    python scripts/download_models.py            # full (default)
    python scripts/download_models.py --all      # lite + full + heavy

lite  ~ fastest, least accurate   (good for laptops without a GPU)
full  ~ balanced                  (default used by the runner)
heavy ~ most accurate, slowest
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_{v}/float16/latest/pose_landmarker_{v}.task"
)
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
VARIANTS = ("lite", "full", "heavy")


def download(variant: str, force: bool = False) -> Path:
    MODELS_DIR.mkdir(exist_ok=True)
    dest = MODELS_DIR / f"pose_landmarker_{variant}.task"
    if dest.exists() and not force:
        print(f"[skip] {dest.name} already exists")
        return dest
    print(f"[get ] {dest.name} ...", end=" ", flush=True)
    urllib.request.urlretrieve(URL.format(v=variant), dest)
    print(f"{dest.stat().st_size / 1e6:.1f} MB")
    return dest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("variant", nargs="?", default="full", choices=VARIANTS)
    p.add_argument("--all", action="store_true", help="download every variant")
    p.add_argument("--force", action="store_true", help="re-download even if present")
    args = p.parse_args()
    for v in VARIANTS if args.all else (args.variant,):
        download(v, args.force)


if __name__ == "__main__":
    main()
