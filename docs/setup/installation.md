# Setup

## Prerequisites
- Python 3.12
- [uv](https://docs.astral.sh/uv/) (recommended, fast) or plain `venv` + pip
- A webcam

## Install

```bash
git clone https://github.com/UTMIST/PhysioPal.git
cd PhysioPal

# uv
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"

# or plain venv
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# MediaPipe pose models (~45 MB, into models/)
python scripts/download_models.py --all
```

## Check it works

```bash
pytest tests/ -v                        # should all pass
python -m physiopal.run                 # webcam window with your skeleton drawn on it
python -m physiopal.run -e squat        # with an exercise loaded
```

Keys, recording, options and troubleshooting: [Running the camera](../running.md).

## Adding packages
Add them to `dependencies` (or the `dev` extra) in `pyproject.toml`, then re-run
`uv pip install -e ".[dev]"`.

## Conda
Conda users: `conda env create -f environment.yml && conda activate physiopal`,
then the `download_models.py` step.

## The ML template
The original UTMIST training template (ResNet/YOLO, Hydra) is parked in `ml/`.
Install its extras with `uv pip install -e ".[ml]"` and run it from inside `ml/`.
