# PhysioPal

Real-time physiotherapy exercise coaching from a webcam. PhysioPal tracks your
body with MediaPipe pose estimation, counts reps, and gives live form feedback
for **squats**, **lunges**, and **shoulder abduction**.

## Problem

*Who is this for and why does it matter?*

## Approach

```
webcam → MediaPipe Pose (33 landmarks) → exercise logic → reps + form feedback → overlay
```

See [docs/architecture.md](docs/architecture.md).

## Quick start

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python scripts/download_models.py --all

python -m physiopal.run                 # webcam + skeleton
python -m physiopal.run -e squat        # with an exercise loaded
pytest tests/ -v
```

Full setup: [docs/setup/installation.md](docs/setup/installation.md) · Camera commands, keys and troubleshooting: [docs/running.md](docs/running.md).

## Exercises

| Exercise | Code | Owner |
|---|---|---|
| Squat | [`physiopal/exercises/squat.py`](physiopal/exercises/squat.py) | _TBD_ |
| Lunge | [`physiopal/exercises/lunge.py`](physiopal/exercises/lunge.py) | _TBD_ |
| Shoulder abduction | [`physiopal/exercises/shoulder_abduction.py`](physiopal/exercises/shoulder_abduction.py) | _TBD_ |

New here? Read [docs/mediapipe.md](docs/mediapipe.md).

## Results

*Demo GIFs, rep-count accuracy, etc.*

## UTMIST metadata

- **Team:** *names and roles*
- **Term:** *e.g. Fall 2026 – Winter 2027*
- **Type:** [x] Engineering · [ ] Research
- **Status:** webcam + pose pipeline working; exercises in progress

## License

[MIT](LICENSE)
