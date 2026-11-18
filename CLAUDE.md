# PhysioPal

Webcam physio coaching: MediaPipe pose → per-exercise logic → reps + form feedback → OpenCV overlay.
Exercises: squat, lunge, shoulder abduction (each owned by a different dev).

## Key commands

```bash
source .venv/bin/activate
python scripts/download_models.py --all          # MediaPipe models -> models/
python -m physiopal.run -e squat                 # live webcam
python -m physiopal.run -e squat --source x.mp4 --headless   # replay a clip, print summary
pytest tests/ -v
ruff check physiopal/ tests/
```

## Architecture

- `physiopal/pose/estimator.py` is the only module that imports mediapipe (Tasks API `PoseLandmarker`,
  VIDEO mode). `mp.solutions` does not exist in mediapipe 1.x.
- Exercises subclass `Exercise` (`physiopal/exercises/base.py`), implement `step(pose) -> ExerciseResult`,
  and use `@register`. The exercise logic (landmarks, features, states, reps, feedback) belongs
  to the exercise devs; don't prescribe or fill it in unless asked.
- `ExerciseResult(reps, state, feedback, debug)` is everything the overlay draws.
- Landmarks by name via `physiopal.pose.L`. Normalized image y points down.

## Conventions

- `ml/` is the parked UTMIST ML-seed template; leave it alone unless asked. Run it from inside `ml/`.
- Branches `feature/<issue>-<desc>`, conventional commits, squash merge (docs/contributing/guidelines.md).
