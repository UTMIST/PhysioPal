# How the repo fits together

```
webcam / video file          physiopal/sources.py
      │ frame
      ▼
MediaPipe pose               physiopal/pose/estimator.py
      │ Pose (33 landmarks), or None if no one is in frame
      ▼
exercise.update(pose)        physiopal/exercises/<name>.py   ← exercise devs
      │ ExerciseResult(reps, state, feedback, debug)
      ▼
overlay + window             physiopal/render/overlay.py, physiopal/run.py
```

The only contract between the pipeline and exercise code is: **you receive a
`Pose`, you return an `ExerciseResult`** (`physiopal/exercises/base.py`). Put
anything you want to see on screen while developing into `debug`, and it'll be
drawn under the rep count.

## Folders

| Path | What | Owner |
|---|---|---|
| `physiopal/pose/` | MediaPipe wrapper, landmark names, `Pose` type | team lead |
| `physiopal/sources.py` | reads webcam or video file | team lead |
| `physiopal/render/overlay.py` | draws skeleton + HUD | team lead |
| `physiopal/run.py` | the runner you launch | team lead |
| `physiopal/exercises/base.py` | the `Exercise` / `ExerciseResult` interface | team lead |
| `physiopal/exercises/squat.py` | squat | squat dev |
| `physiopal/exercises/lunge.py` | lunge | lunge dev |
| `physiopal/exercises/shoulder_abduction.py` | shoulder abduction | shoulder abduction dev |
| `scripts/download_models.py` | downloads MediaPipe models into `models/` | |
| `tests/` | pytest | everyone |
| `ml/` | original UTMIST ML-seed template, parked | |

Exercise devs can add any files they need (helpers, configs, notebooks, data).
Agree with the team lead if something should be shared between exercises.
