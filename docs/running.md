# Running the camera

First time? Do [Setup](setup/installation.md) first (install + download models).

## Every time you open a terminal

```bash
cd ~/PhysioPal              # or wherever you cloned it
source .venv/bin/activate   # you should see (.venv) in your prompt
```

## Start the webcam

```bash
python -m physiopal.run                        # skeleton only
python -m physiopal.run -e squat               # with your exercise loaded
python -m physiopal.run -e lunge
python -m physiopal.run -e shoulder_abduction
```

A window called **PhysioPal** opens with your skeleton drawn on top. If you
don't see it, check behind your editor.

| Key | Does |
|---|---|
| `q` / `Esc` | quit |
| `1` / `2` / `3` | switch to lunge / shoulder abduction / squat |
| `r` | reset (calls your exercise's `reset()`) |
| `m` | toggle mirror view |

Skeleton colours: **green** = MediaPipe is confident, **yellow** = unsure,
**red** = guessing (point is hidden or off-screen).

## Record and replay clips

Recording a clip lets you test your code on the exact same movement every time:

```bash
python -m physiopal.run -e squat --record data/squat_1.mp4    # record while you move
python -m physiopal.run -e squat --source data/squat_1.mp4    # replay it later
```

## All options

| Option | Example | Meaning |
|---|---|---|
| `-e`, `--exercise` | `-e squat` | which exercise to run (omit for skeleton only) |
| `-s`, `--source` | `--source 1`, `--source clip.mp4` | camera number (default `0`) or a video file |
| `-m`, `--model` | `--model heavy` | `lite`, `full` (default), or `heavy` |
| `--record` | `--record out.mp4` | save the annotated video |
| `--no-mirror` | | don't flip the webcam view |
| `--headless` | | no window, just print a summary (for scripts/CI) |
| `--max-frames` | `--max-frames 300` | stop after N frames |

`python -m physiopal.run --help` prints the same list.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not open video source 0` | macOS: System Settings → Privacy & Security → Camera → turn on your terminal / VS Code, then fully quit (Cmd+Q) and reopen it. |
| Wrong camera (e.g. your iPhone) | `--source 1` (or `2`, …) |
| `Pose model not found` | `python scripts/download_models.py --all` |
| `No module named physiopal` / `mediapipe` | You forgot `source .venv/bin/activate`, or you're not in the repo folder. |
| Skeleton jumpy or missing | Step back so your whole body is in frame, improve the lighting, try `--model heavy`. |
| Laggy | `--model lite` |
