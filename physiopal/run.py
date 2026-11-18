"""Dev runner: webcam/video -> pose -> exercise -> overlay.

    python -m physiopal.run                              # webcam, skeleton only
    python -m physiopal.run --exercise squat             # webcam + squat logic
    python -m physiopal.run -e lunge --source clip.mp4   # replay a recorded clip
    python -m physiopal.run -e squat --record out.mp4    # save the annotated video
    python -m physiopal.run --source clip.mp4 --headless # no window, print summary (CI/tests)

Keys:  q/Esc quit   r reset reps   m toggle mirror   1/2/3 switch exercise
"""

from __future__ import annotations

import argparse
import time

import cv2

from .exercises import Exercise, available, build_exercise
from .pose.estimator import PoseEstimator
from .render.overlay import draw_hud, draw_skeleton
from .sources import VideoSource


def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("-e", "--exercise", choices=available(), help="omit for skeleton only")
    p.add_argument("-s", "--source", default="0", help="webcam index or video file path")
    p.add_argument("-m", "--model", default="full", help="lite | full | heavy | path/to.task")
    p.add_argument("--record", help="write annotated video to this .mp4 path")
    p.add_argument("--headless", action="store_true", help="no window; print a summary")
    p.add_argument("--max-frames", type=int, default=0, help="stop after N frames (0 = no limit)")
    p.add_argument("--no-mirror", action="store_true", help="don't flip the webcam view")
    args = p.parse_args(argv)

    source = VideoSource(args.source)
    mirror = source.is_camera and not args.no_mirror
    names = available()
    exercise: Exercise | None = build_exercise(args.exercise) if args.exercise else None
    writer = None
    frames = detected = 0
    fps, last = 0.0, time.monotonic()
    result = None

    with PoseEstimator(args.model) as estimator:
        for frame in source:
            pose = estimator.process(frame.image, frame.timestamp_ms)
            result = exercise.update(pose) if exercise else None
            frames += 1
            detected += pose is not None

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - last, 1e-6))
            last = now

            img = frame.image
            draw_skeleton(img, pose)
            if mirror:
                img = cv2.flip(img, 1)  # flip AFTER pose estimation so left/right stay correct
            title = exercise.display_name if exercise else "Skeleton only"
            draw_hud(img, title, result, fps)

            if args.record:
                if writer is None:
                    h, w = img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.record, fourcc, source.fps, (w, h))
                writer.write(img)

            if not args.headless:
                cv2.imshow("PhysioPal", img)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r") and exercise:
                    exercise.reset()
                if key == ord("m"):
                    mirror = not mirror
                if ord("1") <= key < ord("1") + len(names):
                    exercise = build_exercise(names[key - ord("1")])

            if args.max_frames and frames >= args.max_frames:
                break

    source.close()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    summary = {
        "frames": frames,
        "pose_detected": detected,
        "detection_rate": round(detected / frames, 3) if frames else 0.0,
        "reps": result.reps if result else None,
    }
    print(summary)
    return summary


if __name__ == "__main__":
    main()
