import numpy as np

from physiopal.exercises import ExerciseResult
from physiopal.render.overlay import draw_hud, draw_skeleton


def test_overlay_draws_without_crashing(standing):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    draw_skeleton(frame, standing)
    draw_skeleton(frame, None)
    result = ExerciseResult(reps=3, state="down", feedback=["Chest up"],
                            debug={"knee": 92.4, "side": "left"})
    draw_hud(frame, "Squat", result, fps=30.0)
    draw_hud(frame, "Skeleton only", None, fps=30.0)
    assert frame.any()
