"""The Exercise interface the runner relies on."""

from physiopal.exercises import Exercise, ExerciseResult
from physiopal.pose import Pose


class Counter(Exercise):
    name = "counter"
    display_name = "Counter"

    def reset(self) -> None:
        self.frames = 0

    def step(self, pose: Pose) -> ExerciseResult:
        self.frames += 1
        return ExerciseResult(reps=self.frames)


def test_step_called_with_pose(standing):
    ex = Counter()
    ex.update(standing)
    assert ex.update(standing).reps == 2


def test_no_pose_skips_step():
    ex = Counter()
    r = ex.update(None)
    assert ex.frames == 0 and r.feedback


def test_reset(standing):
    ex = Counter()
    ex.update(standing)
    ex.reset()
    assert ex.frames == 0
