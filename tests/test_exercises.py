"""Every registered exercise must plug into the runner."""

import pytest

from physiopal.exercises import ExerciseResult, available, build_exercise


def test_three_exercises_registered():
    assert {"squat", "lunge", "shoulder_abduction"} <= set(available())


@pytest.mark.parametrize("name", available())
def test_returns_result(name, standing):
    ex = build_exercise(name)
    assert isinstance(ex.update(standing), ExerciseResult)
    assert isinstance(ex.update(None), ExerciseResult)
