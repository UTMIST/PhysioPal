from .base import Exercise, ExerciseResult
from .registry import available, build_exercise, register

# Importing each module runs its @register decorator. Add new exercises here.
from . import lunge, shoulder_abduction, squat  # noqa: E402, F401  isort: skip

__all__ = ["Exercise", "ExerciseResult", "available", "build_exercise", "register"]
