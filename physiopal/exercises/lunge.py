"""Lunge. Owner: <dev name>"""

from __future__ import annotations

from ..pose import Pose
from .base import Exercise, ExerciseResult
from .registry import register


@register
class Lunge(Exercise):
    name = "lunge"
    display_name = "Lunge"

    def reset(self) -> None:
        pass  # TODO: reset your state / rep count here

    def step(self, pose: Pose) -> ExerciseResult:
        # TODO: rep counting, states, and form feedback.
        # Run it live with:  python -m physiopal.run -e lunge
        return ExerciseResult()
