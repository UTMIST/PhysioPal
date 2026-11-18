"""The hand-off point between the pipeline and the exercise code.

The runner gives your exercise one ``Pose`` per frame and draws whatever
``ExerciseResult`` you return. How you get from one to the other (which
landmarks, features, states, rep logic, form checks) is entirely up to you.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

from ..pose import Pose


@dataclass
class ExerciseResult:
    """What the overlay shows for one frame. Every field is optional."""

    reps: int = 0
    state: str = ""  # whatever you call the current phase
    feedback: list[str] = field(default_factory=list)  # messages shown to the user
    debug: dict[str, Any] = field(default_factory=dict)  # any values you want on screen


class Exercise(ABC):
    name: ClassVar[str]  # used on the command line: --exercise <name>
    display_name: ClassVar[str]  # shown at the top of the window

    def __init__(self) -> None:
        self.reset()

    def update(self, pose: Pose | None) -> ExerciseResult:
        """Called once per frame. ``pose`` is None when nobody is detected."""
        if pose is None:
            return ExerciseResult(feedback=["No person detected"])
        return self.step(pose)

    def reset(self) -> None:
        """Called on start and when the user presses 'r'. Override to clear your state."""

    @abstractmethod
    def step(self, pose: Pose) -> ExerciseResult:
        """Your code: look at the pose, return what to show."""
