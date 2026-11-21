"""Lunge exercise tracker (lateral / side view).

Camera setup:
    User stands perpendicular (45–90 deg lateral / sagittal view) to the webcam,
    facing either left or right. All angles and ratios are computed in pixel
    space via ``pose.px()`` and gated on ``pose.visible()``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal

import numpy as np

from ..pose import L, Pose
from .base import Exercise, ExerciseResult
from .registry import register

Side = Literal["LEFT", "RIGHT"]
Facing = Literal[-1, 1]  # -1 = facing left, +1 = facing right

_LEFT_LEG_LMS: tuple[L, ...] = (L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE)
_RIGHT_LEG_LMS: tuple[L, ...] = (L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE)

_FAR_LEG_JOINTS: frozenset[L] = frozenset(
    (L.LEFT_KNEE, L.RIGHT_KNEE, L.LEFT_ANKLE, L.RIGHT_ANKLE)
)

_TRACKED_LMS: tuple[L, ...] = (
    *_LEFT_LEG_LMS,
    *_RIGHT_LEG_LMS,
    L.LEFT_SHOULDER,
    L.RIGHT_SHOULDER,
    L.LEFT_EAR,
    L.RIGHT_EAR,
    L.NOSE,
    L.LEFT_HEEL,
    L.RIGHT_HEEL,
    L.LEFT_FOOT_INDEX,
    L.RIGHT_FOOT_INDEX,
)


class LungePhase(str, Enum):
    """Discrete movement phases reported in ``ExerciseResult.state``."""

    NO_POSE = "NO_POSE"
    STANDING = "STANDING"
    DESCENDING = "DESCENDING"
    BOTTOM = "BOTTOM"
    ASCENDING = "ASCENDING"


@dataclass(frozen=True)
class LungeThresholds:
    """Tunable thresholds for rep counting and form checks."""

    # Visibility & timestamp-based exponential smoothing
    min_visibility: float = 0.5
    far_leg_min_visibility: float = 0.3
    ear_min_visibility: float = 0.6
    smoothing_tau_ms: float = 60.0
    max_gap_ms: int = 500
    no_pose_abort_ms: int = 1500

    # Squat rejection (lunge geometry gates)
    min_lunge_stance_ratio: float = 0.40  # Horizontal ankle span / leg length
    min_knee_drop_ratio: float = 0.22  # Vertical knee height gap / thigh length

    # State machine & rep counting (degrees; smaller = deeper bend)
# State machine & rep counting (degrees; smaller = deeper bend)
    descent_start_knee_deg: float = 145.0
    standing_reset_knee_deg: float = 150.0
    valid_front_knee_deg: float = 105.0
    valid_back_knee_deg: float = 120.0
    turnaround_hysteresis_deg: float = 15.0
    shallow_attempt_knee_deg: float = 135.0
    shallow_feedback_hold_ms: int = 1500

    # Form checks (trigger vs. clear hysteresis)
    knee_past_toe_trigger_ratio: float = 0.2
    knee_past_toe_clear_ratio: float = 0.1

    trunk_forward_trigger_deg: float = 20.0
    trunk_forward_clear_deg: float = 15.0
    trunk_backward_trigger_deg: float = -18.0
    trunk_backward_clear_deg: float = -14.0

    short_stance_check_knee_deg: float = 135.0
    short_stance_trigger_ratio: float = 0.65
    short_stance_clear_ratio: float = 0.72

    forward_head_trigger_deg: float = 50.0
    forward_head_clear_deg: float = 40.0


def _side_lm(side: Side, name: str) -> L:
    return L[f"{side}_{name}"]


def _other_side(side: Side) -> Side:
    return "RIGHT" if side == "LEFT" else "LEFT"


def _joint_angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Return the interior angle at vertex ``b`` in degrees [0, 180]."""
    ba = a - b
    bc = c - b
    denom = float(np.linalg.norm(ba) * np.linalg.norm(bc))
    if denom < 1e-6:
        return 180.0
    cos_val = float(np.dot(ba, bc)) / denom
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_val))))


def _tilt_from_vertical_deg(
    bottom: np.ndarray, top: np.ndarray, facing: Facing
) -> float:
    """Signed angle (deg) of vector ``top - bottom`` from vertical up (0, -1).

    Positive = leaning forward in ``facing`` direction; negative = backward.
    """
    vec = top - bottom
    return math.degrees(math.atan2(facing * float(vec[0]), float(-vec[1])))


@register
class Lunge(Exercise):
    name = "lunge"
    display_name = "Lunge"

    def __init__(self, thresholds: LungeThresholds | None = None) -> None:
        self._t = thresholds or LungeThresholds()
        super().__init__()  # Calls self.reset()

    def reset(self) -> None:
        """Clear all tracking state, counters, smoothing buffers, and feedback."""
        self._reps: int = 0
        self._phase: LungePhase = LungePhase.STANDING
        self._front_leg: Side | None = None
        self._facing: Facing | None = None
        self._front_ankle_x: float | None = None
        self._back_ankle_x: float | None = None
        self._reached_depth: bool = False
        self._min_front_knee_deg: float = 180.0

        self._last_timestamp_ms: int | None = None
        self._no_pose_since_ms: int | None = None
        self._smoothed: dict[L, np.ndarray] = {}

        self._active_errors: set[str] = set()
        self._shallow_until_ms: int | None = None

    def update(self, pose: Pose | None) -> ExerciseResult:
        """Preserve rep count when MediaPipe detects no person on a frame."""
        if pose is None:
            return ExerciseResult(
                reps=self._reps,
                state=LungePhase.NO_POSE.value,
                feedback=["No person detected"],
            )
        return self.step(pose)

    def step(self, pose: Pose) -> ExerciseResult:
        """Process one frame and return rep count, phase, feedback, and debug."""
        ts = pose.timestamp_ms

        near_side = self._detect_near_leg(pose)
        if near_side is None:
            return self._handle_low_visibility(ts)

        self._no_pose_since_ms = None
        pts = self._update_smoothed_points(pose, ts)
        self._last_timestamp_ms = ts

        facing = self._detect_facing(pts, near_side)
        if facing is not None and self._phase == LungePhase.STANDING:
            self._facing = facing
        elif self._facing is None:
            self._facing = facing

        front_side, back_side = self._resolve_leg_sides(pts, near_side)

        front_angle = _joint_angle_deg(
            self._hip_point(pts, front_side, near_side),
            pts[_side_lm(front_side, "KNEE")],
            pts[_side_lm(front_side, "ANKLE")],
        )
        back_angle: float | None = None
        if back_side is not None:
            back_angle = _joint_angle_deg(
                self._hip_point(pts, back_side, near_side),
                pts[_side_lm(back_side, "KNEE")],
                pts[_side_lm(back_side, "ANKLE")],
            )

        stance_ratio, knee_drop_ratio = self._compute_lunge_geometry(
            pts, near_side, front_side, back_side
        )

        shallow_triggered = self._advance_state_machine(
            pts=pts,
            front_side=front_side,
            back_side=back_side,
            front_angle=front_angle,
            back_angle=back_angle,
            stance_ratio=stance_ratio,
            knee_drop_ratio=knee_drop_ratio,
        )

        feedback, form_debug = self._evaluate_form(
            pose=pose,
            pts=pts,
            ts=ts,
            front_side=front_side,
            front_angle=front_angle,
            stance_ratio=stance_ratio,
            shallow_triggered=shallow_triggered,
        )

        debug: dict[str, Any] = {
            "front_leg": self._front_leg or front_side,
            "facing": self._facing,
            "front_knee_deg": round(front_angle, 1),
            "back_knee_deg": None if back_angle is None else round(back_angle, 1),
            "stance_ratio": None if stance_ratio is None else round(stance_ratio, 2),
            "knee_drop": (
                None if knee_drop_ratio is None else round(knee_drop_ratio, 2)
            ),
            **form_debug,
        }
        return ExerciseResult(
            reps=self._reps,
            state=self._phase.value,
            feedback=feedback,
            debug=debug,
        )

    # ------------------------------------------------------------------
    # Visibility gating & timestamp-based smoothing
    # ------------------------------------------------------------------

    def _detect_near_leg(self, pose: Pose) -> Side | None:
        """Return the clearest fully visible leg, or None if neither is visible."""
        left_ok = pose.visible(_LEFT_LEG_LMS, self._t.min_visibility)
        right_ok = pose.visible(_RIGHT_LEG_LMS, self._t.min_visibility)
        if not left_ok and not right_ok:
            return None
        if left_ok and not right_ok:
            return "LEFT"
        if right_ok and not left_ok:
            return "RIGHT"
        left_score = sum(pose.visibility(lm) for lm in _LEFT_LEG_LMS)
        right_score = sum(pose.visibility(lm) for lm in _RIGHT_LEG_LMS)
        return "LEFT" if left_score >= right_score else "RIGHT"

    def _hip_point(
        self, pts: dict[L, np.ndarray], side: Side, near_side: Side
    ) -> np.ndarray:
        """Return ``side`` hip, falling back to ``near_side`` hip if occluded."""
        hip = pts.get(_side_lm(side, "HIP"))
        return hip if hip is not None else pts[_side_lm(near_side, "HIP")]

    def _has_knee_and_ankle(self, pts: dict[L, np.ndarray], side: Side) -> bool:
        return _side_lm(side, "KNEE") in pts and _side_lm(side, "ANKLE") in pts

    def _handle_low_visibility(self, ts: int) -> ExerciseResult:
        if self._no_pose_since_ms is None:
            self._no_pose_since_ms = ts
        elif ts - self._no_pose_since_ms >= self._t.no_pose_abort_ms:
            self._abort_in_progress_rep()
            self._smoothed.clear()
            self._last_timestamp_ms = None

        return ExerciseResult(
            reps=self._reps,
            state=LungePhase.NO_POSE.value,
            feedback=["Step fully into frame (side view)"],
        )

    def _abort_in_progress_rep(self) -> None:
        self._phase = LungePhase.STANDING
        self._front_leg = None
        self._front_ankle_x = None
        self._back_ankle_x = None
        self._reached_depth = False
        self._min_front_knee_deg = 180.0
        self._active_errors.clear()

    def _update_smoothed_points(self, pose: Pose, ts: int) -> dict[L, np.ndarray]:
        dt = 0 if self._last_timestamp_ms is None else ts - self._last_timestamp_ms
        if (
            self._last_timestamp_ms is None
            or dt <= 0
            or dt > self._t.max_gap_ms
            or self._t.smoothing_tau_ms <= 0
        ):
            alpha = 1.0
        else:
            alpha = 1.0 - math.exp(-dt / self._t.smoothing_tau_ms)

        visible_pts: dict[L, np.ndarray] = {}
        for lm in _TRACKED_LMS:
            thresh = (
                self._t.far_leg_min_visibility
                if lm in _FAR_LEG_JOINTS
                else self._t.min_visibility
            )
            if pose.visibility(lm) < thresh:
                self._smoothed.pop(lm, None)
                continue
            raw = pose.px(lm)
            prev = self._smoothed.get(lm)
            smoothed = (
                raw
                if prev is None or alpha >= 1.0
                else (alpha * raw + (1.0 - alpha) * prev)
            )
            self._smoothed[lm] = smoothed
            visible_pts[lm] = smoothed
        return visible_pts

    # ------------------------------------------------------------------
    # Facing detection, front-leg locking & lunge-vs-squat geometry
    # ------------------------------------------------------------------

    def _detect_facing(
        self, pts: dict[L, np.ndarray], near_side: Side
    ) -> Facing | None:
        foot_dx: list[float] = []
        for side in ("LEFT", "RIGHT"):
            heel = pts.get(_side_lm(side, "HEEL"))
            toe = pts.get(_side_lm(side, "FOOT_INDEX"))
            if heel is not None and toe is not None:
                dx = float(toe[0] - heel[0])
                if abs(dx) > 5.0:
                    foot_dx.append(dx)
        if foot_dx:
            return 1 if sum(foot_dx) >= 0 else -1

        knee_protrusion = 0.0
        for side in ("LEFT", "RIGHT"):
            if not self._has_knee_and_ankle(pts, side):
                continue
            hip = self._hip_point(pts, side, near_side)
            knee = pts[_side_lm(side, "KNEE")]
            ankle = pts[_side_lm(side, "ANKLE")]
            chord_mid_x = 0.5 * (float(hip[0]) + float(ankle[0]))
            knee_protrusion += float(knee[0]) - chord_mid_x
        if abs(knee_protrusion) > 15.0:
            return 1 if knee_protrusion > 0 else -1

        nose = pts.get(L.NOSE)
        l_sh = pts.get(L.LEFT_SHOULDER)
        r_sh = pts.get(L.RIGHT_SHOULDER)
        if nose is not None and (l_sh is not None or r_sh is not None):
            sh_x = (
                0.5 * (float(l_sh[0]) + float(r_sh[0]))
                if l_sh is not None and r_sh is not None
                else float((l_sh if l_sh is not None else r_sh)[0])
            )
            dx = float(nose[0]) - sh_x
            if abs(dx) > 5.0:
                return 1 if dx > 0 else -1

        return None

    def _resolve_leg_sides(
        self, pts: dict[L, np.ndarray], near_side: Side
    ) -> tuple[Side, Side | None]:
        """Return (front_side, back_side), guarding against occlusion & swaps."""
        left_avail = self._has_knee_and_ankle(pts, "LEFT")
        right_avail = self._has_knee_and_ankle(pts, "RIGHT")

        if not (left_avail and right_avail):
            only_side: Side = "LEFT" if left_avail else "RIGHT"
            return only_side, None

        l_ankle_x = float(pts[L.LEFT_ANKLE][0])
        r_ankle_x = float(pts[L.RIGHT_ANKLE][0])

        if self._front_leg is not None:
            locked_front = self._front_leg
            locked_back = _other_side(locked_front)
            f_x = float(pts[_side_lm(locked_front, "ANKLE")][0])
            b_x = float(pts[_side_lm(locked_back, "ANKLE")][0])

            if self._front_ankle_x is None or self._back_ankle_x is None:
                self._front_ankle_x = f_x
                self._back_ankle_x = b_x
                return locked_front, locked_back

            keep_cost = abs(f_x - self._front_ankle_x) + \
                abs(b_x - self._back_ankle_x)
            swap_cost = abs(b_x - self._front_ankle_x) + \
                abs(f_x - self._back_ankle_x)

            anchor_sep = abs(self._front_ankle_x - self._back_ankle_x)
            if anchor_sep > 30.0 and swap_cost + 20.0 < keep_cost:
                return locked_back, locked_front

            self._front_ankle_x = 0.8 * self._front_ankle_x + 0.2 * f_x
            self._back_ankle_x = 0.8 * self._back_ankle_x + 0.2 * b_x
            return locked_front, locked_back

        facing = self._facing or 1
        if abs(l_ankle_x - r_ankle_x) < 15.0:
            return near_side, _other_side(near_side)
        if facing * (l_ankle_x - r_ankle_x) >= 0:
            return "LEFT", "RIGHT"
        return "RIGHT", "LEFT"

    def _compute_lunge_geometry(
        self,
        pts: dict[L, np.ndarray],
        near_side: Side,
        front_side: Side,
        back_side: Side | None,
    ) -> tuple[float | None, float | None]:
        """Compute (stance_ratio, knee_drop_ratio) to distinguish lunges from squats."""
        if back_side is None:
            return None, None

        f_hip = self._hip_point(pts, front_side, near_side)
        f_knee = pts[_side_lm(front_side, "KNEE")]
        f_ankle = pts[_side_lm(front_side, "ANKLE")]
        b_knee = pts[_side_lm(back_side, "KNEE")]
        b_ankle = pts[_side_lm(back_side, "ANKLE")]

        thigh_len = float(np.linalg.norm(f_hip - f_knee))
        shin_len = float(np.linalg.norm(f_knee - f_ankle))
        leg_len = thigh_len + shin_len

        stance_ratio = (
            abs(float(f_ankle[0] - b_ankle[0])) /
            leg_len if leg_len > 1e-3 else None
        )
        knee_drop_ratio = (
            abs(float(b_knee[1] - f_knee[1])) /
            thigh_len if thigh_len > 1e-3 else None
        )
        return stance_ratio, knee_drop_ratio

    # ------------------------------------------------------------------
    # Rep state machine
    # ------------------------------------------------------------------

    def _advance_state_machine(
        self,
        *,
        pts: dict[L, np.ndarray],
        front_side: Side,
        back_side: Side | None,
        front_angle: float,
        back_angle: float | None,
        stance_ratio: float | None,
        knee_drop_ratio: float | None,
    ) -> bool:
        """Advance movement phase and return True on a shallow-rep turnaround."""
        shallow_triggered = False
        min_knee = (
            front_angle if back_angle is None else min(front_angle, back_angle)
        )
        is_split = (
            stance_ratio is not None
            and stance_ratio >= self._t.min_lunge_stance_ratio
        )
        has_knee_drop = (
            knee_drop_ratio is not None
            and knee_drop_ratio >= self._t.min_knee_drop_ratio
        )

        if self._phase == LungePhase.STANDING:
            if is_split and min_knee <= self._t.descent_start_knee_deg:
                self._phase = LungePhase.DESCENDING
                self._front_leg = front_side
                self._front_ankle_x = float(
                    pts[_side_lm(front_side, "ANKLE")][0])
                self._back_ankle_x = (
                    float(pts[_side_lm(back_side, "ANKLE")][0])
                    if back_side is not None
                    else None
                )
                self._reached_depth = False
                self._min_front_knee_deg = front_angle

        back_deep_ok = (
            back_angle is not None and back_angle <= self._t.valid_back_knee_deg
        )
        back_standing_ok = (
            back_angle is None
            or back_angle >= self._t.standing_reset_knee_deg - 15.0
        )

        if self._phase == LungePhase.DESCENDING:
            self._min_front_knee_deg = min(
                self._min_front_knee_deg, front_angle)
            if (
                is_split
                and has_knee_drop
                and front_angle <= self._t.valid_front_knee_deg
                and back_deep_ok
            ):
                self._reached_depth = True
                self._phase = LungePhase.BOTTOM
                self._shallow_until_ms = None
            elif (
                front_angle
                >= self._min_front_knee_deg + self._t.turnaround_hysteresis_deg
            ):
                if (
                    is_split
                    and self._min_front_knee_deg <= self._t.shallow_attempt_knee_deg
                ):
                    shallow_triggered = True
                    self._phase = LungePhase.ASCENDING
                elif front_angle >= self._t.standing_reset_knee_deg or not is_split:
                    self._abort_in_progress_rep()
                    return False

        elif self._phase == LungePhase.BOTTOM:
            self._min_front_knee_deg = min(
                self._min_front_knee_deg, front_angle)
            if (
                front_angle >= self._t.standing_reset_knee_deg
                and back_standing_ok
            ):
                self._complete_rep()
                return False
            if (
                front_angle
                >= self._min_front_knee_deg + self._t.turnaround_hysteresis_deg
            ):
                self._phase = LungePhase.ASCENDING

        if self._phase == LungePhase.ASCENDING:
            if (
                not self._reached_depth
                and is_split
                and has_knee_drop
                and front_angle <= self._t.valid_front_knee_deg
                and back_deep_ok
            ):
                # Recover if a foot-plant rebound tripped ASCENDING before full descent
                self._reached_depth = True
                self._phase = LungePhase.BOTTOM
                self._min_front_knee_deg = front_angle
                self._shallow_until_ms = None
                return False
            if (
                front_angle >= self._t.standing_reset_knee_deg
                and back_standing_ok
            ):
                self._complete_rep()

        return shallow_triggered

    def _complete_rep(self) -> None:
        if self._reached_depth:
            self._reps += 1
        self._phase = LungePhase.STANDING
        self._front_leg = None
        self._front_ankle_x = None
        self._back_ankle_x = None
        self._reached_depth = False
        self._min_front_knee_deg = 180.0
        self._active_errors.clear()

    # ------------------------------------------------------------------
    # Form-error checks (5 checks with hysteresis and auto-clearing)
    # ------------------------------------------------------------------

    def _update_hysteresis(
        self, key: str, value: float, trigger: float, clear: float, *, above: bool
    ) -> None:
        if above:
            if value >= trigger:
                self._active_errors.add(key)
            elif value <= clear:
                self._active_errors.discard(key)
        else:
            if value <= trigger:
                self._active_errors.add(key)
            elif value >= clear:
                self._active_errors.discard(key)

    def _best_visible_side(self, pose: Pose, lm_names: tuple[str, ...]) -> Side | None:
        best_side: Side | None = None
        best_score = -1.0
        for side in ("LEFT", "RIGHT"):
            lms = tuple(_side_lm(side, name) for name in lm_names)
            if pose.visible(lms, self._t.min_visibility):
                score = min(pose.visibility(lm) for lm in lms)
                if score > best_score:
                    best_score = score
                    best_side = side
        return best_side

    def _evaluate_form(
        self,
        *,
        pose: Pose,
        pts: dict[L, np.ndarray],
        ts: int,
        front_side: Side,
        front_angle: float,
        stance_ratio: float | None,
        shallow_triggered: bool,
    ) -> tuple[list[str], dict[str, Any]]:
        debug: dict[str, Any] = {}
        active_rep = self._phase in (
            LungePhase.DESCENDING,
            LungePhase.BOTTOM,
            LungePhase.ASCENDING,
        )

        # 1. Shallow depth cue (latched briefly on shallow turnaround, clears on depth)
        if shallow_triggered:
            self._shallow_until_ms = ts + self._t.shallow_feedback_hold_ms
        elif self._shallow_until_ms is not None and ts > self._shallow_until_ms:
            self._shallow_until_ms = None

        if not active_rep:
            self._active_errors.clear()
        else:
            facing = self._facing or 1

            # 2. Front knee past toes (normalized by shin length)
            f_knee = pts[_side_lm(front_side, "KNEE")]
            f_ankle = pts[_side_lm(front_side, "ANKLE")]
            f_toe = pts.get(_side_lm(front_side, "FOOT_INDEX"))
            if f_toe is not None:
                shin_len = float(np.linalg.norm(f_knee - f_ankle))
                if shin_len > 1e-3:
                    ratio = facing * float(f_knee[0] - f_toe[0]) / shin_len
                    debug["knee_toe_ratio"] = round(ratio, 2)
                    self._update_hysteresis(
                        "knee_past_toe",
                        ratio,
                        self._t.knee_past_toe_trigger_ratio,
                        self._t.knee_past_toe_clear_ratio,
                        above=True,
                    )

            # 3. Torso lean (forward or backward relative to vertical)
            torso_side = self._best_visible_side(pose, ("SHOULDER", "HIP"))
            if torso_side is not None:
                sh = pts.get(_side_lm(torso_side, "SHOULDER"))
                hip = pts.get(_side_lm(torso_side, "HIP"))
                if sh is not None and hip is not None:
                    torso_tilt = _tilt_from_vertical_deg(hip, sh, facing)
                    debug["trunk_deg"] = round(torso_tilt, 1)
                    self._update_hysteresis(
                        "trunk_forward",
                        torso_tilt,
                        self._t.trunk_forward_trigger_deg,
                        self._t.trunk_forward_clear_deg,
                        above=True,
                    )
                    self._update_hysteresis(
                        "trunk_backward",
                        torso_tilt,
                        self._t.trunk_backward_trigger_deg,
                        self._t.trunk_backward_clear_deg,
                        above=False,
                    )

            # 4. Stance too short (horizontal ankle span / leg length)
            if stance_ratio is not None:
                if stance_ratio >= self._t.short_stance_clear_ratio:
                    self._active_errors.discard("short_stance")
                elif (
                    self._phase in (LungePhase.DESCENDING, LungePhase.BOTTOM)
                    and front_angle <= self._t.short_stance_check_knee_deg
                    and stance_ratio <= self._t.short_stance_trigger_ratio
                ):
                    self._active_errors.add("short_stance")

            # 5. Forward head posture (neck tilt relative to torso axis)
            head_side = self._best_visible_side(
                pose, ("EAR", "SHOULDER", "HIP"))
            if (
                head_side is not None
                and pose.visibility(_side_lm(head_side, "EAR"))
                >= self._t.ear_min_visibility
            ):
                ear = pts.get(_side_lm(head_side, "EAR"))
                sh = pts.get(_side_lm(head_side, "SHOULDER"))
                hip = pts.get(_side_lm(head_side, "HIP"))
                if ear is not None and sh is not None and hip is not None:
                    neck_tilt = _tilt_from_vertical_deg(sh, ear, facing)
                    base_tilt = _tilt_from_vertical_deg(hip, sh, facing)
                    fwd_head = neck_tilt - max(0.0, base_tilt)
                    debug["head_deg"] = round(fwd_head, 1)
                    self._update_hysteresis(
                        "forward_head",
                        fwd_head,
                        self._t.forward_head_trigger_deg,
                        self._t.forward_head_clear_deg,
                        above=True,
                    )

        messages: list[str] = []
        if self._shallow_until_ms is not None:
            messages.append("Lower deeper — bend front knee closer to 90°")
        if "knee_past_toe" in self._active_errors:
            messages.append("Keep front knee behind your toes")
        if "trunk_forward" in self._active_errors:
            messages.append("Keep your chest upright")
        if "trunk_backward" in self._active_errors:
            messages.append("Avoid leaning backward")
        if "short_stance" in self._active_errors:
            messages.append("Take a longer step forward")
        if "forward_head" in self._active_errors:
            messages.append("Keep your head up and look forward")
        return messages, debug
