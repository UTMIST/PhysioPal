"""Unit tests for the lateral-view lunge exercise."""

from __future__ import annotations

import math

from physiopal.exercises.lunge import Lunge, LungePhase
from physiopal.pose import L, Landmark, Pose

from .conftest import make_pose, standing_pose

_THIGH_LEN = 180.0
_SHIN_LEN = 180.0
_TORSO_LEN = 200.0
_NECK_LEN = 80.0


def _lunge_pose(
    *,
    front_leg: str = "LEFT",
    facing: int = 1,
    front_knee_deg: float = 175.0,
    back_knee_deg: float = 175.0,
    knee_forward_px: float = 0.0,
    trunk_deg: float = 0.0,
    head_deg: float = 0.0,
    swap_legs: bool = False,
    timestamp_ms: int = 0,
    visibility: float = 0.99,
) -> Pose:
    """Build a synthetic lateral-view Pose with exact knee/trunk/head angles."""
    back_leg = "RIGHT" if front_leg == "LEFT" else "LEFT"
    if swap_legs:
        front_leg, back_leg = back_leg, front_leg

    hip_x, hip_y = 500.0, 500.0
    alpha = math.radians(180.0 - front_knee_deg)
    beta = math.radians(180.0 - back_knee_deg)

    # Front leg: thigh angled forward by alpha, shin angled by knee_forward_px
    f_knee_x = hip_x + facing * _THIGH_LEN * math.sin(alpha)
    f_knee_y = hip_y + _THIGH_LEN * math.cos(alpha)
    f_ankle_x = f_knee_x - facing * knee_forward_px
    f_ankle_y = f_knee_y + _SHIN_LEN

    # Back leg: thigh vertical down, shin angled back by beta
    b_knee_x = hip_x
    b_knee_y = hip_y + _THIGH_LEN
    b_ankle_x = b_knee_x - facing * _SHIN_LEN * math.sin(beta)
    b_ankle_y = b_knee_y + _SHIN_LEN * math.cos(beta)

    # Upper body: trunk tilted by trunk_deg, neck tilted by trunk_deg + head_deg
    trunk_rad = math.radians(trunk_deg)
    sh_x = hip_x + facing * _TORSO_LEN * math.sin(trunk_rad)
    sh_y = hip_y - _TORSO_LEN * math.cos(trunk_rad)

    neck_rad = math.radians(trunk_deg + head_deg)
    ear_x = sh_x + facing * _NECK_LEN * math.sin(neck_rad)
    ear_y = sh_y - _NECK_LEN * math.cos(neck_rad)
    nose_x = ear_x + facing * 30.0
    nose_y = ear_y + 10.0

    pts: dict[L, tuple[float, float]] = {
        L[f"{front_leg}_HIP"]: (hip_x, hip_y),
        L[f"{front_leg}_KNEE"]: (f_knee_x, f_knee_y),
        L[f"{front_leg}_ANKLE"]: (f_ankle_x, f_ankle_y),
        L[f"{front_leg}_HEEL"]: (f_ankle_x - facing * 15.0, f_ankle_y + 15.0),
        L[f"{front_leg}_FOOT_INDEX"]: (f_ankle_x + facing * 45.0, f_ankle_y + 15.0),
        L[f"{back_leg}_HIP"]: (hip_x, hip_y),
        L[f"{back_leg}_KNEE"]: (b_knee_x, b_knee_y),
        L[f"{back_leg}_ANKLE"]: (b_ankle_x, b_ankle_y),
        L[f"{back_leg}_HEEL"]: (b_ankle_x - facing * 15.0, b_ankle_y + 15.0),
        L[f"{back_leg}_FOOT_INDEX"]: (b_ankle_x + facing * 45.0, b_ankle_y + 15.0),
        L.LEFT_SHOULDER: (sh_x, sh_y),
        L.RIGHT_SHOULDER: (sh_x, sh_y),
        L.LEFT_EAR: (ear_x, ear_y),
        L.RIGHT_EAR: (ear_x, ear_y),
        L.NOSE: (nose_x, nose_y),
    }
    return make_pose(pts, visibility=visibility, timestamp_ms=timestamp_ms)


def _short_stance_pose(*, timestamp_ms: int) -> Pose:
    """Build a split-stance lunge pose whose ankle span is ~0.55x leg length."""
    hip = (500.0, 500.0)
    f_knee = (673.9, 546.6)
    f_ankle = (673.9, 726.6)  # front knee = 105 deg, leg_len = 360
    b_knee = (530.0, 677.0)
    b_ankle = (476.0, 848.0)  # ankle span = 197.9 -> stance_ratio ~ 0.55
    pts: dict[L, tuple[float, float]] = {
        L.LEFT_HIP: hip,
        L.LEFT_KNEE: f_knee,
        L.LEFT_ANKLE: f_ankle,
        L.LEFT_HEEL: (f_ankle[0] - 15.0, f_ankle[1] + 15.0),
        L.LEFT_FOOT_INDEX: (f_ankle[0] + 45.0, f_ankle[1] + 15.0),
        L.RIGHT_HIP: hip,
        L.RIGHT_KNEE: b_knee,
        L.RIGHT_ANKLE: b_ankle,
        L.RIGHT_HEEL: (b_ankle[0] - 15.0, b_ankle[1] + 15.0),
        L.RIGHT_FOOT_INDEX: (b_ankle[0] + 45.0, b_ankle[1] + 15.0),
        L.LEFT_SHOULDER: (500.0, 300.0),
        L.RIGHT_SHOULDER: (500.0, 300.0),
        L.LEFT_EAR: (500.0, 220.0),
        L.RIGHT_EAR: (500.0, 220.0),
        L.NOSE: (530.0, 230.0),
    }
    return make_pose(pts, timestamp_ms=timestamp_ms)


def _squat_pose(*, knee_x_offset: float, ankle_sep: float, timestamp_ms: int) -> Pose:
    """Build a side/45-deg squat pose (both knees bend equally at the same height)."""
    hip = (500.0, 620.0)
    l_knee = (500.0 + knee_x_offset, 640.0)
    r_knee = (500.0 + knee_x_offset - 5.0, 642.0)
    l_ankle = (520.0 + ankle_sep / 2.0, 800.0)
    r_ankle = (520.0 - ankle_sep / 2.0, 800.0)
    pts: dict[L, tuple[float, float]] = {
        L.LEFT_HIP: hip,
        L.RIGHT_HIP: hip,
        L.LEFT_KNEE: l_knee,
        L.RIGHT_KNEE: r_knee,
        L.LEFT_ANKLE: l_ankle,
        L.RIGHT_ANKLE: r_ankle,
        L.LEFT_HEEL: (l_ankle[0] - 15.0, l_ankle[1] + 15.0),
        L.LEFT_FOOT_INDEX: (l_ankle[0] + 45.0, l_ankle[1] + 15.0),
        L.RIGHT_HEEL: (r_ankle[0] - 15.0, r_ankle[1] + 15.0),
        L.RIGHT_FOOT_INDEX: (r_ankle[0] + 45.0, r_ankle[1] + 15.0),
        L.LEFT_SHOULDER: (520.0, 420.0),
        L.RIGHT_SHOULDER: (520.0, 420.0),
    }
    return make_pose(pts, timestamp_ms=timestamp_ms)


def _run_rep(
    ex: Lunge,
    *,
    start_ms: int,
    front_leg: str = "LEFT",
    facing: int = 1,
    min_angle: float = 90.0,
    swap_at_bottom: bool = False,
) -> tuple[list[str], int, list[list[str]]]:
    """Drive a lunge sequence (dt=600ms skips smoothing lag for exact angle tests)."""
    angles = [175.0, 140.0, 115.0, min_angle, 120.0, 145.0, 175.0]
    states: list[str] = []
    feedbacks: list[list[str]] = []
    ts = start_ms
    for idx, angle in enumerate(angles):
        swap = swap_at_bottom and idx == 3
        pose = _lunge_pose(
            front_leg=front_leg,
            facing=facing,
            front_knee_deg=angle,
            back_knee_deg=angle,
            swap_legs=swap,
            timestamp_ms=ts,
        )
        res = ex.step(pose)
        states.append(res.state)
        feedbacks.append(res.feedback)
        ts += 600
    return states, res.reps, feedbacks


def test_full_lunge_counts_once_and_tracks_phases() -> None:
    ex = Lunge()
    states, reps, _ = _run_rep(ex, start_ms=0, front_leg="LEFT", facing=1)

    assert reps == 1
    assert states[0] == LungePhase.STANDING.value
    assert LungePhase.DESCENDING.value in states
    assert LungePhase.BOTTOM.value in states
    assert LungePhase.ASCENDING.value in states
    assert states[-1] == LungePhase.STANDING.value


def test_alternating_legs_and_both_facing_directions_count_reps() -> None:
    ex = Lunge()
    _, reps_1, _ = _run_rep(ex, start_ms=0, front_leg="LEFT", facing=1)
    _, reps_2, _ = _run_rep(ex, start_ms=5000, front_leg="RIGHT", facing=1)
    _, reps_3, _ = _run_rep(ex, start_ms=10000, front_leg="LEFT", facing=-1)

    assert (reps_1, reps_2, reps_3) == (1, 2, 3)


def test_mid_rep_landmark_swap_still_counts_rep() -> None:
    ex = Lunge()
    _, reps, _ = _run_rep(
        ex, start_ms=0, front_leg="LEFT", facing=1, swap_at_bottom=True
    )
    assert reps == 1


def test_shallow_lunge_rejected_and_emits_depth_feedback_that_clears() -> None:
    ex = Lunge()
    _, reps, feedbacks = _run_rep(ex, start_ms=0, min_angle=120.0)

    assert reps == 0
    assert any("Lower deeper" in msg for fb in feedbacks for msg in fb)

    # Feedback clears once hold duration elapses (timestamp-based, > 1500 ms)
    later = ex.step(_lunge_pose(front_knee_deg=175.0,
                    back_knee_deg=175.0, timestamp_ms=10000))
    assert not any("Lower deeper" in msg for msg in later.feedback)


def test_squat_and_stance_shift_do_not_count_reps() -> None:
    ex = Lunge()
    ts = 0

    # Deep squat (both stacked and wide 45-deg stance)
    for ankle_sep in (5.0, 180.0):
        ex.step(standing_pose(timestamp_ms=ts))
        ts += 600
        ex.step(_squat_pose(knee_x_offset=150.0,
                ankle_sep=ankle_sep, timestamp_ms=ts))
        ts += 600
        res = ex.step(standing_pose(timestamp_ms=ts))
        ts += 600
        assert res.reps == 0

    # Slight stance shift (knee only bends to 152 deg)
    ex.step(_lunge_pose(front_knee_deg=152.0,
            back_knee_deg=152.0, timestamp_ms=ts))
    res = ex.step(_lunge_pose(front_knee_deg=175.0,
                  back_knee_deg=175.0, timestamp_ms=ts + 600))
    assert res.reps == 0


def test_knee_past_toe_trunk_lean_and_forward_head_trigger_and_clear() -> None:
    ex = Lunge()
    ex.step(_lunge_pose(front_knee_deg=175.0,
            back_knee_deg=175.0, timestamp_ms=0))
    ex.step(_lunge_pose(front_knee_deg=140.0,
            back_knee_deg=140.0, timestamp_ms=600))

    # Trigger 3 form errors simultaneously at bottom
    bad = ex.step(
        _lunge_pose(
            front_knee_deg=90.0,
            back_knee_deg=90.0,
            knee_forward_px=95.0,
            trunk_deg=28.0,
            head_deg=55.0,
            timestamp_ms=1200,
        )
    )
    assert "Keep front knee behind your toes" in bad.feedback
    assert "Keep your chest upright" in bad.feedback
    assert "Keep your head up and look forward" in bad.feedback

    # Correct all 3 errors mid-lunge -> all messages clear immediately
    fixed = ex.step(
        _lunge_pose(
            front_knee_deg=90.0,
            back_knee_deg=90.0,
            knee_forward_px=0.0,
            trunk_deg=5.0,
            head_deg=5.0,
            timestamp_ms=1800,
        )
    )
    assert fixed.feedback == []


def test_short_stance_triggers_and_clears_when_widened() -> None:
    ex = Lunge()
    ex.step(_lunge_pose(front_knee_deg=175.0,
            back_knee_deg=175.0, timestamp_ms=0))

    short_res = ex.step(_short_stance_pose(timestamp_ms=600))
    assert "Take a longer step forward" in short_res.feedback

    # Widening stance (even while knee opens past 135 deg) clears feedback right away
    widened = ex.step(
        _lunge_pose(front_knee_deg=140.0,
                    back_knee_deg=120.0, timestamp_ms=1200)
    )
    assert "Take a longer step forward" not in widened.feedback


def test_side_view_far_hip_occlusion_still_counts_rep() -> None:
    """At a 90-deg side view, occluded far hip should not trigger NO_POSE."""
    ex = Lunge()
    ts = 0
    for angle in (175.0, 140.0, 90.0, 130.0, 175.0):
        base = _lunge_pose(front_knee_deg=angle,
                           back_knee_deg=angle, timestamp_ms=ts)
        lms = list(base.landmarks)
        # Simulate far-side (RIGHT) hip fully occluded, far knee/ankle partially visible
        rh = lms[L.RIGHT_HIP]
        rk = lms[L.RIGHT_KNEE]
        ra = lms[L.RIGHT_ANKLE]
        lms[L.RIGHT_HIP] = Landmark(
            rh.x, rh.y, rh.z, visibility=0.1, presence=0.1)
        lms[L.RIGHT_KNEE] = Landmark(
            rk.x, rk.y, rk.z, visibility=0.4, presence=0.4)
        lms[L.RIGHT_ANKLE] = Landmark(
            ra.x, ra.y, ra.z, visibility=0.4, presence=0.4)
        occluded_pose = Pose(tuple(lms), base.world,
                             base.width, base.height, ts)
        res = ex.step(occluded_pose)
        assert res.state != LungePhase.NO_POSE.value
        ts += 600

    assert res.reps == 1


def test_low_visibility_and_no_person_frames_are_safe() -> None:
    ex = Lunge()
    # Start descending
    ex.step(_lunge_pose(front_knee_deg=175.0,
            back_knee_deg=175.0, timestamp_ms=0))
    ex.step(_lunge_pose(front_knee_deg=90.0,
            back_knee_deg=90.0, timestamp_ms=600))

    # Empty / invisible frames beyond abort timeout (1500 ms)
    low_vis_1 = ex.step(make_pose({}, timestamp_ms=1200))
    low_vis_2 = ex.step(make_pose({}, timestamp_ms=3000))
    assert low_vis_1.state == LungePhase.NO_POSE.value
    assert low_vis_2.state == LungePhase.NO_POSE.value
    assert low_vis_2.reps == 0

    # Stepping back into frame standing must not award a phantom rep
    back = ex.step(_lunge_pose(front_knee_deg=175.0,
                   back_knee_deg=175.0, timestamp_ms=3600))
    assert back.reps == 0
    assert back.state == LungePhase.STANDING.value

    # update(None) path from Exercise base class
    none_res = ex.update(None)
    assert none_res.reps == 0


def test_reset_clears_all_state() -> None:
    ex = Lunge()
    _run_rep(ex, start_ms=0)
    assert ex.update(None).reps == 1
    # Enter a second rep with active form error
    ex.step(_lunge_pose(front_knee_deg=140.0,
            back_knee_deg=140.0, timestamp_ms=5000))
    ex.step(
        _lunge_pose(
            front_knee_deg=90.0,
            back_knee_deg=90.0,
            trunk_deg=30.0,
            timestamp_ms=5600,
        )
    )

    ex.reset()
    res = ex.step(standing_pose(timestamp_ms=6200))
    assert res.reps == 0
    assert res.state == LungePhase.STANDING.value
    assert res.feedback == []
