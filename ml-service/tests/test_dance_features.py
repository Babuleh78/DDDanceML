"""Unit tests for the difficulty scoring helpers (pure numpy/scipy logic)."""

import pytest

from app.services.dance_features import _norm, compute_difficulty


@pytest.mark.parametrize(
    "x, lo, hi, expected",
    [
        (5.0, 0.0, 10.0, 0.5),
        (-1.0, 0.0, 10.0, 0.0),   # clipped below
        (20.0, 0.0, 10.0, 1.0),   # clipped above
        (3.0, 3.0, 3.0, 0.0),     # degenerate range -> 0
        (1.0, 5.0, 0.0, 0.0),     # hi <= lo guard
    ],
)
def test_norm_clamps_to_unit_interval(x, lo, hi, expected):
    assert _norm(x, lo, hi) == pytest.approx(expected)


def test_difficulty_falls_back_to_medium_on_empty_features():
    out = compute_difficulty({}, duration_sec=60.0)
    assert out == {"difficulty_score": 50, "difficulty_label": "medium"}


def test_difficulty_score_is_bounded_and_labelled():
    # Deliberately extreme feature values — score must still stay in [0, 100].
    features = {
        "angular_velocity_mean": 999.0,
        "angular_velocity_max": 999.0,
        "moving_limbs_avg_count": 6.0,
        "pose_entropy": 99.0,
        "jumps": 50,
        "rotations": 50,
        "com_dispersion": 99.0,
    }
    out = compute_difficulty(features, duration_sec=30.0)
    assert 0 <= out["difficulty_score"] <= 100
    assert out["difficulty_label"] in {"easy", "medium", "hard"}


def test_low_activity_dance_is_easier_than_high_activity():
    calm = {
        "angular_velocity_mean": 0.1,
        "moving_limbs_avg_count": 1.0,
        "pose_entropy": 0.5,
    }
    intense = {
        "angular_velocity_mean": 5.0,
        "angular_velocity_max": 10.0,
        "moving_limbs_avg_count": 6.0,
        "pose_entropy": 5.0,
        "jumps": 10,
        "rotations": 10,
        "com_dispersion": 2.0,
    }
    assert (
        compute_difficulty(calm, 60.0)["difficulty_score"]
        < compute_difficulty(intense, 60.0)["difficulty_score"]
    )
