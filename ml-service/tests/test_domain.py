"""Smoke tests for the dependency-free domain layer.

These types must stay importable with only the stdlib (no services/core
imports), so this module doubles as a guard against accidental coupling.
"""

from app.domain.comparison import ComparisonResult, SegmentResult
from app.domain.dance import Dance, Segment


def test_comparison_result_defaults_to_empty_segments():
    result = ComparisonResult(total_score=87.5)
    assert result.total_score == 87.5
    assert result.segments == []


def test_comparison_result_collects_segments():
    seg = SegmentResult(index=0, score=0.9, timing_offset=0.1, amplitude_ratio=1.0)
    result = ComparisonResult(total_score=90.0, segments=[seg])
    assert len(result.segments) == 1
    assert result.segments[0].index == 0


def test_dance_segments_and_descriptions():
    dance = Dance(
        id="d1",
        title="Test",
        segments=[Segment(index=0, start_time=0.0, end_time=1.5)],
    )
    assert dance.segments[0].description is None
    assert dance.description is None
    assert dance.segments[0].end_time == 1.5


def test_two_dances_do_not_share_default_segment_list():
    a = Dance(id="a", title="A")
    b = Dance(id="b", title="B")
    a.segments.append(Segment(index=0, start_time=0.0, end_time=1.0))
    assert b.segments == []  # field(default_factory=list) — not a shared mutable default
