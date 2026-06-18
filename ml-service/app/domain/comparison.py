from dataclasses import dataclass, field


@dataclass
class SegmentResult:
    index: int
    score: float
    timing_offset: float
    amplitude_ratio: float


@dataclass
class ComparisonResult:
    total_score: float
    segments: list[SegmentResult] = field(default_factory=list)
