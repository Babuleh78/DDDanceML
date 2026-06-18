from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Segment:
    index: int
    start_time: float
    end_time: float
    description: str | None = None


@dataclass
class Dance:
    id: str
    title: str
    segments: list[Segment] = field(default_factory=list)
    description: str | None = None
