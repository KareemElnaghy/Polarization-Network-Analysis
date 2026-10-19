"""Shared data contracts for the polarization project."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True)
class Window:
    """A half-open time interval used to group graph interactions."""

    start: datetime
    end: datetime

    def __post_init__(self) -> None:
        if self.end <= self.start:
            raise ValueError("Window end must be later than its start")


@dataclass(frozen=True)
class SignedEdge:
    """A directed interaction with a sign and optional confidence."""

    source: str
    target: str
    sign: int
    timestamp: datetime | None = None
    confidence: float | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sign not in (-1, 1):
            raise ValueError("SignedEdge sign must be -1 or 1")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("SignedEdge confidence must be between 0 and 1")


@dataclass(frozen=True)
class GraphWindow:
    """A graph and the time window and metadata used to construct it."""

    window: Window
    graph: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineScore:
    """A polarization score produced by a pipeline for one graph window."""

    window: Window
    score: float
    pipeline_name: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
