"""Pydantic models for memory and forgetting analysis."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class RetentionType(str, Enum):
    EXACT = "exact"
    SEMANTIC = "semantic"
    PARTIAL = "partial"


class MemoryProbe(BaseModel):
    """A single memory probe with fact and expected recall."""

    probe_id: int
    fact: str
    expected_answer: str
    context_position: int = 0  # position in context where fact was placed
    category: str = "general"


class RetentionResult(BaseModel):
    """Result of a retention test for one probe."""

    probe: MemoryProbe
    recalled: bool
    retention_score: float = Field(ge=0.0, le=1.0)
    context_distance: int = 0  # tokens between fact and query
    retention_type: RetentionType = RetentionType.EXACT


class ForgettingCurvePoint(BaseModel):
    """A single point on a forgetting curve."""

    time_step: float
    retention_rate: float = Field(ge=0.0, le=1.0)
    num_probes: int = 0


class ForgettingCurve(BaseModel):
    """A complete forgetting curve."""

    points: list[ForgettingCurvePoint] = Field(default_factory=list)
    stability: float = 1.0  # Ebbinghaus S parameter
    initial_strength: float = 1.0  # Ebbinghaus B parameter
    model_type: str = "ebbinghaus"
    r_squared: float = 0.0


class InterferenceResult(BaseModel):
    """Result of an interference test."""

    baseline_retention: float
    post_interference_retention: float
    interference_type: str  # "proactive" or "retroactive"
    interference_magnitude: float = 0.0
    num_interfering_items: int = 0


class UnlearningResult(BaseModel):
    """Result of a knowledge unlearning attempt."""

    target_fact: str
    pre_unlearn_confidence: float
    post_unlearn_confidence: float
    collateral_damage: float = 0.0  # drop in unrelated knowledge
    num_gradient_steps: int = 0
    success: bool = False


class ForgetNetReport(BaseModel):
    """Complete analysis report."""

    retention_results: list[RetentionResult] = Field(default_factory=list)
    forgetting_curve: Optional[ForgettingCurve] = None
    interference_results: list[InterferenceResult] = Field(default_factory=list)
    unlearning_results: list[UnlearningResult] = Field(default_factory=list)
