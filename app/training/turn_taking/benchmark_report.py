from __future__ import annotations

from pydantic import Field

from app.training.turn_taking.benchmark_metrics import (
    BreakdownMetrics,
    CalibrationMetrics,
    EvaluationConfiguration,
    PolicySweepPoint,
)
from app.training.turn_taking.benchmark_models import BenchmarkModel, DetectorProvenance


class OperatingPoints(BenchmarkModel):
    best_at_300ms_latency: PolicySweepPoint | None
    best_at_600ms_latency: PolicySweepPoint | None
    best_at_5_percent_cutoff: PolicySweepPoint | None
    best_at_10_percent_cutoff: PolicySweepPoint | None


class BenchmarkAnalysisReport(BenchmarkModel):
    schema_version: str = "voice-light-causal-benchmark-analysis-v1"
    inventory_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    predictions_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    detector: DetectorProvenance
    evaluation: EvaluationConfiguration
    calibration: CalibrationMetrics | None
    sweep: tuple[PolicySweepPoint, ...]
    pareto_frontier: tuple[PolicySweepPoint, ...]
    operating_points: OperatingPoints
    selected_validation_point: PolicySweepPoint
    selected_breakdowns: tuple[BreakdownMetrics, ...]
