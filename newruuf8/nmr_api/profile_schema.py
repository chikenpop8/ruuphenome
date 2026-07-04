"""Shared schema for the confidence-gated profiler workflow."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence

from pydantic import BaseModel, Field


ProfileStatus = Literal["accept", "review", "reject"]
ConcentrationUnit = Literal["uM", "a.u."]


class AssignmentEvidence(BaseModel):
    source: str = Field(..., description="Assignment backend, e.g. nmrformer or pattern-match")
    prob: float = Field(..., ge=0.0, le=1.0, description="Assignment probability")


class PeakUsed(BaseModel):
    ppm: float
    amp: float


class ConcentrationEstimate(BaseModel):
    value: float
    unit: ConcentrationUnit
    ci_low: Optional[float] = None
    ci_high: Optional[float] = None


class ResultProvenance(BaseModel):
    deconv: str = "nnls"
    fit_residual: float = Field(..., ge=0.0)
    model_version: str
    flags: List[str] = Field(default_factory=list)


class MetaboliteResult(BaseModel):
    name: str
    assignment: AssignmentEvidence
    peaks_used: List[PeakUsed]
    concentration: ConcentrationEstimate
    confidence: float = Field(..., ge=0.0, le=1.0)
    fdr: float = Field(..., ge=0.0, le=1.0)
    status: ProfileStatus
    provenance: ResultProvenance


def status_from_confidence(
    confidence: float,
    *,
    hi: float = 0.85,
    lo: float = 0.5,
) -> ProfileStatus:
    if confidence >= hi:
        return "accept"
    if confidence < lo:
        return "reject"
    return "review"


def make_result(
    *,
    name: str,
    assignment_source: str,
    assignment_prob: float,
    peaks_used: Sequence[Dict[str, Any] | PeakUsed],
    concentration_value: float,
    concentration_unit: ConcentrationUnit = "a.u.",
    ci_low: Optional[float] = None,
    ci_high: Optional[float] = None,
    confidence: float,
    fdr: float,
    fit_residual: float,
    model_version: str,
    deconv: str = "nnls",
    flags: Optional[Sequence[str]] = None,
    status: Optional[ProfileStatus] = None,
    hi: float = 0.85,
    lo: float = 0.5,
) -> MetaboliteResult:
    return MetaboliteResult(
        name=name,
        assignment=AssignmentEvidence(
            source=assignment_source,
            prob=assignment_prob,
        ),
        peaks_used=list(peaks_used),
        concentration=ConcentrationEstimate(
            value=concentration_value,
            unit=concentration_unit,
            ci_low=ci_low,
            ci_high=ci_high,
        ),
        confidence=confidence,
        fdr=fdr,
        status=status or status_from_confidence(confidence, hi=hi, lo=lo),
        provenance=ResultProvenance(
            deconv=deconv,
            fit_residual=fit_residual,
            model_version=model_version,
            flags=list(flags or []),
        ),
    )


__all__ = [
    "AssignmentEvidence",
    "ConcentrationEstimate",
    "MetaboliteResult",
    "PeakUsed",
    "ResultProvenance",
    "make_result",
    "status_from_confidence",
]
