"""Pydantic response/request models for the NMR API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ObservedPeak(BaseModel):
    """A single annotated peak from a processed spectrum (Domain 1)."""

    metabolite: str = Field(..., description="Annotated metabolite label")
    observed_shift: float = Field(..., description="Chemical shift in ppm")
    region: Optional[str] = Field(
        default=None,
        description="Spectral region: aliphatic | oxygenated | anomeric | aromatic",
    )


class MetaboliteMatch(BaseModel):
    """Scoring of one Domain 2 candidate against the observed peak list."""

    metabolite: str
    chebi_id: str
    chemical_formula: str
    smiles: str
    predicted_shifts: List[float]
    n_shifts_predicted: int
    peaks_matched: int
    match_score: float = Field(..., description="0-100, % of predicted shifts hit")
    matched_detail: str
    mean_abundance: Optional[float] = None
    cv_percent: Optional[float] = None
    detected_percent: Optional[float] = None


class AnalysisSummary(BaseModel):
    total_metabolites: int
    total_samples: int
    metabolites_with_smiles: int
    prediction_backend: str = Field(
        ..., description="NMRTransformer or HMDB-fallback"
    )
    tolerance_ppm: float


class AnalysisResponse(BaseModel):
    summary: AnalysisSummary
    matches: List[MetaboliteMatch]


class HealthResponse(BaseModel):
    ok: bool
    backend: str
    nmrtransformer_available: bool
    notes: str


class LaboratoryQCRequest(BaseModel):
    """Spectrum/batch observations used by the laboratory release gate."""

    qc_score: Optional[float] = Field(default=None, ge=0, le=100)
    max_snr: Optional[float] = Field(default=None, ge=0)
    negative_area_fraction: Optional[float] = Field(default=None, ge=0, le=1)
    reference_method: Optional[str] = None
    pooled_qc_cv_percent: Optional[float] = Field(default=None, ge=0)
    drift_percent: Optional[float] = Field(default=None, ge=0)
    blank_contamination: Optional[bool] = None
    instrument_suitability_passed: Optional[bool] = None
    sample_identity_verified: Optional[bool] = None
