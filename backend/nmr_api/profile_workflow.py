"""Confidence-gated single-spectrum profiler workflow."""

from __future__ import annotations

import csv
import io
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import nmrformer_backend, signal_processing, spectral_cohort
from .profile_schema import MetaboliteResult, make_result, status_from_confidence


MODEL_VERSION = "onedTrans_0.9782"


def _prepare_processed_spectrum(raw: bytes, filename: str = "") -> Dict:
    ppm, intensity = signal_processing.parse_processed_spectrum(raw, filename)
    ppm = np.asarray(ppm, dtype=float).reshape(-1)
    y = np.asarray(intensity, dtype=float).reshape(-1)
    if len(ppm) != len(y) or len(y) < 16:
        raise ValueError("Spectrum requires equal ppm/intensity arrays with at least 16 points.")

    finite = np.isfinite(ppm) & np.isfinite(y)
    ppm, y = ppm[finite], y[finite]
    order = np.argsort(ppm)[::-1]
    ppm, y = ppm[order], y[order]

    window = (ppm >= 0.0) & (ppm <= 10.0)
    ppm, y = ppm[window], y[window]
    if len(y) < 16:
        raise ValueError("Spectrum contains too few points in the 0-10 ppm window.")

    referenced_ppm, reference_info = signal_processing.reference_to_internal_standard(ppm, y)
    corrected = signal_processing.baseline_correct(y)
    scale = float(np.max(np.abs(corrected))) or 1.0
    normalized = corrected / scale
    return {
        "ppm": referenced_ppm,
        "intensity": normalized,
        "reference": reference_info,
        "scale": scale,
        "n_points": int(len(normalized)),
    }


def qc_spectrum(raw: bytes, filename: str = "", *, snr_threshold: float = 5.0) -> Dict:
    prepared = _prepare_processed_spectrum(raw, filename)
    ppm = prepared["ppm"]
    y = prepared["intensity"]
    peaks = signal_processing.pick_peaks(ppm, y, snr=snr_threshold)
    qc = signal_processing.quality_control(
        y,
        peaks,
        reference=prepared["reference"],
    )

    quiet = np.asarray(y, dtype=float)
    baseline_residual = float(np.median(np.abs(quiet))) if quiet.size else 1.0
    baseline_score = float(np.clip(1.0 - 8.0 * baseline_residual, 0.0, 1.0))
    water = np.abs(quiet[(ppm >= 4.5) & (ppm <= 5.0)])
    water_residual = float(np.max(water)) if water.size else 0.0
    reference = prepared["reference"]
    referenced = (
        reference.get("method") == "internal standard"
        or abs(float(reference.get("offset_ppm", 0.0))) <= 0.02
    )
    snr = float(qc.get("max_snr", 0.0))

    reasons: List[str] = []
    fail = False
    if snr < snr_threshold:
        fail = True
        reasons.append("snr_below_threshold")
    elif snr < 10.0:
        reasons.append("low_snr")
    if baseline_score < 0.4:
        fail = True
        reasons.append("baseline_unstable")
    elif baseline_score < 0.7:
        reasons.append("baseline_warn")
    if water_residual > 0.65:
        fail = True
        reasons.append("water_residual_high")
    elif water_residual > 0.35:
        reasons.append("water_residual_warn")
    if qc.get("peaks_non_artifact", 0) <= 0:
        fail = True
        reasons.append("no_non_artifact_peaks")
    if not referenced:
        reasons.append("not_internally_referenced")

    verdict = "fail" if fail else "warn" if reasons else "pass"
    return {
        "snr": round(snr, 2),
        "baseline_score": round(baseline_score, 3),
        "water_residual": round(water_residual, 4),
        "referenced": bool(referenced),
        "verdict": verdict,
        "reasons": reasons,
        "quality_control": qc,
    }


def _assignments(
    ppm: np.ndarray,
    y: np.ndarray,
    peaks: Sequence[Dict],
    reference_shifts: Dict[str, List[float]],
    compound_names: Dict[str, str],
    *,
    tolerance_ppm: float,
    assignment_backend: str,
) -> Tuple[List[Dict], str, Dict]:
    pattern = signal_processing.assign_metabolites(
        peaks,
        reference_shifts,
        compound_names,
        tolerance_ppm=tolerance_ppm,
    )
    backend_status = nmrformer_backend.status()
    neural = []
    if assignment_backend in ("hybrid", "nmrformer") and backend_status["available"]:
        neural = nmrformer_backend.predict(ppm, y, peaks)
    if assignment_backend == "nmrformer" and backend_status["available"]:
        return nmrformer_backend.hybridize([], neural), "nmrformer", backend_status
    if assignment_backend == "hybrid" and backend_status["available"]:
        return (
            nmrformer_backend.hybridize(pattern, neural),
            "hybrid-pattern+nmrformer",
            backend_status,
        )
    return pattern, "reference-pattern-matcher", backend_status


def _deconvolve_single(
    ppm: np.ndarray,
    y: np.ndarray,
    reference_shifts: Dict[str, List[float]],
    *,
    fdr: float,
) -> Dict:
    frame = pd.DataFrame([np.clip(y, 0, None)], index=["sample"], columns=ppm)
    return spectral_cohort.deconvolve(
        frame,
        np.asarray(ppm, dtype=float),
        reference_shifts=reference_shifts,
        fdr=fdr,
        standard_um=None,
    )


def _peak_lookup(peaks: Sequence[Dict]) -> Dict[float, Dict]:
    return {float(peak["ppm"]): peak for peak in peaks if "ppm" in peak}


def _peaks_used(assignment: Dict, peaks: Sequence[Dict]) -> List[Dict]:
    by_ppm = _peak_lookup(peaks)
    used = []
    for match in assignment.get("matched_peaks") or []:
        ppm_value = float(match.get("observed_ppm", match.get("ppm", 0.0)))
        nearest = min(by_ppm, key=lambda value: abs(value - ppm_value)) if by_ppm else ppm_value
        peak = by_ppm.get(nearest, {})
        used.append({"ppm": ppm_value, "amp": float(peak.get("intensity", 0.0))})
    return used


def _assignment_source_and_prob(assignment: Dict) -> Tuple[str, float]:
    if assignment.get("nmrformer_confidence") is not None:
        return "nmrformer", float(assignment["nmrformer_confidence"]) / 100.0
    if assignment.get("source") == "nmrformer-only":
        return "nmrformer", float(assignment.get("confidence", 0.0)) / 100.0
    return "pattern-match", float(assignment.get("confidence", 0.0)) / 100.0


def _q_value(meta: Optional[Dict], fdr_level: float) -> float:
    if not meta:
        return 1.0
    if meta.get("passes_fdr"):
        return min(float(fdr_level), 1.0)
    snr = max(float(meta.get("snr_vs_decoy", 0.0)), 0.0)
    return float(np.clip(1.0 / (1.0 + snr), fdr_level, 1.0))


def _confidence(assignment_prob: float, fit_residual: float, q_value: float, fdr_level: float) -> float:
    fit_score = 1.0 - float(np.clip(fit_residual, 0.0, 1.0))
    fdr_score = 1.0 if q_value <= fdr_level else max(0.0, 1.0 - q_value)
    combined = 0.45 * assignment_prob + 0.35 * fit_score + 0.20 * fdr_score
    return round(float(np.clip(combined, 0.0, 1.0)), 3)


def auto_profile(
    raw: bytes,
    filename: str,
    reference_shifts: Dict[str, List[float]],
    compound_names: Dict[str, str],
    *,
    snr_threshold: float = 5.0,
    tolerance_ppm: float = 0.04,
    assignment_backend: str = "hybrid",
    fdr: float = 0.05,
    hi: float = 0.85,
    lo: float = 0.5,
    bootstrap_iterations: int = 0,
) -> Dict:
    prepared = _prepare_processed_spectrum(raw, filename)
    ppm = prepared["ppm"]
    y = prepared["intensity"]
    peaks = signal_processing.pick_peaks(ppm, y, snr=snr_threshold)
    assignments, assignment_method, backend_status = _assignments(
        ppm,
        y,
        peaks,
        reference_shifts,
        compound_names,
        tolerance_ppm=tolerance_ppm,
        assignment_backend=assignment_backend,
    )
    deconv = _deconvolve_single(ppm, y, reference_shifts, fdr=fdr)
    concentrations = deconv["concentrations"].iloc[0].to_dict()
    quantified = {item["metabolite"].casefold(): item for item in deconv["metabolites"]}
    fit_residual = round(float(np.clip(1.0 - float(deconv.get("mean_fit_r2", 0.0)), 0.0, 1.0)), 4)

    intervals = {}
    if bootstrap_iterations > 0:
        intervals = bootstrap_uncertainty(
            ppm,
            y,
            reference_shifts,
            fdr=fdr,
            iterations=bootstrap_iterations,
        )

    results: List[MetaboliteResult] = []
    for assignment in assignments:
        name = str(assignment["metabolite"])
        library_key = str(assignment.get("library_key") or name)
        concentration_value = float(
            concentrations.get(
                library_key,
                concentrations.get(name, concentrations.get(name.casefold(), 0.0)),
            )
        )
        q_value = _q_value(
            quantified.get(library_key.casefold(), quantified.get(name.casefold())),
            fdr,
        )
        source, assignment_prob = _assignment_source_and_prob(assignment)
        confidence = _confidence(assignment_prob, fit_residual, q_value, fdr)
        flags = []
        if assignment.get("ambiguity") and float(assignment["ambiguity"]) > 1.2:
            flags.append("overlap")
        if q_value > fdr:
            flags.append("fdr_not_passed")
        ci = intervals.get(library_key.casefold(), intervals.get(name.casefold(), {}))
        ci_low = ci.get("ci_low")
        ci_high = ci.get("ci_high")
        if ci_low is not None:
            ci_low = min(float(ci_low), concentration_value)
        if ci_high is not None:
            ci_high = max(float(ci_high), concentration_value)
        results.append(
            make_result(
                name=name,
                assignment_source=source,
                assignment_prob=round(float(np.clip(assignment_prob, 0.0, 1.0)), 3),
                peaks_used=_peaks_used(assignment, peaks),
                concentration_value=round(concentration_value, 5),
                concentration_unit="uM" if deconv.get("units") == "µM" else "a.u.",
                ci_low=round(ci_low, 5) if ci_low is not None else None,
                ci_high=round(ci_high, 5) if ci_high is not None else None,
                confidence=confidence,
                fdr=round(q_value, 4),
                fit_residual=fit_residual,
                model_version=MODEL_VERSION if source == "nmrformer" else "reference-shift-library",
                flags=flags,
                status=status_from_confidence(confidence, hi=hi, lo=lo),
                hi=hi,
                lo=lo,
            )
        )
    results.sort(key=lambda item: (-item.confidence, item.name.casefold()))

    return {
        "spectrum_meta": {
            "source": filename,
            "n_points": prepared["n_points"],
            "n_peaks": len(peaks),
            "assignment_method": assignment_method,
            "assignment_backend": backend_status,
            "deconvolution": {
                "method": deconv["method"],
                "units": "uM" if deconv.get("units") == "µM" else "a.u.",
                "fit_residual": fit_residual,
                "fdr_level": fdr,
                "decoy_null_level": deconv.get("decoy_null_level"),
            },
            "reference": prepared["reference"],
        },
        "metabolites": results,
    }


def bootstrap_uncertainty(
    ppm: np.ndarray,
    y: np.ndarray,
    reference_shifts: Dict[str, List[float]],
    *,
    fdr: float,
    iterations: int = 24,
    seed: int = 20260630,
) -> Dict[str, Dict[str, float]]:
    noise = max(signal_processing.estimate_noise(y), 1e-6)
    rng = np.random.default_rng(seed)
    samples: Dict[str, List[float]] = {}
    for _ in range(max(1, iterations)):
        yb = np.clip(y + rng.normal(0.0, noise, size=len(y)), 0, None)
        dec = _deconvolve_single(ppm, yb, reference_shifts, fdr=fdr)
        row = dec["concentrations"].iloc[0]
        for name, value in row.items():
            samples.setdefault(str(name).casefold(), []).append(float(value))
    intervals = {}
    for key, values in samples.items():
        arr = np.asarray(values, dtype=float)
        intervals[key] = {
            "ci_low": round(float(np.percentile(arr, 2.5)), 5),
            "ci_high": round(float(np.percentile(arr, 97.5)), 5),
        }
    return intervals


def triage(results: Sequence[MetaboliteResult], *, hi: float = 0.85, lo: float = 0.5) -> Dict:
    accepted = []
    review = []
    rejected = []
    for result in results:
        result.status = status_from_confidence(result.confidence, hi=hi, lo=lo)
        if result.status == "accept":
            accepted.append(result)
        elif result.status == "reject":
            rejected.append(result)
        else:
            review.append(result)
    return {"accepted": accepted, "review": review, "rejected": rejected}


def report_payload(
    *,
    qc: Optional[Dict],
    auto: Dict,
    triaged: Dict,
    ncd_panel: Dict,
    signed_by: Optional[str],
) -> Dict:
    signed_at = datetime.now(timezone.utc).isoformat() if signed_by else None
    return {
        "qc": qc,
        "metabolites": auto.get("metabolites", []),
        "triage": triaged,
        "ncd_panel": ncd_panel,
        "provenance": {
            "spectrum_meta": auto.get("spectrum_meta", {}),
            "workflow": "qc-gate -> auto-profile -> triage -> quantify -> ncd-panel -> report",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "signed_by": signed_by,
        "signed_at": signed_at,
    }


def metabolites_csv(results: Sequence[MetaboliteResult]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "name",
            "concentration",
            "unit",
            "ci_low",
            "ci_high",
            "confidence",
            "fdr",
            "status",
            "assignment_source",
            "assignment_prob",
            "fit_residual",
            "model_version",
            "flags",
        ]
    )
    for item in results:
        writer.writerow(
            [
                item.name,
                item.concentration.value,
                item.concentration.unit,
                item.concentration.ci_low,
                item.concentration.ci_high,
                item.confidence,
                item.fdr,
                item.status,
                item.assignment.source,
                item.assignment.prob,
                item.provenance.fit_residual,
                item.provenance.model_version,
                ";".join(item.provenance.flags),
            ]
        )
    return buffer.getvalue()
