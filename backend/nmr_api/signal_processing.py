"""
Domain 1 — robust 1D ¹H NMR processing and metabolite assignment.

The public API remains compatible with the original prototype (``ppm``,
``intensity`` and ``peaks`` are still returned), but the processing is now
closer to a defensible metabolomics workflow:

  Bruker FID
    → digital-filter removal
    → DC correction + apodization + power-of-two zero filling
    → FFT + automatic zero/first-order phase correction
    → acquisition-aware ppm axis
    → asymmetric least-squares baseline correction
    → robust MAD noise estimation
    → prominence/width-aware peak picking
    → reference-library pattern matching with confidence and ambiguity
    → quality-control metrics

NMRformer can eventually replace the reference-pattern matcher. Until its
weights/runtime are available, this module provides a transparent classical
baseline rather than pretending a deep-learning model is active.
"""

from __future__ import annotations

import io
import math
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import sparse
from scipy.signal import find_peaks, peak_widths
from scipy.sparse.linalg import spsolve

from . import nmrformer_backend

try:
    import nmrglue as ng

    NMRGLUE = True
except Exception:  # pragma: no cover - processing still works for numeric data
    ng = None
    NMRGLUE = False


DEFAULT_ARTIFACT_REGIONS = (
    {"name": "residual water", "min_ppm": 4.50, "max_ppm": 5.00},
)
MAX_ZIP_UNCOMPRESSED_BYTES = 1_000_000_000
MAX_ZIP_MEMBERS = 20_000


# ── Low-level processing ────────────────────────────────────────────────────
def dc_correct(fid: np.ndarray, tail_fraction: float = 0.1) -> np.ndarray:
    """Remove the receiver DC offset using the quiet tail of the FID."""
    fid = np.asarray(fid, dtype=complex)
    n_tail = max(8, int(len(fid) * tail_fraction))
    return fid - np.mean(fid[-n_tail:])


def apodize(fid: np.ndarray, lb_hz: float, sw_hz: float) -> np.ndarray:
    """Apply exponential line broadening to improve signal-to-noise."""
    if sw_hz <= 0:
        raise ValueError("Sweep width must be positive.")
    t = np.arange(len(fid), dtype=float) / sw_hz
    return np.asarray(fid) * np.exp(-max(0.0, lb_hz) * np.pi * t)


def zerofill_fft(fid: np.ndarray, factor: int = 2) -> np.ndarray:
    """Zero-fill to a power of two, then Fourier transform."""
    if factor < 1:
        raise ValueError("Zero-fill factor must be at least 1.")
    target = 1 << int(math.ceil(math.log2(max(2, len(fid)))))
    target *= int(factor)
    return np.fft.fftshift(np.fft.fft(fid, target))


def _apply_linear_phase(spec: np.ndarray, p0: float, p1: float) -> np.ndarray:
    """Apply zero/first-order phase in degrees using the nmrglue convention."""
    if NMRGLUE:
        return ng.proc_base.ps(np.asarray(spec), p0=p0, p1=p1)
    phase = np.deg2rad(p0 + p1 * np.arange(len(spec)) / max(1, len(spec) - 1))
    return np.asarray(spec) * np.exp(1j * phase)


def auto_phase(
    spec: np.ndarray,
    *,
    return_phases: bool = False,
    max_points: int = 8192,
) -> np.ndarray | Tuple[np.ndarray, Dict[str, float]]:
    """
    Automatic zero- and first-order phase correction.

    ACME is used when nmrglue is installed. A deterministic zero-order search
    remains available for numeric-spectrum use without nmrglue.
    """
    spec = np.asarray(spec, dtype=complex)
    if NMRGLUE:
        stride = max(1, len(spec) // max_points)
        sample = spec[::stride]
        try:
            _sample_phased, phases = ng.process.proc_autophase.autops(
                sample,
                "acme",
                return_phases=True,
                disp=False,
                maxiter=500,
                ftol=1e-7,
            )
            p0, p1 = float(phases[0]), float(phases[1])
            phased = _apply_linear_phase(spec, p0, p1)
        except Exception:
            phased, p0, p1 = _zero_order_phase(spec)
    else:
        phased, p0, p1 = _zero_order_phase(spec)

    # A correct absorption-mode spectrum should carry most area above zero.
    real = np.real(phased)
    if np.sum(np.clip(real, 0, None)) < np.sum(np.clip(-real, 0, None)):
        phased = -phased
        p0 += 180.0

    result = (phased, {"p0_degrees": round(p0, 3), "p1_degrees": round(p1, 3)})
    return result if return_phases else phased


def _zero_order_phase(spec: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Fallback phase search minimizing negative spectral area."""
    best_phase, best_score = 0.0, np.inf
    for p0 in np.linspace(-180.0, 180.0, 361):
        real = np.real(_apply_linear_phase(spec, p0, 0.0))
        score = np.sum(np.clip(-real, 0, None)) / (np.sum(np.abs(real)) + 1e-12)
        if score < best_score:
            best_phase, best_score = float(p0), float(score)
    return _apply_linear_phase(spec, best_phase, 0.0), best_phase, 0.0


def baseline_correct(
    y: np.ndarray,
    lam: float = 1e6,
    asymmetry: float = 0.001,
    iterations: int = 10,
) -> np.ndarray:
    """
    Asymmetric least-squares baseline correction.

    This handles sloped and curved baselines without using peak-rich regions as
    the baseline, unlike the previous rolling-minimum approximation.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        return y - np.median(y)
    asymmetry = float(np.clip(asymmetry, 1e-6, 1 - 1e-6))
    difference = sparse.diags(
        [np.ones(len(y) - 2), -2 * np.ones(len(y) - 2), np.ones(len(y) - 2)],
        [0, 1, 2],
        shape=(len(y) - 2, len(y)),
        format="csc",
    )
    penalty = float(lam) * (difference.T @ difference)
    weights = np.ones(len(y))
    baseline = np.zeros_like(y)
    for _ in range(max(1, int(iterations))):
        weight_matrix = sparse.spdiags(weights, 0, len(y), len(y))
        baseline = spsolve(weight_matrix + penalty, weights * y)
        weights = asymmetry * (y > baseline) + (1 - asymmetry) * (y <= baseline)
    corrected = y - baseline
    return corrected - np.median(corrected)


def estimate_noise(y: np.ndarray) -> float:
    """
    Robust noise σ from first differences.

    Differencing suppresses slowly varying baselines and the MAD prevents a few
    strong peaks from dominating the estimate.
    """
    y = np.asarray(y, dtype=float)
    if len(y) < 3:
        return float(np.std(y))
    differences = np.diff(y)
    median = np.median(differences)
    mad = np.median(np.abs(differences - median))
    sigma = 1.4826 * mad / math.sqrt(2.0)
    if sigma <= 0:
        sigma = float(np.std(differences) / math.sqrt(2.0))
    # Floor the estimate relative to the signal amplitude. A clean or synthetic,
    # essentially noise-free spectrum drives σ→0 and would report SNR in the
    # millions. Real ¹H noise is ≳0.01% of peak height (SNR ≲ 1e4), so keeping
    # σ ≥ 1e-4 × peak height caps a degenerate SNR at a believable "excellent"
    # value without ever binding on a genuine experimental spectrum.
    amplitude = float(np.max(np.abs(y)))
    return float(max(sigma, 1e-4 * amplitude))


def _region_for_ppm(ppm: float, regions: Sequence[Dict]) -> Optional[str]:
    for region in regions:
        if float(region["min_ppm"]) <= ppm <= float(region["max_ppm"]):
            return str(region["name"])
    return None


def reference_to_internal_standard(
    ppm: np.ndarray,
    y: np.ndarray,
    *,
    expected_ppm: float = 0.0,
    search_min_ppm: float = -0.5,
    search_max_ppm: float = 0.5,
) -> Tuple[np.ndarray, Dict]:
    """
    Reference DSS/TSP/TMS to its expected position.

    A narrow, high-SNR peak must be present near zero; otherwise acquisition
    metadata is retained. This conservative guard avoids referencing ordinary
    metabolite peaks as an internal standard.
    """
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (ppm >= search_min_ppm) & (ppm <= search_max_ppm)
    if mask.sum() < 16:
        return ppm, {"method": "acquisition metadata", "offset_ppm": 0.0}
    local_ppm, local_y = ppm[mask], y[mask]
    noise = max(estimate_noise(y), 1e-12)
    spacing = max(float(np.median(np.abs(np.diff(local_ppm)))), 1e-12)
    indices, props = find_peaks(
        local_y,
        height=20 * noise,
        prominence=10 * noise,
        width=(2, max(3, int(0.03 / spacing))),
    )
    if not len(indices):
        return ppm, {"method": "acquisition metadata", "offset_ppm": 0.0}
    widths = peak_widths(local_y, indices, rel_height=0.5)[0] * spacing
    scores = props["prominences"] / np.maximum(widths, spacing)
    best = int(np.argmax(scores))
    observed = float(local_ppm[indices[best]])
    offset = float(expected_ppm - observed)
    if abs(offset) > 0.5:
        return ppm, {"method": "acquisition metadata", "offset_ppm": 0.0}
    return ppm + offset, {
        "method": "internal standard",
        "standard": "DSS/TSP/TMS",
        "observed_ppm": round(observed, 5),
        "target_ppm": round(float(expected_ppm), 5),
        "offset_ppm": round(offset, 5),
        "snr": round(float(local_y[indices[best]] / noise), 1),
    }


# ── Peak picking and assignment ─────────────────────────────────────────────
def pick_peaks(
    ppm: np.ndarray,
    y: np.ndarray,
    snr: float = 5.0,
    *,
    min_prominence_snr: float = 3.0,
    min_distance_ppm: float = 0.003,
    artifact_regions: Sequence[Dict] = DEFAULT_ARTIFACT_REGIONS,
) -> List[Dict]:
    """Pick positive absorption peaks with SNR, prominence, width and area."""
    ppm = np.asarray(ppm, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(ppm) != len(y) or len(y) < 3:
        return []

    noise = max(estimate_noise(y), 1e-12)
    spacing = max(float(np.median(np.abs(np.diff(ppm)))), 1e-12)
    distance_points = max(1, int(round(min_distance_ppm / spacing)))
    indices, properties = find_peaks(
        y,
        height=float(snr) * noise,
        prominence=max(float(min_prominence_snr), float(snr)) * noise,
        distance=distance_points,
        width=(2.0, None),
    )
    if not len(indices):
        return []

    widths, _height, left_ips, right_ips = peak_widths(y, indices, rel_height=0.5)
    peaks: List[Dict] = []
    for position, index in enumerate(indices):
        left = max(0, int(math.floor(left_ips[position])))
        right = min(len(y) - 1, int(math.ceil(right_ips[position])))
        peak_ppm = float(ppm[index])
        area = float(abs(np.trapezoid(y[left : right + 1], ppm[left : right + 1])))
        artifact = _region_for_ppm(peak_ppm, artifact_regions)
        peaks.append(
            {
                "ppm": round(peak_ppm, 4),
                "intensity": round(float(y[index]), 6),
                "snr": round(float(y[index] / noise), 1),
                "prominence": round(float(properties["prominences"][position]), 6),
                "fwhm_ppm": round(float(widths[position] * spacing), 5),
                "area": round(area, 8),
                "artifact": artifact,
            }
        )
    peaks.sort(key=lambda peak: (-peak["intensity"], peak["ppm"]))
    return peaks


def assign_metabolites(
    peaks: Sequence[Dict],
    reference_shifts: Dict[str, List[float]],
    compound_names: Optional[Dict[str, str]] = None,
    *,
    tolerance_ppm: float = 0.04,
    minimum_confidence: float = 30.0,
) -> List[Dict]:
    """
    Match complete metabolite peak patterns, not isolated peak labels.

    Confidence combines shift coverage, ppm error, SNR, amount of evidence and
    peak ambiguity. Single-peak compounds are intentionally capped because one
    resonance alone is rarely a secure identification in a mixture.
    """
    usable = [peak for peak in peaks if not peak.get("artifact")]
    candidates: List[Dict] = []
    peak_support: Dict[float, int] = {}

    for key, expected in reference_shifts.items():
        expected = sorted(float(shift) for shift in expected if 0 <= float(shift) <= 12)
        if not expected:
            continue
        available = list(range(len(usable)))
        matches = []
        for shift in expected:
            if not available:
                break
            best = min(available, key=lambda i: abs(float(usable[i]["ppm"]) - shift))
            error = abs(float(usable[best]["ppm"]) - shift)
            if error <= tolerance_ppm:
                peak = usable[best]
                matches.append(
                    {
                        "expected_ppm": round(shift, 4),
                        "observed_ppm": peak["ppm"],
                        "error_ppm": round(error, 5),
                        "snr": peak["snr"],
                    }
                )
                available.remove(best)
                peak_support[float(peak["ppm"])] = peak_support.get(float(peak["ppm"]), 0) + 1

        if not matches:
            continue
        coverage = len(matches) / len(expected)
        mean_error = float(np.mean([match["error_ppm"] for match in matches]))
        median_snr = float(np.median([match["snr"] for match in matches]))
        shift_score = math.exp(-mean_error / max(tolerance_ppm * 0.65, 1e-9))
        snr_score = 1.0 - math.exp(-median_snr / 12.0)
        evidence_score = min(1.0, len(matches) / 2.0)
        confidence = 100.0 * (
            0.50 * coverage + 0.30 * shift_score + 0.20 * snr_score
        ) * evidence_score
        if len(expected) == 1:
            confidence = min(confidence, 68.0)

        candidates.append(
            {
                "metabolite": (compound_names or {}).get(key, key),
                "library_key": key,
                "expected_shifts": expected,
                "matched_peaks": matches,
                "matched_count": len(matches),
                "expected_count": len(expected),
                "coverage": round(coverage, 3),
                "mean_ppm_error": round(mean_error, 5),
                "median_snr": round(median_snr, 1),
                "_raw_confidence": confidence,
            }
        )

    assignments: List[Dict] = []
    for candidate in candidates:
        ambiguity = float(
            np.mean(
                [
                    peak_support.get(float(match["observed_ppm"]), 1)
                    for match in candidate["matched_peaks"]
                ]
            )
        )
        confidence = candidate.pop("_raw_confidence") / (1.0 + 0.12 * (ambiguity - 1.0))
        confidence = round(float(np.clip(confidence, 0, 100)), 1)
        if confidence < minimum_confidence:
            continue
        candidate["ambiguity"] = round(ambiguity, 2)
        candidate["confidence"] = confidence
        candidate["confidence_label"] = (
            "high" if confidence >= 75 else "medium" if confidence >= 50 else "tentative"
        )
        assignments.append(candidate)

    assignments.sort(
        key=lambda item: (
            -item["confidence"],
            -item["matched_count"],
            item["mean_ppm_error"],
        )
    )
    return assignments


def quality_control(
    y: np.ndarray,
    peaks: Sequence[Dict],
    phase: Optional[Dict[str, float]] = None,
    reference: Optional[Dict] = None,
) -> Dict:
    """Calculate compact, auditable spectrum-quality indicators."""
    y = np.asarray(y, dtype=float)
    noise = max(estimate_noise(y), 1e-12)
    # Ignore symmetric baseline noise; phase quality is reflected by signal
    # excursions beyond three noise standard deviations.
    positive = float(np.sum(np.clip(y - 3 * noise, 0, None)))
    negative = float(np.sum(np.clip(-y - 3 * noise, 0, None)))
    negative_fraction = negative / max(positive + negative, 1e-12)
    max_snr = float(np.max(y) / noise) if len(y) else 0.0
    non_artifact = sum(not peak.get("artifact") for peak in peaks)

    score = 100.0
    score -= min(35.0, negative_fraction * 140.0)
    if max_snr < 10:
        score -= 25
    elif max_snr < 20:
        score -= 10
    if non_artifact < 3:
        score -= 20
    score = round(float(np.clip(score, 0, 100)), 1)
    grade = "excellent" if score >= 90 else "good" if score >= 75 else "review" if score >= 55 else "poor"

    return {
        "score": score,
        "grade": grade,
        "noise_sigma": round(noise, 7),
        "max_snr": round(max_snr, 1),
        "negative_area_fraction": round(negative_fraction, 5),
        "peaks_total": len(peaks),
        "peaks_non_artifact": non_artifact,
        "phase": phase or {},
        "reference": reference or {"method": "acquisition metadata", "offset_ppm": 0.0},
    }


# ── Spectrum-level analysis ─────────────────────────────────────────────────
def analyze_spectrum(
    ppm: np.ndarray,
    intensity: np.ndarray,
    *,
    snr: float = 5.0,
    baseline: bool = True,
    reference_shifts: Optional[Dict[str, List[float]]] = None,
    compound_names: Optional[Dict[str, str]] = None,
    tolerance_ppm: float = 0.04,
    phase: Optional[Dict[str, float]] = None,
    processing_steps: Optional[List[str]] = None,
    assignment_backend: str = "hybrid",
    reference_info: Optional[Dict] = None,
) -> Dict:
    """Baseline-correct, normalize, peak-pick, assign and quality-score a spectrum."""
    ppm = np.asarray(ppm, dtype=float).reshape(-1)
    y = np.asarray(intensity, dtype=float).reshape(-1)
    if len(ppm) != len(y) or len(y) < 16:
        raise ValueError("Spectrum requires equal ppm/intensity arrays with at least 16 points.")
    finite = np.isfinite(ppm) & np.isfinite(y)
    ppm, y = ppm[finite], y[finite]
    order = np.argsort(ppm)[::-1]
    ppm, y = ppm[order], y[order]

    mask = (ppm >= 0.0) & (ppm <= 10.0)
    ppm, y = ppm[mask], y[mask]
    if len(y) < 16:
        raise ValueError("Spectrum contains too few points in the 0–10 ppm window.")

    steps = list(processing_steps or [])
    if baseline:
        y = baseline_correct(y)
        steps.append("asymmetric least-squares baseline")
    scale = float(np.max(np.abs(y))) or 1.0
    y = y / scale
    steps.append("0–10 ppm window + normalization")

    peaks = pick_peaks(ppm, y, snr=snr)
    pattern_assignments = assign_metabolites(
        peaks,
        reference_shifts or {},
        compound_names,
        tolerance_ppm=tolerance_ppm,
    )
    neural_assignments = []
    backend_status = nmrformer_backend.status()
    if assignment_backend in ("hybrid", "nmrformer") and backend_status["available"]:
        neural_assignments = nmrformer_backend.predict(ppm, y, peaks)
    if assignment_backend == "nmrformer" and backend_status["available"]:
        assignments = nmrformer_backend.hybridize([], neural_assignments)
        method = "nmrformer"
    elif assignment_backend == "hybrid" and backend_status["available"]:
        assignments = nmrformer_backend.hybridize(
            pattern_assignments, neural_assignments
        )
        method = "hybrid-pattern+nmrformer"
    else:
        assignments = pattern_assignments
        method = "reference-pattern-matcher"
    ssl_matches = []
    ssl_status = {"available": False, "trained": False}
    try:
        from . import self_supervised

        ssl_status = self_supervised.status()
        if ssl_status.get("trained"):
            ssl_matches = self_supervised.identify(ppm, y, top_k=5)
            similarity_by_name = {
                item["metabolite"].casefold(): item["cosine_similarity"]
                for item in ssl_matches
            }
            for assignment in assignments:
                assignment["self_supervised_similarity"] = similarity_by_name.get(
                    assignment["metabolite"].casefold()
                )
    except Exception as exc:
        ssl_status = {
            "available": False,
            "trained": False,
            "reason": str(exc),
        }
    qc = quality_control(y, peaks, phase, reference_info)
    display_ppm, display_y = _downsample_spectrum(ppm, y)

    return {
        "ppm": [round(float(value), 4) for value in display_ppm],
        "intensity": [round(float(value), 6) for value in display_y],
        "peaks": peaks[:120],
        "assignments": assignments,
        "quality_control": qc,
        "noise_sigma": qc["noise_sigma"],  # backward-compatible convenience
        "n_points": int(len(ppm)),
        "processing": steps,
        "assignment_method": method,
        "nmrformer": backend_status,
        "self_supervised": ssl_status,
        "self_supervised_matches": ssl_matches,
    }


def process_fid(
    fid: np.ndarray,
    sw_hz: float,
    sf_mhz: float,
    lb_hz: float = 0.3,
    *,
    carrier_ppm: float = 4.7,
    zero_fill_factor: int = 2,
    snr: float = 5.0,
    reference_shifts: Optional[Dict[str, List[float]]] = None,
    compound_names: Optional[Dict[str, str]] = None,
    tolerance_ppm: float = 0.04,
    assignment_backend: str = "hybrid",
    auto_reference: bool = True,
) -> Dict:
    """Full FID → processed spectrum, peaks, assignments and QC."""
    fid = np.asarray(fid, dtype=complex).squeeze()
    if fid.ndim != 1 or len(fid) < 32:
        raise ValueError("Only one-dimensional FIDs with at least 32 complex points are supported.")
    if sw_hz <= 0 or sf_mhz <= 0:
        raise ValueError("Sweep width and spectrometer frequency must be positive.")

    fid = dc_correct(fid)
    fid = apodize(fid, lb_hz, sw_hz)
    spectrum = zerofill_fft(fid, factor=zero_fill_factor)
    spectrum, phase = auto_phase(spectrum, return_phases=True)

    sw_ppm = sw_hz / sf_mhz
    ppm = np.linspace(
        carrier_ppm + sw_ppm / 2.0,
        carrier_ppm - sw_ppm / 2.0,
        len(spectrum),
        endpoint=False,
    )
    reference_info = {"method": "acquisition metadata", "offset_ppm": 0.0}
    if auto_reference:
        ppm, reference_info = reference_to_internal_standard(ppm, np.real(spectrum))
    result = analyze_spectrum(
        ppm,
        np.real(spectrum),
        snr=snr,
        reference_shifts=reference_shifts,
        compound_names=compound_names,
        tolerance_ppm=tolerance_ppm,
        phase=phase,
        assignment_backend=assignment_backend,
        reference_info=reference_info,
        processing_steps=[
            "DC correction",
            f"exponential apodization ({lb_hz:g} Hz)",
            f"power-of-two zero fill ×{zero_fill_factor}",
            "FFT",
            "ACME zero/first-order auto-phase" if NMRGLUE else "zero-order auto-phase",
            f"acquisition-aware ppm axis (carrier {carrier_ppm:.4f} ppm)",
            (
                f"internal-standard reference ({reference_info['offset_ppm']:+.5f} ppm)"
                if reference_info["method"] == "internal standard"
                else "acquisition-metadata reference"
            ),
        ],
    )
    result["acquisition_axis"] = {
        "sweep_width_hz": round(float(sw_hz), 4),
        "spectrometer_mhz": round(float(sf_mhz), 6),
        "carrier_ppm": round(float(carrier_ppm), 6),
    }
    return result


def parse_processed_spectrum(raw: bytes, filename: str = "") -> Tuple[np.ndarray, np.ndarray]:
    """Parse a two-column CSV/TSV containing ppm and intensity."""
    import pandas as pd

    separator = "\t" if filename.lower().endswith((".tsv", ".txt")) else None
    frame = pd.read_csv(io.BytesIO(raw), sep=separator, engine="python")
    normalized = {str(column).strip().lower(): column for column in frame.columns}
    ppm_column = next(
        (normalized[name] for name in ("ppm", "chemical_shift", "shift", "position") if name in normalized),
        None,
    )
    intensity_column = next(
        (normalized[name] for name in ("intensity", "amplitude", "signal", "value") if name in normalized),
        None,
    )
    if ppm_column is None or intensity_column is None:
        numeric = frame.apply(pd.to_numeric, errors="coerce")
        usable = [column for column in numeric.columns if numeric[column].notna().sum() >= 16]
        if len(usable) < 2:
            raise ValueError("Expected columns named ppm and intensity, or at least two numeric columns.")
        ppm_column, intensity_column = usable[:2]
    ppm = pd.to_numeric(frame[ppm_column], errors="coerce").to_numpy(float)
    intensity = pd.to_numeric(frame[intensity_column], errors="coerce").to_numpy(float)
    return ppm, intensity


def _downsample_spectrum(
    ppm: np.ndarray,
    y: np.ndarray,
    max_points: int = 5000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Peak-preserving downsampling for browser display."""
    if len(y) <= max_points:
        return ppm, y
    edges = np.linspace(0, len(y), max_points + 1, dtype=int)
    chosen = []
    for start, stop in zip(edges[:-1], edges[1:]):
        if stop <= start:
            continue
        chosen.append(start + int(np.argmax(np.abs(y[start:stop]))))
    indices = np.array(sorted(set(chosen)), dtype=int)
    return ppm[indices], y[indices]


# ── Bruker reader ───────────────────────────────────────────────────────────
def _validate_zip(archive: zipfile.ZipFile) -> None:
    members = archive.infolist()
    if len(members) > MAX_ZIP_MEMBERS:
        raise ValueError("Archive contains too many files.")
    total_size = 0
    for member in members:
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("Archive contains an unsafe path.")
        mode = (member.external_attr >> 16) & 0o170000
        if mode == 0o120000:
            raise ValueError("Archive symlinks are not accepted.")
        total_size += member.file_size
    if total_size > MAX_ZIP_UNCOMPRESSED_BYTES:
        raise ValueError("Archive is too large after decompression.")


def read_bruker_zip(raw: bytes) -> Tuple[np.ndarray, Dict]:
    """Safely read a zipped one-dimensional Bruker experiment via nmrglue."""
    if not NMRGLUE:
        raise RuntimeError("nmrglue not installed")
    if not raw:
        raise ValueError("Uploaded archive is empty.")

    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        _validate_zip(archive)
        with tempfile.TemporaryDirectory(prefix="ruuphenome_bruker_") as directory:
            root_dir = Path(directory)
            archive.extractall(root_dir)
            candidates = sorted(
                root_dir.rglob("fid"),
                key=lambda path: (len(path.relative_to(root_dir).parts), str(path)),
            )
            if not candidates:
                raise ValueError("No Bruker 'fid' file was found in the archive.")
            experiment_dir = candidates[0].parent
            dic, data = ng.bruker.read(str(experiment_dir))
            # Bruker direct-dimension quadrature is opposite to the synthetic
            # positive-frequency convention used by numpy's FFT.
            data = np.conj(np.asarray(data).squeeze())
            if data.ndim != 1:
                raise ValueError(f"Expected a 1D Bruker FID, received shape {data.shape}.")

            digital_filter_removed = False
            try:
                data = ng.bruker.remove_digital_filter(dic, data)
                digital_filter_removed = True
            except (KeyError, ValueError, IndexError):
                pass

            acqus = dic.get("acqus", {})
            sw_hz = float(acqus["SW_h"])
            sf_mhz = float(acqus["SFO1"])
            o1_hz = float(acqus.get("O1", 4.7 * sf_mhz))
            carrier_ppm = o1_hz / sf_mhz
            metadata = {
                "sw_hz": sw_hz,
                "sf_mhz": sf_mhz,
                "carrier_ppm": carrier_ppm,
                "size": int(len(data)),
                "nucleus": str(acqus.get("NUC1", "")).strip("<>"),
                "solvent": str(acqus.get("SOLVENT", "")).strip("<>"),
                "temperature_k": _optional_float(acqus.get("TE")),
                "pulse_program": str(acqus.get("PULPROG", "")).strip("<>"),
                "digital_filter_removed": digital_filter_removed,
                "quadrature_conjugated": True,
                "experiment_folder": experiment_dir.name,
            }
            return np.asarray(data, dtype=complex), metadata


def _optional_float(value) -> Optional[float]:
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


# ── Synthetic validation/demo ───────────────────────────────────────────────
def demo_spectrum(
    reference_shifts: Dict[str, List[float]],
    abundances: Optional[Dict[str, float]] = None,
    compound_names: Optional[Dict[str, str]] = None,
) -> Dict:
    """
    Generate a deliberately imperfect serum-like FID, then process it.

    Phase error, baseline-producing DC offset and noise are included so this
    endpoint exercises the correction pipeline instead of drawing an already
    perfect spectrum.
    """
    sf_mhz, sw_ppm = 600.0, 12.0
    sw_hz = sw_ppm * sf_mhz
    n = 16384
    t = np.arange(n, dtype=float) / sw_hz

    fid = np.zeros(n, dtype=complex)
    rng = np.random.default_rng(42)
    for key, shifts in reference_shifts.items():
        amplitude = float((abundances or {}).get(key, 1.0))
        for shift in shifts:
            frequency = (4.7 - float(shift)) * sf_mhz
            decay = np.exp(-t * rng.uniform(1.8, 3.5))
            fid += amplitude * np.exp(2j * np.pi * frequency * t) * decay

    # Realistic imperfections: receiver phase, DC offset, and complex noise.
    fid *= np.exp(1j * np.deg2rad(37.0))
    fid += (0.002 + 0.0015j) * max(np.max(np.abs(fid)), 1.0)
    noise_scale = 0.012 * max(np.max(np.abs(fid)), 1.0)
    fid += rng.normal(0, noise_scale, n) + 1j * rng.normal(0, noise_scale, n)

    result = process_fid(
        fid,
        sw_hz,
        sf_mhz,
        lb_hz=0.3,
        carrier_ppm=4.7,
        zero_fill_factor=2,
        snr=5.0,
        reference_shifts=reference_shifts,
        compound_names=compound_names,
    )
    result["source"] = "synthetic-validation-demo"
    result["synthetic_truth"] = {
        "metabolites": len([shifts for shifts in reference_shifts.values() if shifts]),
        "resonances": sum(len(shifts) for shifts in reference_shifts.values()),
        "injected_phase_degrees": 37.0,
    }
    return result
