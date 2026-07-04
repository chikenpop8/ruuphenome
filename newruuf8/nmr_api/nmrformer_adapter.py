"""
NMRformer adapter — wires the real zza1211/NMRformer transformer model into
the nmrformer_backend.py contract.

Activated automatically when NMRFORMER_ADAPTER_MODULE=nmr_api.nmrformer_adapter
is set (run.sh does this).

How it works:
  1. Resamples the input spectrum to NMRformer's required density (5000 pts/ppm).
  2. Extracts peak ppm positions from the peaks list.
  3. Calls NMRformer's dataGene.test_m() — a per-peak metabolite classifier.
  4. Aggregates peak-level predictions into a ranked metabolite list.

Model: NMRformer/onedTrans_0.9782  (19 MB, 72-class transformer)
Reference: https://github.com/zza1211/NMRformer
"""

from __future__ import annotations

import sys
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

_MODULE_DIR = Path(__file__).resolve().parent
_NMRFORMER_DIR = _MODULE_DIR / "NMRformer"
_MODEL_PATH = _NMRFORMER_DIR / "onedTrans_0.9782"
_META_PATH = _NMRFORMER_DIR / "all_meta.txt"

# NMRformer spectrum resolution: 5000 samples per ppm, 0-12 ppm = 60000 points
_PTS_PER_PPM = 5000
_PPM_MAX = 12
_SPECTRUM_LEN = _PTS_PER_PPM * _PPM_MAX  # 60000


def _available() -> bool:
    return _MODEL_PATH.exists() and _META_PATH.exists() and _NMRFORMER_DIR.exists()


@lru_cache(maxsize=1)
def _load_meta() -> List[str]:
    return [line.strip() for line in _META_PATH.read_text().splitlines() if line.strip()]


def _resample_spectrum(ppm: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """Interpolate spectrum onto NMRformer's 60000-point 0-12 ppm grid."""
    grid = np.linspace(0, _PPM_MAX, _SPECTRUM_LEN)
    order = np.argsort(ppm)
    x = np.asarray(ppm, dtype=float)[order]
    y = np.asarray(intensity, dtype=float)[order]
    resampled = np.interp(grid, x, y, left=0.0, right=0.0)
    peak_val = np.percentile(np.abs(resampled), 99.5) or 1.0
    return np.clip(resampled / peak_val, -1.5, 1.5).astype(np.float32)


def _run_nmrformer(spectrum60k: np.ndarray, peak_ppms: List[float]):
    """Call NMRformer's dataGene classifier. Returns a pandas DataFrame."""
    nmrformer_str = str(_NMRFORMER_DIR)
    if nmrformer_str not in sys.path:
        sys.path.insert(0, nmrformer_str)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        from dataGene import dataGene as DataGene  # noqa: PLC0415

    meta = _load_meta()
    dg = DataGene(allMeta=meta)

    margin = 64 / _PTS_PER_PPM  # 0.0128 ppm — minimum distance from edge
    valid_peaks = [p for p in peak_ppms if margin < p < (_PPM_MAX - margin)]
    if not valid_peaks:
        return None

    return dg.test_m(
        model_pth=str(_MODEL_PATH),
        spectra=spectrum60k,
        peaks=valid_peaks,
    )


def predict_assignments(
    ppm: np.ndarray,
    intensity: np.ndarray,
    peaks: Sequence[Dict],
) -> List[Dict]:
    """
    Map a spectrum to metabolite assignment candidates using NMRformer.

    Returns items conforming to the nmrformer_backend contract:
      {"metabolite": str, "confidence": float 0-100, ...}
    """
    if not _available():
        return []

    try:
        ppm_arr = np.asarray(ppm, dtype=float)
        int_arr = np.asarray(intensity, dtype=float)

        peak_ppms: List[float] = []
        for pk in peaks:
            pos = pk.get("ppm") or pk.get("chemical_shift") or pk.get("position")
            if pos is not None:
                peak_ppms.append(float(pos))

        if not peak_ppms:
            return []

        spectrum60k = _resample_spectrum(ppm_arr, int_arr)
        df = _run_nmrformer(spectrum60k, peak_ppms)
        if df is None or df.empty:
            return []

        # Aggregate per-peak predictions into metabolite-level confidence scores.
        scores: Dict[str, float] = {}
        counts: Dict[str, int] = {}
        for _, row in df.iterrows():
            for rank in range(3):
                name = row.get(f"pred_{rank}", [None])
                prob = row.get(f"prob_{rank}", [0.0])
                if isinstance(name, list):
                    for n, p in zip(name, prob):
                        if n:
                            weight = float(p) * (0.7 ** rank)
                            scores[n] = scores.get(n, 0.0) + weight
                            counts[n] = counts.get(n, 0) + 1
                elif name:
                    weight = float(prob) * (0.7 ** rank)
                    scores[str(name)] = scores.get(str(name), 0.0) + weight
                    counts[str(name)] = counts.get(str(name), 0) + 1

        if not scores:
            return []

        max_score = max(scores.values()) or 1.0
        results = []
        for metabolite, score in sorted(scores.items(), key=lambda x: -x[1]):
            confidence = min(100.0, (score / max_score) * 100.0)
            results.append({
                "metabolite": metabolite,
                "confidence": round(confidence, 1),
                "peak_indices": list(range(counts[metabolite])),
                "source": "nmrformer",
                "peak_votes": counts[metabolite],
            })
        return results

    except Exception:
        return []
