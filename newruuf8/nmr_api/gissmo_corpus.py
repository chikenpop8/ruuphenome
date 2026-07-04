"""
GISSMO corpus + mixture simulator for the F8 quantifier (open data, RUO).

Builds a labeled training set for the supervised quantifier from **GISSMO** spin
systems (physically-exact ¹H shifts fit to experimental BMRB spectra). Each
training example is a synthetic ¹H mixture (superposed compound fingerprints at
random concentrations, with ppm-drift + noise augmentation) paired with the
per-compound concentration vector — the ground truth a quantifier regresses to.

**Governance:** the corpus is FETCHED off-VM (`build_corpus`, network) and BUNDLED
to `open_data/gissmo_corpus.json`; on the H100/LiCO node it is only LOADED
(`load_corpus`, no download). Open data only; the closed dataset is never used.

Real peak positions/density come from GISSMO — the whole point of F8 is to train on
the real patterns (not the simplified reference library), to close the sim-to-real
gap the held-out validation exposed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from . import external_reference as ext
    from . import pscnn
except ImportError:  # pragma: no cover - direct execution
    import external_reference as ext  # type: ignore
    import pscnn  # type: ignore

CORPUS_PATH = Path(__file__).resolve().parent / "open_data" / "gissmo_corpus.json"


def build_corpus(*, id_range: Tuple[int, int] = (1, 120), extra_ids: Optional[Sequence[str]] = None,
                 out_path: Path = CORPUS_PATH) -> Dict:
    """OFF-VM: fetch GISSMO ¹H shifts for a range of bmse IDs (+ the verified panel)
    and bundle them. Run this on a networked machine, commit the JSON, then load it
    on the (offline) training node."""
    ids = [f"bmse{i:06d}" for i in range(id_range[0], id_range[1])]
    ids += list(ext.GISSMO_IDS.values())
    ids += list(extra_ids or [])
    corpus: Dict[str, Dict] = {}
    for eid in dict.fromkeys(ids):                      # de-dup, keep order
        try:
            d = ext.fetch_shifts(eid)
        except Exception:
            continue
        shifts = [s for s in d.get("shifts", []) if 0.3 <= s <= 10.0]
        if len(shifts) >= 2 and d.get("name"):
            key = d["name"].strip()
            corpus.setdefault(key, {"id": eid, "shifts": shifts})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"n_compounds": len(corpus), "compounds": corpus}, indent=0))
    return {"n_compounds": len(corpus), "path": str(out_path)}


def load_corpus(path: Path = CORPUS_PATH) -> Dict[str, List[float]]:
    """Load {compound: shifts}. Falls back to fetching the verified panel if the
    bundled corpus is absent (so it works before `build_corpus` has been run)."""
    if Path(path).exists():
        d = json.loads(Path(path).read_text())
        return {name: rec["shifts"] for name, rec in d["compounds"].items()}
    # fallback: the verified GISSMO panel (fetched + cached)
    real = ext.real_shifts()
    return {name: shifts for name, shifts in real.items() if len(shifts) >= 2}


def _translate(fp: np.ndarray, grid: np.ndarray, d: float) -> np.ndarray:
    """Shift a fingerprint by `d` ppm on a UNIFORM grid. A ppm drift translates every
    peak equally, so f_drifted(x) = f(x - d); on a uniform grid that is one linear-
    interpolation, ~30x cheaper than recomputing the Gaussian sum per peak (the old
    drift path — a Python loop of np.exp over the whole grid per shift — dominated
    wall-clock at ~640 ms/batch; this is ~20 ms/batch, GPU-idle time eliminated)."""
    return np.interp(grid - d, grid, fp, left=0.0, right=0.0).astype(np.float32)


def build_drift_bank(corpus_fps: Dict[str, np.ndarray], names: Sequence[str], grid: np.ndarray,
                     drift: float, *, n_offsets: int = 24,
                     rng: Optional[np.random.Generator] = None) -> Dict[str, np.ndarray]:
    """Pre-translate each fingerprint to `n_offsets` random ppm drifts, so sampling a
    drifted fingerprint at train time is a free array index instead of an interp.
    Rebuild it per epoch for fresh offsets → cheap (≈n_compounds×n_offsets interps)
    yet high diversity, and it keeps the GPU fed instead of waiting on CPU data-gen."""
    rng = rng or np.random.default_rng(0)
    offs = rng.normal(0.0, drift, size=n_offsets).astype(np.float32)
    return {n: np.stack([_translate(corpus_fps[n], grid, float(d)) for d in offs]) for n in names}


def simulate_batch(corpus_fps: Dict[str, np.ndarray], names: Sequence[str], grid: np.ndarray,
                   batch_size: int, rng: np.random.Generator, *,
                   corpus_shifts: Optional[Dict[str, List[float]]] = None,
                   drift_bank: Optional[Dict[str, np.ndarray]] = None,
                   drift: float = 0.01, noise: float = 0.02) -> Tuple[np.ndarray, np.ndarray]:
    """One batch of (spectrum, concentration-vector) pairs. Spectra and targets are
    scaled by the same factor (unit-max spectrum) → a scale-invariant RELATIVE
    quantifier. ppm-drift augmentation teaches robustness to real shift variation:
    sampled from `drift_bank` (fast, precomputed) if given, else applied as a live
    fingerprint translation."""
    K = len(names)
    n_bins = len(grid)
    X = np.zeros((batch_size, n_bins), dtype=np.float32)
    C = np.zeros((batch_size, K), dtype=np.float32)
    for b in range(batch_size):
        k = int(rng.integers(2, max(3, K // 2 + 1)))
        present = rng.choice(K, size=min(k, K), replace=False)
        for i in present:
            conc = float(rng.lognormal(0.0, 0.5))
            if drift_bank is not None:
                variants = drift_bank[names[i]]
                fp = variants[int(rng.integers(len(variants)))]            # free index
            elif drift:
                fp = _translate(corpus_fps[names[i]], grid, float(rng.normal(0.0, drift)))
            else:
                fp = corpus_fps[names[i]]
            X[b] += conc * fp
            C[b, i] = conc
        X[b] += (noise * np.abs(rng.normal(size=n_bins))).astype(np.float32)
        m = float(X[b].max())
        if m > 0:
            X[b] /= m
            C[b] /= m                                   # keep S = Σ cᵢ·fpᵢ after scaling
    return X, C
