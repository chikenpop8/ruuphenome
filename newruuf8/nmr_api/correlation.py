"""
Correlation & network analysis for Track 2.

Two products on the samples × metabolites matrix:

1. **Pairwise correlation** (Pearson or Spearman) with per-pair p-values and
   Benjamini-Hochberg FDR — a descriptive heatmap of co-variation.

2. **Partial-correlation network (Gaussian Graphical Model)** via Ledoit-Wolf
   shrinkage of the covariance. Following Krumsiek et al. 2011 (BMC Syst Biol
   5:21), edges are DIRECT associations — each pair conditioned on all other
   metabolites — which removes the indirect, system-wide correlations that make
   raw-correlation networks dense and hard to interpret. Ledoit-Wolf shrinkage
   keeps the covariance invertible even when features approach/exceed samples
   (the p≫n regime), so the precision matrix is always defined.

Also supports **metabolite-vs-numeric-covariate** correlation against any
numeric metadata column (age, BMI, a lab value, …).

All descriptive — no train/test split applies; p-values are approximate under
shrinkage (a ranking device, not exact inference) and are FDR-controlled.
Depends only on numpy / scipy / scikit-learn (already required).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats
from sklearn.covariance import LedoitWolf

from .biomarker_engine import bh_qvalues

# Cap the number of metabolites entering a correlation/network product. For the
# identified-metabolite table (tens of metabolites) this never binds; for raw
# ~20k-bin spectra we restrict to the most variable features and warn, because a
# 20k×20k matrix is neither returnable nor a well-posed precision estimate.
MAX_FEATURES = 80


def _prepare(X: np.ndarray, feature_names: Sequence[str], max_features: int):
    """Drop near-empty metabolites, median-impute, and (if too wide) keep the
    most variable `max_features`. Returns (Xc, names, warnings)."""
    X = np.asarray(X, dtype=float)
    names = list(feature_names)
    warnings: List[str] = []

    finite = np.isfinite(X)
    keep = [j for j in range(X.shape[1]) if finite[:, j].sum() >= 3
            and np.nanstd(np.where(finite[:, j], X[:, j], np.nan)) > 1e-9]
    if len(keep) < X.shape[1]:
        warnings.append(f"Dropped {X.shape[1] - len(keep)} empty/constant metabolite(s).")
    X = X[:, keep]
    names = [names[j] for j in keep]

    if X.shape[1] > max_features:
        variances = np.nanvar(X, axis=0)
        top = np.argsort(variances)[-max_features:]
        top = sorted(top.tolist())
        warnings.append(
            f"{X.shape[1]} metabolites exceeds the {max_features}-feature cap for "
            f"correlation/network analysis; restricted to the {max_features} most "
            f"variable. (Raw p≫n spectra should be reduced to identified metabolites first.)")
        X = X[:, top]
        names = [names[j] for j in top]

    # median-impute remaining missing cells (column medians)
    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    bad = ~np.isfinite(X)
    if bad.any():
        X = X.copy()
        X[bad] = np.take(med, np.where(bad)[1])
    return X, names, warnings


def _pairwise(Xc: np.ndarray, method: str) -> np.ndarray:
    """Correlation matrix; Spearman = Pearson on column ranks."""
    if method == "spearman":
        Xc = np.apply_along_axis(stats.rankdata, 0, Xc)
    # guard constant columns
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(Xc, rowvar=False)
    C = np.nan_to_num(C, nan=0.0)
    np.fill_diagonal(C, 1.0)
    return C


def _edges_from_matrix(M: np.ndarray, names: Sequence[str], n: int, dof: int,
                       alpha: float, r_threshold: float) -> Dict:
    """Turn a correlation/partial-correlation matrix into FDR-filtered edges.

    p-values via Fisher's z-transform with the given degrees of freedom `dof`
    (n-2 for pairwise; n-k-2 for partial correlation, k = conditioned vars)."""
    p = M.shape[0]
    iu = np.triu_indices(p, k=1)
    r = M[iu]
    r_clip = np.clip(r, -0.999999, 0.999999)
    scale = max(dof - 1, 1)
    z = np.arctanh(r_clip) * np.sqrt(scale)
    pv = 2 * stats.norm.sf(np.abs(z))
    q = bh_qvalues(pv) if len(pv) else pv

    edges = []
    for idx in range(len(r)):
        i, j = int(iu[0][idx]), int(iu[1][idx])
        if q[idx] < alpha and abs(r[idx]) >= r_threshold:
            edges.append({
                "source": str(names[i]),
                "target": str(names[j]),
                "r": round(float(r[idx]), 3),
                "p_value": round(float(pv[idx]), 6),
                "q_value": round(float(q[idx]), 6),
            })
    edges.sort(key=lambda e: abs(e["r"]), reverse=True)
    return {"edges": edges, "n_edges": len(edges),
            "nodes": [{"id": str(nm)} for nm in names]}


def partial_correlation(Xc: np.ndarray) -> np.ndarray:
    """Partial-correlation matrix from the Ledoit-Wolf precision (inverse
    covariance). ζ_ij = −ω_ij / √(ω_ii·ω_jj); diagonal set to 1."""
    lw = LedoitWolf().fit(Xc)
    P = lw.precision_
    d = np.sqrt(np.clip(np.diag(P), 1e-12, None))
    pcorr = -P / np.outer(d, d)
    np.fill_diagonal(pcorr, 1.0)
    return np.clip(pcorr, -1.0, 1.0)


def analyze(
    X: np.ndarray,
    feature_names: Sequence[str],
    *,
    method: str = "spearman",
    alpha: float = 0.05,
    r_threshold: float = 0.3,
    max_features: int = MAX_FEATURES,
) -> Dict:
    """Full correlation product: pairwise heatmap + FDR edges + partial-correlation
    (GGM) network. `method` is 'spearman' (default, robust) or 'pearson'."""
    method = "pearson" if method == "pearson" else "spearman"
    Xc, names, warns = _prepare(X, feature_names, max_features)
    n, p = Xc.shape
    out: Dict = {
        "method": method,
        "n_samples": int(n),
        "n_metabolites": int(p),
        "features": [str(nm) for nm in names],
        "warnings": warns,
        "alpha": alpha,
        "r_threshold": r_threshold,
    }
    if p < 2 or n < 4:
        out["error"] = "Need ≥2 metabolites and ≥4 samples for correlation analysis."
        return out

    # 1) pairwise correlation heatmap + significant edges
    C = _pairwise(Xc, method)
    out["correlation_matrix"] = np.round(C, 3).tolist()
    out["pairwise"] = _edges_from_matrix(C, names, n, dof=n - 2,
                                         alpha=alpha, r_threshold=r_threshold)

    # 2) partial-correlation (GGM) network — direct associations only
    try:
        P = partial_correlation(Xc)
        out["partial_correlation_matrix"] = np.round(P, 3).tolist()
        out["network"] = _edges_from_matrix(P, names, n, dof=n - p,
                                            alpha=alpha, r_threshold=r_threshold)
        out["network"]["kind"] = "gaussian_graphical_model (Ledoit-Wolf shrinkage)"
        out["network_notes"] = (
            "Edges are partial correlations (each pair conditioned on all other "
            "metabolites) — direct associations, more biologically interpretable "
            "than raw correlation (Krumsiek et al. 2011). p-values are approximate "
            "under shrinkage and FDR-controlled; treat as a ranking of direct links.")
    except Exception as exc:
        out["network_error"] = f"Partial-correlation network unavailable: {exc}"

    return out


def covariate_correlation(
    X: np.ndarray,
    feature_names: Sequence[str],
    covariate: Sequence[float],
    covariate_name: str = "covariate",
    *,
    method: str = "spearman",
    alpha: float = 0.05,
) -> Dict:
    """Correlate every metabolite with one numeric covariate (age/BMI/lab value)."""
    X = np.asarray(X, dtype=float)
    cov = np.asarray(covariate, dtype=float)
    corr_fn = stats.spearmanr if method != "pearson" else stats.pearsonr
    rows: List[Dict] = []
    pvals: List[float] = []
    for j, name in enumerate(feature_names):
        col = X[:, j]
        m = np.isfinite(col) & np.isfinite(cov)
        if m.sum() < 4 or np.nanstd(col[m]) < 1e-12 or np.nanstd(cov[m]) < 1e-12:
            rows.append({"metabolite": str(name), "r": None, "p_value": None})
            pvals.append(np.nan)
            continue
        r, p = corr_fn(col[m], cov[m])
        rows.append({"metabolite": str(name), "r": round(float(r), 3),
                     "p_value": round(float(p), 6), "n": int(m.sum())})
        pvals.append(float(p))
    pv = np.array(pvals, dtype=float)
    valid = np.isfinite(pv)
    q = np.full(len(pv), np.nan)
    if valid.any():
        q[valid] = bh_qvalues(pv[valid])
    for i, row in enumerate(rows):
        row["q_value"] = round(float(q[i]), 6) if np.isfinite(q[i]) else None
    ranked = sorted(rows, key=lambda r: (r["q_value"] is None, r["q_value"] if r["q_value"] is not None else 1.0))
    return {
        "covariate": covariate_name,
        "method": "pearson" if method == "pearson" else "spearman",
        "n_metabolites": int(X.shape[1]),
        "n_significant": int(sum(1 for r in rows if r["q_value"] is not None and r["q_value"] < alpha)),
        "fdr_method": "benjamini_hochberg",
        "table": ranked,
    }
