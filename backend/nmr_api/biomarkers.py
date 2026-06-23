"""
Domain 2 — Biomarker discovery (scikit-learn).

Implements the document's second stage: "merge the metabolite table with patient
metadata, then find the *minimal biomarker set* that separates the risk groups
(diabetes / hypertension) from controls."

Given the sample × metabolite abundance matrix and a grouping label, this:
  1. builds the feature matrix (samples × metabolites)
  2. imputes + standardizes
  3. fits an L1-regularized logistic regression (sparse → minimal markers)
  4. ranks metabolites by |coefficient| and permutation importance
  5. reports cross-validated ROC-AUC / accuracy
  6. returns the smallest marker panel that retains most of the signal

The MTBLS242 sample columns are prefixed by collection time point (0..4). With
no disease labels in the public file, the default task contrasts the earliest
vs latest time point — a real metabolomic question — and the same engine accepts
external diabetes/hypertension labels when available.
"""

from __future__ import annotations

import io
import re
import warnings
from typing import Dict, List, Optional, Tuple

# scikit-learn 1.8+ deprecated the `penalty=` arg; the liblinear L1 path still
# works correctly, so silence the transitional FutureWarning to keep API
# responses and logs clean.
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance

from .pipeline import NON_SAMPLE_COLS, META_COLS, read_results_table


_GROUP_RE = re.compile(r"^(\d+)[_-]")
_PATIENT_RE = re.compile(r"^[0-4][_-](\d+)")


def _sample_groups(sample_cols: List[str]) -> Dict[str, int]:
    """Map each sample column to its leading time-point group (0..4)."""
    groups = {}
    for c in sample_cols:
        m = _GROUP_RE.match(c)
        if m:
            groups[c] = int(m.group(1))
    return groups


def sample_patient_groups(sample_cols: List[str]) -> Dict[str, str]:
    """
    Map longitudinal sample names to patient IDs.

    Unknown naming schemes receive a unique ID, which is conservative: they
    can never leak across cross-validation folds as if they were one patient.
    """
    groups = {}
    for sample in sample_cols:
        match = _PATIENT_RE.match(sample)
        groups[sample] = match.group(1) if match else f"sample:{sample}"
    return groups


def build_matrix(tsv_bytes: bytes) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """Return (abundance matrix [samples × metabolites], metabolite names, groups)."""
    df = read_results_table(tsv_bytes)   # CSV/TSV + header-variant tolerant
    sample_cols = [c for c in df.columns if c not in NON_SAMPLE_COLS]
    names = df["metabolite_identification"].fillna("unknown").tolist()

    X = df[sample_cols].apply(pd.to_numeric, errors="coerce").T  # samples × metabolites
    X.columns = names
    groups = _sample_groups(sample_cols)
    return X, names, groups


def discover(
    tsv_bytes: bytes,
    group_a: Optional[int] = None,
    group_b: Optional[int] = None,
    max_panel: int = 6,
) -> Dict:
    """
    Run biomarker discovery contrasting two time-point groups.
    Returns ranked markers, model performance, and a minimal panel.
    """
    X, names, groups = build_matrix(tsv_bytes)

    available = sorted(set(groups.values()))
    if group_a is None or group_b is None:
        if len(available) < 2:
            raise ValueError("Need at least two sample groups for discovery.")
        group_a, group_b = available[0], available[-1]

    # rows belonging to the two groups
    rows_a = [s for s, g in groups.items() if g == group_a]
    rows_b = [s for s, g in groups.items() if g == group_b]
    Xa, Xb = X.loc[rows_a], X.loc[rows_b]
    Xsub = pd.concat([Xa, Xb])
    y = np.array([0] * len(Xa) + [1] * len(Xb))

    # drop metabolites that are entirely missing
    Xsub = Xsub.dropna(axis=1, how="all")
    feat_names = list(Xsub.columns)

    # full model: L1 logistic regression → sparse coefficients
    full = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(penalty="l1", solver="liblinear",
                                   C=0.5, max_iter=2000)),
    ])
    full.fit(Xsub.values, y)
    coefs = full.named_steps["clf"].coef_[0]

    # cross-validated performance of the full model
    cv = min(5, len(Xa), len(Xb))
    auc = _safe_cv(full, Xsub.values, y, cv, "roc_auc")
    acc = _safe_cv(full, Xsub.values, y, cv, "accuracy")

    # permutation importance (model-agnostic ranking)
    full.fit(Xsub.values, y)
    perm = permutation_importance(full, Xsub.values, y, n_repeats=10,
                                  random_state=0, scoring="roc_auc")

    ranked = sorted(
        [{
            "metabolite": feat_names[i],
            "coefficient": round(float(coefs[i]), 4),
            "abs_coefficient": round(float(abs(coefs[i])), 4),
            "permutation_importance": round(float(perm.importances_mean[i]), 4),
            "direction": "↑ in group B" if coefs[i] > 0 else ("↓ in group B" if coefs[i] < 0 else "—"),
        } for i in range(len(feat_names))],
        key=lambda r: (r["abs_coefficient"], r["permutation_importance"]),
        reverse=True,
    )

    # minimal panel: smallest set of non-zero markers, retrain, report AUC
    nonzero = [r for r in ranked if r["abs_coefficient"] > 0][:max_panel]
    panel_names = [r["metabolite"] for r in nonzero] or [ranked[0]["metabolite"]]
    panel_idx = [feat_names.index(n) for n in panel_names]
    panel_model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000)),
    ])
    Xpanel = Xsub.values[:, panel_idx]
    panel_auc = _safe_cv(panel_model, Xpanel, y, cv, "roc_auc")

    return {
        "task": f"time-point {group_a} vs {group_b}",
        "groups_available": available,
        "n_samples": {"group_a": len(Xa), "group_b": len(Xb)},
        "n_features": len(feat_names),
        "full_model": {
            "roc_auc": auc,
            "accuracy": acc,
            "n_nonzero_markers": int(np.sum(np.abs(coefs) > 1e-6)),
        },
        "minimal_panel": {
            "markers": panel_names,
            "size": len(panel_names),
            "roc_auc": panel_auc,
        },
        "ranked_markers": ranked,
    }


def _safe_cv(model, X, y, cv, scoring) -> Optional[float]:
    """Cross-val score with guards for tiny / single-class folds."""
    try:
        if cv < 2 or len(set(y)) < 2:
            return None
        scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
        return round(float(np.mean(scores)), 3)
    except Exception:
        return None
