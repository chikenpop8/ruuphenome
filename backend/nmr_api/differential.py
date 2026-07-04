"""
Differential abundance analysis for Track 2 — whole-matrix, descriptive.

Given a samples × metabolites concentration matrix and group labels, this
computes, for EVERY metabolite:

  • per-group means,
  • log2 fold-change (2-group: positive class vs reference),
  • a hypothesis test — Welch t-test + Mann-Whitney U for two groups; one-way
    ANOVA + Kruskal-Wallis for >2 groups (the non-parametric test is the
    robust default headline for non-normal NMR data),
  • Benjamini-Hochberg q-values across ALL metabolites (FDR control),
  • an effect size (Cohen's d for 2 groups, η² for >2),
  • a volcano-ready array (2-group).

This is *descriptive* differential expression, not a predictive/generalization
metric, so it is computed on the full data by design — no train/test split is
appropriate here and no accuracy is claimed, so there is no leakage concern.
It complements (does not replace) the leakage-safe biomarker_engine.discover().
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats

from .biomarker_engine import bh_qvalues


def _cohens_d(a: np.ndarray, b: np.ndarray) -> Optional[float]:
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return None
    sa2, sb2 = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((na - 1) * sa2 + (nb - 1) * sb2) / (na + nb - 2))
    if pooled == 0:
        return None
    return float((np.mean(b) - np.mean(a)) / pooled)


def _eta_squared(groups: List[np.ndarray]) -> Optional[float]:
    allv = np.concatenate(groups)
    n = len(allv)
    if n < 3:
        return None
    grand = allv.mean()
    ss_total = float(np.sum((allv - grand) ** 2))
    if ss_total <= 0:
        return None
    ss_between = float(sum(len(g) * (g.mean() - grand) ** 2 for g in groups))
    return ss_between / ss_total


def differential_analysis(
    X: np.ndarray,
    y: Sequence[int],
    feature_names: Sequence[str],
    *,
    class_names: Optional[Dict[int, str]] = None,
    alpha: float = 0.05,
) -> Dict:
    """Whole-matrix differential analysis. Returns a ranked table (by q-value),
    a volcano array (2-group), and summary counts."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    classes = sorted(np.unique(y).tolist())
    multiclass = len(classes) > 2
    cname = {int(c): (class_names or {}).get(int(c), f"class_{int(c)}") for c in classes}

    rows: List[Dict] = []
    pvals: List[float] = []
    for j, name in enumerate(feature_names):
        col = X[:, j]
        finite = np.isfinite(col)
        gvals = [col[(y == c) & finite] for c in classes]
        group_means = {cname[int(c)]: (round(float(g.mean()), 4) if g.size else None)
                       for c, g in zip(classes, gvals)}
        entry: Dict = {"metabolite": str(name), "group_means": group_means}

        if any(g.size < 2 for g in gvals):
            entry.update({"p_value": None, "q_value": None, "neg_log10_q": None,
                          "log2_fold_change": None, "effect_size": None, "test": None,
                          "direction": None})
            rows.append(entry)
            pvals.append(np.nan)
            continue

        if not multiclass:
            a, b = gvals[0], gvals[1]      # reference (class 0) vs positive (class 1)
            ma, mb = float(a.mean()), float(b.mean())
            lfc = float(np.log2(mb / ma)) if (ma > 0 and mb > 0) else None
            try:
                t_p = float(stats.ttest_ind(b, a, equal_var=False).pvalue)
            except Exception:
                t_p = np.nan
            try:
                mw_p = float(stats.mannwhitneyu(b, a, alternative="two-sided").pvalue)
            except ValueError:
                mw_p = np.nan
            headline = mw_p if np.isfinite(mw_p) else t_p       # robust default
            entry.update({
                "log2_fold_change": round(lfc, 3) if lfc is not None else None,
                "welch_t_p": round(t_p, 6) if np.isfinite(t_p) else None,
                "mannwhitney_p": round(mw_p, 6) if np.isfinite(mw_p) else None,
                "effect_size": (round(_cohens_d(a, b), 3)
                                if _cohens_d(a, b) is not None else None),
                "effect_size_kind": "cohens_d",
                "direction": ("higher_in_" + cname[classes[1]]) if mb >= ma
                             else ("higher_in_" + cname[classes[0]]),
                "test": "mann_whitney_u",
            })
        else:
            try:
                f_p = float(stats.f_oneway(*gvals).pvalue)
            except Exception:
                f_p = np.nan
            try:
                k_p = float(stats.kruskal(*gvals).pvalue)
            except ValueError:
                k_p = np.nan
            headline = k_p if np.isfinite(k_p) else f_p
            eta = _eta_squared(gvals)
            present = {c: g.mean() for c, g in zip(classes, gvals) if g.size}
            highest = max(present, key=present.get) if present else None
            entry.update({
                "anova_p": round(f_p, 6) if np.isfinite(f_p) else None,
                "kruskal_p": round(k_p, 6) if np.isfinite(k_p) else None,
                "effect_size": round(eta, 3) if eta is not None else None,
                "effect_size_kind": "eta_squared",
                "direction": ("highest_in_" + cname[int(highest)]) if highest is not None else None,
                "test": "kruskal_wallis",
            })

        entry["p_value"] = round(headline, 6) if np.isfinite(headline) else None
        rows.append(entry)
        pvals.append(headline if np.isfinite(headline) else np.nan)

    # BH q-values across all metabolites with a finite p
    pv = np.array(pvals, dtype=float)
    valid = np.isfinite(pv)
    q = np.full(len(pv), np.nan)
    if valid.any():
        q[valid] = bh_qvalues(pv[valid])
    for i, row in enumerate(rows):
        qi = q[i]
        row["q_value"] = round(float(qi), 6) if np.isfinite(qi) else None
        row["neg_log10_q"] = (round(float(-np.log10(max(qi, 1e-300))), 3)
                              if np.isfinite(qi) else None)

    ranked = sorted(rows, key=lambda r: (r["q_value"] is None, r["q_value"] if r["q_value"] is not None else 1.0))
    n_sig = int(sum(1 for r in rows if r["q_value"] is not None and r["q_value"] < alpha))

    volcano = None
    if not multiclass:
        volcano = [
            {"metabolite": r["metabolite"],
             "log2_fold_change": r["log2_fold_change"],
             "neg_log10_q": r["neg_log10_q"],
             "significant": bool(r["q_value"] is not None and r["q_value"] < alpha
                                 and r["log2_fold_change"] is not None)}
            for r in rows if r["log2_fold_change"] is not None and r["neg_log10_q"] is not None
        ]

    return {
        "task_type": "multiclass" if multiclass else "binary",
        "n_classes": len(classes),
        "class_labels": {int(c): cname[int(c)] for c in classes},
        "n_metabolites": int(X.shape[1]),
        "n_significant": n_sig,
        "alpha": alpha,
        "test": "kruskal_wallis + one-way ANOVA" if multiclass else "mann_whitney_u + welch_t",
        "fdr_method": "benjamini_hochberg",
        "table": ranked,
        "volcano": volcano,
        "notes": (
            "Descriptive differential analysis on the full cohort (not a predictive "
            "estimate). Headline p-value is the rank-based test (robust for NMR); "
            "q-values are Benjamini-Hochberg FDR across all metabolites."
        ),
    }
