"""
Biomarker discovery engine — leakage-safe, p >> n.

Built to the BDI-Hackathon Phenome rubric: with ~20,000 ppm features and only
hundreds of samples, naive feature selection on the full dataset produces
spurious, overfit panels. This engine does it correctly:

  1. variance / prevalence filter          (drop dead features)
  2. univariate screening + Benjamini-Hochberg FDR  (control 20k tests)
  3. feature selection INSIDE every CV fold (no leakage)
  4. nested / repeated stratified CV        (honest AUC + F1)
  5. Top-k biomarker stability (Jaccard)    (reliability of the panel)

It also reports the *leaky* AUC (selection on all data, then CV) so the
inflation from data leakage is visible and quantified.

Works on any samples × features matrix — the 20k-ppm raw spectra (true p>>n) or
the smaller identified-metabolite table.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import t as _tdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


# ── statistics helpers ──────────────────────────────────────────────────────
def benjamini_hochberg(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Return a boolean mask of features passing BH-FDR at level alpha."""
    n = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    crit = alpha * (np.arange(1, n + 1) / n)
    passed = ranked <= crit
    if not passed.any():
        return np.zeros(n, dtype=bool)
    cutoff = ranked[np.max(np.where(passed)[0])]
    return pvals <= cutoff


def screen_univariate(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized point-biserial correlation of every feature with the binary
    target, plus a two-sided t-test p-value. O(n·p), never builds a p×p matrix.
    """
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    denom = np.sqrt((Xc ** 2).sum(0)) * np.sqrt((yc ** 2).sum()) + 1e-12
    r = (Xc * yc[:, None]).sum(0) / denom
    n = len(y)
    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = r * np.sqrt((n - 2) / np.clip(1 - r ** 2, 1e-12, None))
    pvals = 2 * _tdist.sf(np.abs(tstat), df=n - 2)
    return np.abs(r), pvals


def variance_filter(X: np.ndarray, min_var: float = 1e-8) -> np.ndarray:
    """Boolean mask of features with non-trivial variance."""
    return X.var(axis=0) > min_var


# ── core discovery ──────────────────────────────────────────────────────────
def _select_in_fold(Xtr, ytr, k, fdr) -> np.ndarray:
    """Leakage-safe feature selection using ONLY the training fold."""
    keep = np.where(variance_filter(Xtr))[0]
    absr, pvals = screen_univariate(Xtr[:, keep], ytr)
    sig = benjamini_hochberg(pvals, fdr)
    cand = keep[sig] if sig.any() else keep[np.argsort(absr)[-k:]]
    # rank survivors by |r|, take top-k
    absr_c, _ = screen_univariate(Xtr[:, cand], ytr)
    sel = cand[np.argsort(absr_c)[-k:]]
    return sel


def _fit_eval(Xtr, ytr, Xte, sel):
    sc = StandardScaler().fit(Xtr[:, sel])
    clf = LogisticRegression(max_iter=2000)
    clf.fit(sc.transform(Xtr[:, sel]), ytr)
    return clf.predict_proba(sc.transform(Xte[:, sel]))[:, 1]


def _q2_score(y: np.ndarray, oof: np.ndarray) -> float:
    """Q² (predictive R²) from out-of-fold predictions — the metabolomics
    robustness standard. Q² = 1 − PRESS/TSS; >0 means predictive, ~1 is strong."""
    y = np.asarray(y, dtype=float)
    oof = np.asarray(oof, dtype=float)
    mask = ~np.isnan(oof)
    if mask.sum() < 2:
        return float("nan")
    yt, ot = y[mask], oof[mask]
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot <= 0:
        return float("nan")
    return float(1.0 - np.sum((yt - ot) ** 2) / ss_tot)


def _vip_scores(X, y, feature_names, n_components: int = 2) -> Dict[str, float]:
    """
    Variable Importance in Projection (VIP) from PLS-DA — the standard
    metabolomics biomarker-ranking score. VIP > 1 marks influential features.
    """
    Xs = StandardScaler().fit_transform(np.asarray(X, dtype=float))
    p = Xs.shape[1]
    ncomp = max(1, min(n_components, p, Xs.shape[0] - 1))
    try:
        pls = PLSRegression(n_components=ncomp)
        pls.fit(Xs, np.asarray(y, dtype=float))
    except Exception:
        return {}
    t = pls.x_scores_                      # (n_samples, ncomp)
    w = pls.x_weights_                     # (p, ncomp), unit-norm columns
    q = pls.y_loadings_.ravel()            # (ncomp,)
    ssy = np.sum((t ** 2), axis=0) * (q ** 2)   # explained Y SS per component
    total = float(np.sum(ssy))
    if total <= 0:
        return {}
    vip = np.sqrt(p * ((w ** 2) @ ssy) / total)
    return {feature_names[i]: round(float(vip[i]), 3) for i in range(p)}


def _cv_auc_once(X, y, k, fdr, n_splits, seed, groups) -> float:
    """One leakage-safe CV pass → ROC-AUC (used by the permutation test)."""
    skf = (StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
           if groups is not None
           else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed))
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y, groups):
        sel = _select_in_fold(X[tr], y[tr], k, fdr)
        oof[te] = _fit_eval(X[tr], y[tr], X[te], sel)
    return roc_auc_score(y, oof)


def _permutation_pvalue(X, y, k, fdr, n_splits, seed, groups, real_auc,
                        n_perm: int = 100) -> float:
    """
    Label-permutation test (best practice for metabolomics validation): shuffle
    the outcome n_perm times, recompute the leakage-safe CV AUC, and report
    p = (1 + #{perm AUC ≥ real}) / (n_perm + 1).
    """
    rng = np.random.default_rng(seed)
    count = 0
    done = 0
    for i in range(n_perm):
        yp = rng.permutation(y)
        if len(np.unique(yp)) < 2:
            continue
        try:
            if _cv_auc_once(X, yp, k, fdr, n_splits, seed + i + 1, groups) >= real_auc:
                count += 1
            done += 1
        except Exception:
            continue
    return (1 + count) / (done + 1)


def discover(
    X: np.ndarray,
    y: np.ndarray,
    k: int = 20,
    fdr: float = 0.05,
    n_splits: int = 5,
    repeats: int = 2,
    seed: int = 0,
    feature_names: Optional[Sequence[str]] = None,
    groups: Optional[Sequence] = None,
    permutations: int = 100,
    compute_vip: bool = True,
) -> Dict:
    """
    Honest p>>n biomarker discovery. Returns AUC, F1, Q², permutation p-value,
    Top-k stability + VIP-ranked stable panel, and the leaky AUC for comparison.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups_array = np.asarray(groups) if groups is not None else None
    n_splits = min(n_splits, np.bincount(y).min())  # guard tiny classes
    if groups_array is not None:
        per_class_groups = [
            len(np.unique(groups_array[y == klass])) for klass in np.unique(y)
        ]
        n_splits = min(n_splits, min(per_class_groups), len(np.unique(groups_array)))

    fold_sets: List[set] = []
    rep_aucs, rep_f1s, rep_q2s = [], [], []

    for rep in range(repeats):
        skf = (
            StratifiedGroupKFold(
                n_splits=n_splits, shuffle=True, random_state=seed + rep
            )
            if groups_array is not None
            else StratifiedKFold(
                n_splits=n_splits, shuffle=True, random_state=seed + rep
            )
        )
        oof = np.full(len(y), np.nan)
        for tr, te in skf.split(X, y, groups_array):
            sel = _select_in_fold(X[tr], y[tr], k, fdr)
            fold_sets.append(set(sel.tolist()))
            oof[te] = _fit_eval(X[tr], y[tr], X[te], sel)
        rep_aucs.append(roc_auc_score(y, oof))
        rep_f1s.append(f1_score(y, (oof >= 0.5).astype(int)))
        rep_q2s.append(_q2_score(y, oof))

    honest_auc = float(np.mean(rep_aucs))
    honest_f1 = float(np.mean(rep_f1s))
    honest_q2 = float(np.nanmean(rep_q2s))

    # Top-k stability — mean pairwise Jaccard of the per-fold selections
    def jac(a, b):
        return len(a & b) / max(1, len(a | b))
    pairs = [(i, j) for i in range(len(fold_sets)) for j in range(i + 1, len(fold_sets))]
    stability = float(np.mean([jac(fold_sets[i], fold_sets[j]) for i, j in pairs])) if pairs else 0.0

    # stable panel: features chosen in >= half of all folds
    cnt = Counter()
    for s in fold_sets:
        cnt.update(s)
    half = len(fold_sets) / 2
    stable_idx = sorted([f for f, c in cnt.items() if c >= half], key=lambda f: -cnt[f])

    # leaky AUC (select on ALL data first → optimistic) for comparison
    leaky = leaky_auc(X, y, k, n_splits, seed, groups_array)

    def name(i):
        return feature_names[i] if feature_names is not None else f"feature_{i}"

    stable_names = [name(i) for i in stable_idx]

    # VIP (PLS-DA) ranking of the stable panel — standard biomarker importance
    vip_panel = {}
    if compute_vip and stable_idx:
        names_all = [name(i) for i in range(X.shape[1])]
        all_vip = _vip_scores(X, y, names_all)
        vip_panel = {nm: all_vip.get(nm) for nm in stable_names if nm in all_vip}

    # Permutation test p-value on the honest AUC (label-shuffle null)
    perm_p = None
    if permutations and permutations > 0:
        perm_p = _permutation_pvalue(
            X, y, k, fdr, n_splits, seed, groups_array, honest_auc, n_perm=permutations)

    return {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "honest_roc_auc": round(honest_auc, 4),
        "honest_f1": round(honest_f1, 4),
        "honest_q2": round(honest_q2, 4),
        "permutation_p_value": round(perm_p, 4) if perm_p is not None else None,
        "n_permutations": int(permutations) if perm_p is not None else 0,
        "leaky_roc_auc": round(leaky, 4),
        "leakage_inflation": round(leaky - honest_auc, 4),
        "topk_stability_jaccard": round(stability, 4),
        "stable_panel": stable_names,
        "stable_panel_counts": {name(i): cnt[i] for i in stable_idx},
        "vip_scores": vip_panel,
        "validation": (
            "patient-grouped repeated stratified CV"
            if groups_array is not None
            else "repeated stratified CV"
        ),
        "validation_notes": (
            "Q² = predictive R² from out-of-fold predictions; "
            "permutation p-value from label-shuffle null; "
            "VIP > 1 marks PLS-DA-influential metabolites."
        ),
        "params": {"k": k, "fdr": fdr, "n_splits": n_splits, "repeats": repeats},
    }


def leaky_auc(X, y, k, n_splits, seed, groups=None) -> float:
    """AUC when features are chosen on the FULL data before CV (data leakage)."""
    absr, _ = screen_univariate(X, y)            # uses all samples → leakage
    sel = np.argsort(absr)[-k:]
    skf = (
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if groups is not None
        else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    oof = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y, groups):
        oof[te] = _fit_eval(X[tr], y[tr], X[te], sel)
    return float(roc_auc_score(y, oof))


# ── p>>n simulation (the brief's design) ────────────────────────────────────
def simulate(n: int = 200, p: int = 20000, n_true: int = 8,
             effect: float = 1.2, seed: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Random NMR-like matrix with `n_true` real biomarkers embedded among p noise
    features. Returns (X, y, true_indices).
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    X = rng.normal(0, 1, size=(n, p)).astype(np.float32)
    true_idx = rng.choice(p, size=n_true, replace=False)
    for j in true_idx:
        X[:, j] += effect * y
    return X, y, true_idx


def benchmark(n_datasets: int = 100, n: int = 200, p: int = 20000,
              n_true: int = 8, effect: float = 1.2, k: int = 20) -> Dict:
    """
    Run the engine across `n_datasets` independent p>>n cohorts and summarize
    honest AUC, leaky AUC, stability, and true-biomarker recovery.
    """
    honest, leaky, stab, recov = [], [], [], []
    for s in range(n_datasets):
        X, y, true_idx = simulate(n=n, p=p, n_true=n_true, effect=effect, seed=s)
        res = discover(X, y, k=k, repeats=1, seed=s)
        honest.append(res["honest_roc_auc"])
        leaky.append(res["leaky_roc_auc"])
        stab.append(res["topk_stability_jaccard"])
        # recovery: fraction of true biomarkers in the stable panel
        stable_ids = {int(nm.split("_")[1]) for nm in res["stable_panel"] if nm.startswith("feature_")}
        recov.append(len(set(true_idx.tolist()) & stable_ids) / n_true)
    return {
        "n_datasets": n_datasets,
        "config": {"n": n, "p": p, "n_true_biomarkers": n_true, "effect": effect, "k": k},
        "honest_auc_mean": round(float(np.mean(honest)), 4),
        "honest_auc_std": round(float(np.std(honest)), 4),
        "leaky_auc_mean": round(float(np.mean(leaky)), 4),
        "mean_leakage_inflation": round(float(np.mean(np.array(leaky) - np.array(honest))), 4),
        "topk_stability_mean": round(float(np.mean(stab)), 4),
        "true_biomarker_recovery_mean": round(float(np.mean(recov)), 4),
    }
