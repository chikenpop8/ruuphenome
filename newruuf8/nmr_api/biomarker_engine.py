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

Supports BINARY and MULTI-CLASS (≥3 group) tasks. The binary path is unchanged;
multi-class uses ANOVA-F screening, multinomial logistic regression, and
macro-averaged one-vs-rest ROC-AUC + macro-F1 + per-class recall — the
standard honest metrics for small-n multi-group metabolomics classification.
"""

from __future__ import annotations

import warnings
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import t as _tdist
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
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


def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values) for a vector of p-values.

    Returns one q per input p (same order), monotone-corrected — used by the
    differential-analysis table to report FDR per metabolite across the whole
    feature set."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity from the largest rank down
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.clip(q, 0.0, 1.0)
    out = np.empty(n, dtype=float)
    out[order] = q
    return out


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


def screen_features(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Univariate feature screen returning (score, pvals).

    Dispatches on the number of classes so the binary path is byte-for-byte the
    proven point-biserial screen, while ≥3-group tasks use the ANOVA F-test
    (f_classif) — the standard univariate filter for multi-class feature
    selection. Higher score = more group-separating in both cases."""
    y = np.asarray(y)
    X = np.asarray(X, dtype=float)
    if not np.isfinite(X).all():        # f_classif / correlation need finite input
        (X,) = _impute_median(X)
    if len(np.unique(y)) <= 2:
        return screen_univariate(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        F, p = f_classif(X, y)
    F = np.nan_to_num(np.asarray(F, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    p = np.where(np.isfinite(p), p, 1.0)
    return F, p


def variance_filter(X: np.ndarray, min_var: float = 1e-8) -> np.ndarray:
    """Boolean mask of features with non-trivial variance."""
    return X.var(axis=0) > min_var


def classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], labels: Optional[Sequence[int]] = None
) -> Dict:
    """Confusion-matrix-derived classification metrics for binary or multi-class.

    Binary → accuracy, sensitivity (recall of the positive class), specificity,
    precision, recall, F1. Multi-class → accuracy, macro precision/recall/F1 and
    per-class recall (sensitivity). Always returns the confusion matrix so a
    judge/clinician can read class-imbalance behaviour that AUC/F1 alone hide."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if labels is None:
        labels = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    labels = [int(l) for l in labels]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    out: Dict = {
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": labels,
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
    }
    if len(labels) == 2:
        tn, fp, fn, tp = cm.ravel()
        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        out.update({
            "sensitivity": round(float(sens), 4),
            "specificity": round(float(spec), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        })
    else:
        row_sums = cm.sum(axis=1)
        per_class = {
            int(labels[i]): (round(float(cm[i, i] / row_sums[i]), 4) if row_sums[i] else None)
            for i in range(len(labels))
        }
        out.update({
            "precision_macro": round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "recall_macro": round(float(recall_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
            "per_class_recall": per_class,
        })
    return out


# ── core discovery ──────────────────────────────────────────────────────────
def _impute_median(Xtr: np.ndarray, *others: np.ndarray):
    """Leakage-safe median imputation: medians are learned from the TRAINING fold
    only, then applied to the training fold and any held-out arrays. Real
    MetaboLights MAFs routinely carry a few missing cells; without this the
    downstream scaler / logistic-regression reject NaN and the whole pass fails.
    Datasets with no missing values are unaffected (nothing to fill)."""
    med = np.nanmedian(Xtr, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)  # all-NaN column → neutral 0

    def fill(A):
        if A is None or np.isfinite(A).all():
            return A
        A = A.copy()
        bad = ~np.isfinite(A)
        A[bad] = np.take(med, np.where(bad)[1])
        return A

    return tuple(fill(A) for A in (Xtr, *others))


def _select_in_fold(Xtr, ytr, k, fdr) -> np.ndarray:
    """Leakage-safe feature selection using ONLY the training fold.

    Binary → point-biserial screen; multi-class → ANOVA-F (via screen_features)."""
    (Xtr,) = _impute_median(Xtr)
    keep = np.where(variance_filter(Xtr))[0]
    if keep.size == 0:
        return np.arange(min(k, Xtr.shape[1]))
    absr, pvals = screen_features(Xtr[:, keep], ytr)
    sig = benjamini_hochberg(pvals, fdr)
    cand = keep[sig] if sig.any() else keep[np.argsort(absr)[-k:]]
    # rank survivors by score, take top-k
    absr_c, _ = screen_features(Xtr[:, cand], ytr)
    sel = cand[np.argsort(absr_c)[-k:]]
    return sel


def _fit_eval(Xtr, ytr, Xte, sel):
    """Binary path: probability of the positive class for held-out samples."""
    Xtr, Xte = _impute_median(Xtr[:, sel], Xte[:, sel])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(sc.transform(Xtr), ytr)
    return clf.predict_proba(sc.transform(Xte))[:, 1]


def _fit_eval_multiclass(Xtr, ytr, Xte, sel, classes):
    """Multi-class path: per-class probabilities (columns aligned to `classes`)."""
    Xtr, Xte = _impute_median(Xtr[:, sel], Xte[:, sel])
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), ytr)
    proba = clf.predict_proba(sc.transform(Xte))
    classes = list(classes)
    out = np.zeros((Xte.shape[0], len(classes)), dtype=float)
    for j, c in enumerate(clf.classes_):
        out[:, classes.index(int(c))] = proba[:, j]
    return out


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
    """One leakage-safe CV pass → ROC-AUC (used by the permutation test).

    Binary → standard ROC-AUC; multi-class → macro one-vs-rest ROC-AUC."""
    classes = sorted(np.unique(y).tolist())
    multiclass = len(classes) > 2
    skf = (StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
           if groups is not None
           else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed))
    if multiclass:
        oof = np.full((len(y), len(classes)), np.nan)
        for tr, te in skf.split(X, y, groups):
            sel = _select_in_fold(X[tr], y[tr], k, fdr)
            oof[te] = _fit_eval_multiclass(X[tr], y[tr], X[te], sel, classes)
        return roc_auc_score(y, oof, multi_class="ovr", average="macro", labels=classes)
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
    p = (1 + #{perm AUC ≥ real}) / (n_perm + 1). Works for binary and multi-class
    (the AUC statistic itself adapts).
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


def _panel_effect_sizes(X, y, stable_idx, counts, name_fn) -> List[Dict]:
    """Per-biomarker direction, fold-change and single-marker AUC for the panel.

    The positive class (y == 1) is the reference "condition" group, so a result
    reads as e.g. "higher in the positive class, 2.3x, single-marker AUC 0.81".
    Means use finite values only, so a partially-missing metabolite still gets an
    honest effect size instead of being silently dropped. This turns the panel
    from a list of names into an actual finding — which way each marker moves and
    how strongly it separates the two groups on its own.
    """
    y = np.asarray(y, dtype=int)
    pos, neg = y == 1, y == 0
    out: List[Dict] = []
    for i in stable_idx:
        col = np.asarray(X[:, i], dtype=float)
        finite = np.isfinite(col)
        cp, cn = col[pos & finite], col[neg & finite]
        if cp.size == 0 or cn.size == 0:
            continue
        mp, mn = float(np.mean(cp)), float(np.mean(cn))
        entry = {
            "metabolite": name_fn(i),
            "selected_in_folds": int(counts.get(i, 0)),
            "mean_positive": round(mp, 4),
            "mean_negative": round(mn, 4),
            "direction": "higher_in_positive" if mp >= mn else "higher_in_negative",
            "fold_change": None,
            "log2_fold_change": None,
            "univariate_auc": None,
        }
        if mp > 0 and mn > 0:
            fc = mp / mn
            entry["fold_change"] = round(fc, 3)
            entry["log2_fold_change"] = round(float(np.log2(fc)), 3)
        yy = y[finite]
        if len(np.unique(yy)) == 2:
            raw = roc_auc_score(yy, col[finite])
            # orient to >= 0.5 so it reads as "separation strength"; direction is
            # reported separately from the class means above.
            entry["univariate_auc"] = round(float(max(raw, 1.0 - raw)), 3)
        out.append(entry)
    out.sort(key=lambda e: (e["univariate_auc"] is not None, e["univariate_auc"] or 0.0),
             reverse=True)
    return out


def _panel_effect_sizes_multiclass(X, y, stable_idx, counts, name_fn, classes) -> List[Dict]:
    """Multi-class panel effect sizes: per-class mean for each stable marker and
    the class in which it is highest (its descriptive "up in" group)."""
    y = np.asarray(y, dtype=int)
    classes = [int(c) for c in classes]
    masks = {c: (y == c) for c in classes}
    out: List[Dict] = []
    for i in stable_idx:
        col = np.asarray(X[:, i], dtype=float)
        finite = np.isfinite(col)
        class_means = {}
        for c in classes:
            vals = col[masks[c] & finite]
            class_means[c] = round(float(np.mean(vals)), 4) if vals.size else None
        present = {c: m for c, m in class_means.items() if m is not None}
        highest = max(present, key=present.get) if present else None
        out.append({
            "metabolite": name_fn(i),
            "selected_in_folds": int(counts.get(i, 0)),
            "class_means": class_means,
            "highest_in_class": highest,
        })
    out.sort(key=lambda e: e["selected_in_folds"], reverse=True)
    return out


def _repeated_cv(X, y, k, fdr, n_splits, repeats, seed, groups_array,
                 multiclass, classes) -> Dict:
    """One block of repeated stratified (optionally patient-grouped) CV.

    Returns per-repeat AUC/F1/Q², the per-fold selected feature sets (for
    stability), and the pooled out-of-fold true/predicted labels (for the
    confusion-matrix metrics). The binary branch is identical to the original
    engine so previously-reported numbers reproduce exactly."""
    fold_sets: List[set] = []
    rep_aucs, rep_f1s, rep_q2s = [], [], []
    pooled_true: List[int] = []
    pooled_pred: List[int] = []
    # accumulate out-of-fold prediction scores (averaged over repeats) so a
    # bootstrap confidence interval can be computed on the honest AUC
    oof_accum = np.zeros((len(y), len(classes))) if multiclass else np.zeros(len(y))
    for rep in range(repeats):
        skf = (
            StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
            if groups_array is not None
            else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + rep)
        )
        if multiclass:
            oof = np.full((len(y), len(classes)), np.nan)
            for tr, te in skf.split(X, y, groups_array):
                sel = _select_in_fold(X[tr], y[tr], k, fdr)
                fold_sets.append(set(sel.tolist()))
                oof[te] = _fit_eval_multiclass(X[tr], y[tr], X[te], sel, classes)
            rep_aucs.append(roc_auc_score(y, oof, multi_class="ovr", average="macro", labels=classes))
            pred = np.asarray(classes)[np.argmax(oof, axis=1)]
            rep_f1s.append(f1_score(y, pred, average="macro"))
            rep_q2s.append(float("nan"))
            pooled_true.extend(y.tolist())
            pooled_pred.extend(pred.tolist())
        else:
            oof = np.full(len(y), np.nan)
            for tr, te in skf.split(X, y, groups_array):
                sel = _select_in_fold(X[tr], y[tr], k, fdr)
                fold_sets.append(set(sel.tolist()))
                oof[te] = _fit_eval(X[tr], y[tr], X[te], sel)
            rep_aucs.append(roc_auc_score(y, oof))
            pred = (oof >= 0.5).astype(int)
            rep_f1s.append(f1_score(y, pred))
            rep_q2s.append(_q2_score(y, oof))
            pooled_true.extend(y.tolist())
            pooled_pred.extend(pred.tolist())
        oof_accum = oof_accum + oof
    return {
        "fold_sets": fold_sets,
        "rep_aucs": rep_aucs,
        "rep_f1s": rep_f1s,
        "rep_q2s": rep_q2s,
        "pooled_true": pooled_true,
        "pooled_pred": pooled_pred,
        "oof_scores": oof_accum / repeats,
    }


def _bootstrap_auc_ci(y, scores, classes, groups=None, n_boot: int = 1000,
                      seed: int = 0) -> Optional[List[float]]:
    """Percentile-bootstrap 95% CI for the (cross-validated) ROC-AUC.

    Resamples with replacement at the PATIENT/GROUP level when groups are given
    (so repeated measures of one subject move together), otherwise at the sample
    level. Standard way to express the sampling uncertainty of an AUC on small n.
    Binary → ROC-AUC; multi-class → macro one-vs-rest."""
    y = np.asarray(y)
    scores = np.asarray(scores)
    classes = list(classes)
    multiclass = len(classes) > 2
    rng = np.random.default_rng(seed)
    if groups is not None:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        gidx = {g: np.where(groups == g)[0] for g in uniq}
    aucs: List[float] = []
    for _ in range(n_boot):
        if groups is not None:
            gsamp = rng.choice(len(uniq), size=len(uniq), replace=True)
            idx = np.concatenate([gidx[uniq[g]] for g in gsamp])
        else:
            idx = rng.integers(0, len(y), size=len(y))
        yb = y[idx]
        if len(np.unique(yb)) < len(classes):
            continue
        sb = scores[idx]
        try:
            if multiclass:
                aucs.append(roc_auc_score(yb, sb, multi_class="ovr", average="macro", labels=classes))
            else:
                aucs.append(roc_auc_score(yb, sb))
        except Exception:
            continue
    if len(aucs) < 20:
        return None
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return [round(float(lo), 4), round(float(hi), 4)]


def _stable_from_folds(fold_sets: List[set]) -> Tuple[List[int], Counter, float]:
    """Stable panel (features chosen in ≥ half the folds), selection counts, and
    mean pairwise Jaccard stability of the per-fold selections."""
    def jac(a, b):
        return len(a & b) / max(1, len(a | b))
    pairs = [(i, j) for i in range(len(fold_sets)) for j in range(i + 1, len(fold_sets))]
    stability = float(np.mean([jac(fold_sets[i], fold_sets[j]) for i, j in pairs])) if pairs else 0.0
    cnt: Counter = Counter()
    for s in fold_sets:
        cnt.update(s)
    half = len(fold_sets) / 2
    stable_idx = sorted([f for f, c in cnt.items() if c >= half], key=lambda f: -cnt[f])
    return stable_idx, cnt, stability


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
    panel_sizes: Optional[Sequence[int]] = (1, 3, 5, 10),
    ci_boot: int = 1000,
) -> Dict:
    """
    Honest p>>n biomarker discovery. Returns AUC, F1, Q², permutation p-value,
    Top-k stability + VIP-ranked stable panel, the full confusion-matrix metric
    set (accuracy/sensitivity/specificity/precision/recall or their multi-class
    macro forms), a top-1/3/5/10 panel-size sweep, and the leaky AUC.

    Binary and multi-class (≥3 group) tasks are both supported.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    # Fail fast with a clear message if no feature has any real signal at all
    # (e.g. an empty/placeholder MetaboLights MAF where every concentration
    # cell is missing). Without this check, every column gets imputed to a
    # constant, variance_filter correctly drops all of them, and the CV loop
    # only discovers "zero features" several calls deep inside sklearn with a
    # cryptic error — this catches it upfront on the whole dataset instead.
    finite = np.isfinite(X)
    any_value = finite.any(axis=0)
    with np.errstate(invalid="ignore"):  # all-NaN columns are handled by any_value below
        col_std = np.nanstd(np.where(finite, X, np.nan), axis=0)
    has_signal = any_value & (col_std > 1e-8)
    if not has_signal.any():
        raise ValueError(
            "No metabolite has any real (non-missing, non-constant) concentration "
            "value across the provided samples — check that the concentrations "
            "table actually has data in its sample columns, not just metabolite names.")

    classes = sorted(np.unique(y).tolist())
    n_classes = len(classes)
    multiclass = n_classes > 2

    groups_array = np.asarray(groups) if groups is not None else None
    # int(...) here: min() against a numpy scalar (np.bincount(...).min()) returns
    # that numpy type unchanged when it's the smaller operand, which then fails
    # JSON serialization wherever n_splits ends up in the response (params).
    n_splits = int(min(n_splits, np.bincount(y).min()))  # guard tiny classes
    if groups_array is not None:
        per_class_groups = [
            len(np.unique(groups_array[y == klass])) for klass in np.unique(y)
        ]
        n_splits = int(min(n_splits, min(per_class_groups), len(np.unique(groups_array))))
    n_splits = max(2, n_splits)

    # ── main run at the requested k ─────────────────────────────────────────
    main = _repeated_cv(X, y, k, fdr, n_splits, repeats, seed, groups_array,
                        multiclass, classes)
    fold_sets = main["fold_sets"]
    honest_auc = float(np.mean(main["rep_aucs"]))
    honest_f1 = float(np.mean(main["rep_f1s"]))
    honest_q2 = None if multiclass else float(np.nanmean(main["rep_q2s"]))
    auc_ci = (_bootstrap_auc_ci(y, main["oof_scores"], classes, groups_array, ci_boot, seed)
              if ci_boot and ci_boot > 0 else None)

    stable_idx, cnt, stability = _stable_from_folds(fold_sets)

    # full confusion-matrix metric set from pooled out-of-fold predictions
    metrics = classification_metrics(main["pooled_true"], main["pooled_pred"], labels=classes)

    # leaky AUC (select on ALL data first → optimistic) for comparison
    leaky = leaky_auc(X, y, k, n_splits, seed, groups_array)

    def name(i):
        return feature_names[i] if feature_names is not None else f"feature_{i}"

    stable_names = [name(i) for i in stable_idx]

    # VIP (PLS-DA) ranking of the stable panel — binary only (PLS-DA is a
    # 2-class projection; multi-class uses selection-frequency ranking instead)
    vip_panel = {}
    if compute_vip and stable_idx and not multiclass:
        names_all = [name(i) for i in range(X.shape[1])]
        all_vip = _vip_scores(X, y, names_all)
        vip_panel = {nm: all_vip.get(nm) for nm in stable_names if nm in all_vip}

    # Permutation test p-value on the honest AUC (label-shuffle null)
    perm_p = None
    if permutations and permutations > 0:
        perm_p = _permutation_pvalue(
            X, y, k, fdr, n_splits, seed, groups_array, honest_auc, n_perm=permutations)

    if multiclass:
        panel_stats = _panel_effect_sizes_multiclass(X, y, stable_idx, cnt, name, classes)
    else:
        panel_stats = _panel_effect_sizes(X, y, stable_idx, cnt, name)

    # ── panel-size sweep: how few metabolites still separate the groups ─────
    panel_sweep: List[Dict] = []
    if panel_sizes:
        sizes = sorted({int(s) for s in list(panel_sizes) + [k] if 1 <= int(s) <= X.shape[1]})
        for ks in sizes:
            if ks == k:
                sw = main  # reuse — identical settings
            else:
                sw = _repeated_cv(X, y, ks, fdr, n_splits, repeats, seed,
                                  groups_array, multiclass, classes)
            sw_idx, _sw_cnt, _sw_stab = _stable_from_folds(sw["fold_sets"])
            entry = {
                "k": ks,
                "honest_roc_auc": round(float(np.mean(sw["rep_aucs"])), 4),
                "honest_f1": round(float(np.mean(sw["rep_f1s"])), 4),
                "n_stable_features": len(sw_idx),
                "stable_panel": [name(i) for i in sw_idx],
            }
            if not multiclass:
                entry["honest_q2"] = round(float(np.nanmean(sw["rep_q2s"])), 4)
            panel_sweep.append(entry)

    result = {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "task_type": "multiclass" if multiclass else "binary",
        "n_classes": int(n_classes),
        "class_labels": [int(c) for c in classes],
        "honest_roc_auc": round(honest_auc, 4),
        "honest_roc_auc_ci95": auc_ci,
        "honest_f1": round(honest_f1, 4),
        "honest_q2": round(honest_q2, 4) if honest_q2 is not None else None,
        "classification_metrics": metrics,
        "panel_sweep": panel_sweep,
        "permutation_p_value": round(perm_p, 4) if perm_p is not None else None,
        "n_permutations": int(permutations) if perm_p is not None else 0,
        "leaky_roc_auc": round(leaky, 4),
        "leakage_inflation": round(leaky - honest_auc, 4),
        "topk_stability_jaccard": round(stability, 4),
        "stable_panel": stable_names,
        "panel_stats": panel_stats,
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
            "VIP > 1 marks PLS-DA-influential metabolites; "
            "multi-class ROC-AUC is macro-averaged one-vs-rest; "
            "honest_roc_auc_ci95 is a group-level percentile bootstrap of the "
            "out-of-fold scores — accepted but APPROXIMATE (no unbiased estimator "
            "of K-fold CV variance exists; Bengio & Grandvalet 2004), so an "
            "independent external cohort remains the primary evidence of generalization."
        ),
        "params": {"k": k, "fdr": fdr, "n_splits": n_splits, "repeats": repeats},
    }
    return result


def leaky_auc(X, y, k, n_splits, seed, groups=None) -> float:
    """AUC when features are chosen on the FULL data before CV (data leakage).

    Binary → ROC-AUC; multi-class → macro one-vs-rest ROC-AUC. Diagnostic only —
    never quote this as performance."""
    classes = sorted(np.unique(y).tolist())
    multiclass = len(classes) > 2
    absr, _ = screen_features(X, y)              # uses all samples → leakage
    sel = np.argsort(absr)[-k:]
    skf = (
        StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if groups is not None
        else StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    if multiclass:
        oof = np.full((len(y), len(classes)), np.nan)
        for tr, te in skf.split(X, y, groups):
            oof[te] = _fit_eval_multiclass(X[tr], y[tr], X[te], sel, classes)
        return float(roc_auc_score(y, oof, multi_class="ovr", average="macro", labels=classes))
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
        res = discover(X, y, k=k, repeats=1, seed=s, permutations=0, panel_sizes=None, ci_boot=0)
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
