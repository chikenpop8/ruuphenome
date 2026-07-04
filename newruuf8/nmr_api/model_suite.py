"""
Leakage-safe multi-model evaluation for Domain 2.

Every outer test fold holds out complete patients. Feature selection, median
imputation and model tuning happen only inside the corresponding training data.
This keeps nonlinear challengers honest instead of rewarding leakage.

Supports BINARY and MULTI-CLASS (≥3 group) tasks. Multi-class uses macro
one-vs-rest ROC-AUC + macro-F1 + per-class recall and a confusion matrix; Brier
and calibration error (binary-only concepts) are reported for binary tasks only.
"""

from __future__ import annotations

import importlib.util
import os
import warnings
from collections import Counter
from typing import Callable, Dict, List, Optional, Sequence

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"
warnings.filterwarnings(
    "ignore",
    message=".*probability.*deprecated.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Could not find the number of physical cores.*",
    category=UserWarning,
)

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .biomarker_engine import _select_in_fold, classification_metrics


def dependency_status() -> Dict[str, Dict]:
    return {
        "elastic_net_logistic": {"available": True, "role": "interpretable baseline"},
        "linear_svm": {"available": True, "role": "high-dimensional linear challenger"},
        "pca_logistic": {
            "available": True,
            "role": "all-feature PCA baseline with fold-internal dimensionality reduction",
        },
        "pca_linear_svm": {
            "available": True,
            "role": "all-feature PCA linear-margin challenger",
        },
        "random_forest": {"available": True, "role": "bagged tree challenger"},
        "hist_gradient_boosting": {"available": True, "role": "nonlinear sklearn challenger"},
        "xgboost": {
            "available": importlib.util.find_spec("xgboost") is not None,
            "role": "regularized nonlinear tree challenger",
        },
        "catboost": {
            "available": importlib.util.find_spec("catboost") is not None,
            "eligible": False,
            "role": "reserved for future categorical clinical metadata",
        },
        "domain2_deep_learning": {
            "available": importlib.util.find_spec("torch") is not None,
            "eligible": False,
            "role": "disabled for the current small cohort; use pretrained spectral models first",
        },
    }


def _model_specs(seed: int, multiclass: bool = False) -> Dict[str, Dict]:
    specs: Dict[str, Dict] = {
        "elastic_net_logistic": {
            "feature_mode": "selected",
            "builders": [
            lambda c=c, ratio=ratio: Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            solver="saga",
                            C=c,
                            l1_ratio=ratio,
                            class_weight="balanced",
                            max_iter=5000,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            for c, ratio in ((0.1, 0.0), (0.3, 0.5), (1.0, 0.5), (0.3, 1.0))
            ],
        },
        "linear_svm": {
            "feature_mode": "selected",
            "builders": [
            lambda c=c: Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "model",
                        SVC(
                            C=c,
                            kernel="linear",
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            )
            for c in (0.05, 0.2, 1.0)
            ],
        },
        "pca_logistic": {
            "feature_mode": "all",
            "builders": [
                lambda: Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        (
                            "pca",
                            PCA(
                                n_components=0.95,
                                svd_solver="full",
                            ),
                        ),
                        (
                            "model",
                            LogisticRegression(
                                C=0.3,
                                class_weight="balanced",
                                max_iter=5000,
                                random_state=seed,
                            ),
                        ),
                    ]
                )
            ],
        },
        "pca_linear_svm": {
            "feature_mode": "all",
            "builders": [
                lambda: Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                        (
                            "pca",
                            PCA(
                                n_components=0.95,
                                svd_solver="full",
                            ),
                        ),
                        (
                            "model",
                            SVC(
                                C=0.2,
                                kernel="linear",
                                class_weight="balanced",
                                random_state=seed,
                            ),
                        ),
                    ]
                )
            ],
        },
        "random_forest": {
            "feature_mode": "selected",
            "builders": [
            lambda leaf=leaf: Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "model",
                        RandomForestClassifier(
                            n_estimators=300,
                            min_samples_leaf=leaf,
                            max_features="sqrt",
                            class_weight="balanced",
                            n_jobs=1,
                            random_state=seed,
                        ),
                    ),
                ]
            )
            for leaf in (2, 4)
            ],
        },
        "hist_gradient_boosting": {
            "feature_mode": "selected",
            "builders": [
            lambda leaves=leaves, l2=l2: Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            learning_rate=0.05,
                            max_iter=150,
                            max_leaf_nodes=leaves,
                            min_samples_leaf=12,
                            l2_regularization=l2,
                            class_weight="balanced",
                            random_state=seed,
                        ),
                    ),
                ]
            )
            for leaves, l2 in ((7, 1.0), (15, 2.0))
            ],
        },
    }
    if importlib.util.find_spec("xgboost") is not None:
        from xgboost import XGBClassifier

        # For multi-class let XGBoost infer objective (multi:softprob) from y;
        # forcing binary:logistic would break ≥3-class fitting.
        xgb_obj = {} if multiclass else {"objective": "binary:logistic", "eval_metric": "logloss"}
        specs["xgboost"] = {
            "feature_mode": "selected",
            "builders": [
                lambda depth=depth, child=child: Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        (
                            "model",
                            XGBClassifier(
                                n_estimators=180,
                                learning_rate=0.04,
                                max_depth=depth,
                                min_child_weight=child,
                                subsample=0.8,
                                colsample_bytree=0.8,
                                reg_alpha=0.5,
                                reg_lambda=2.0,
                                tree_method="hist",
                                n_jobs=1,
                                random_state=seed,
                                **xgb_obj,
                            ),
                        ),
                    ]
                )
                for depth, child in ((2, 5), (3, 8))
            ],
        }
    return specs


def _predict_scores(model, X: np.ndarray, classes: Sequence) -> np.ndarray:
    """Held-out scores. Binary → P(positive) as a 1-D vector; multi-class →
    per-class probabilities aligned to `classes` (n_samples × n_classes)."""
    classes = list(classes)
    if hasattr(model, "predict_proba"):
        proba = np.asarray(model.predict_proba(X))
        if len(classes) == 2:
            return proba[:, 1]
        out = np.zeros((X.shape[0], len(classes)), dtype=float)
        for j, c in enumerate(model.classes_):
            out[:, classes.index(int(c))] = proba[:, j]
        return out
    # decision_function fallback (linear SVM): binary → sigmoid; multi → softmax
    score = np.asarray(model.decision_function(X), dtype=float)
    if len(classes) == 2:
        return 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))
    order = list(getattr(model, "classes_", classes))
    e = np.exp(score - score.max(axis=1, keepdims=True))
    soft = e / e.sum(axis=1, keepdims=True)
    out = np.zeros((X.shape[0], len(classes)), dtype=float)
    for j, c in enumerate(order):
        out[:, classes.index(int(c))] = soft[:, j]
    return out


def _auc(y: np.ndarray, scores: np.ndarray, classes: Sequence) -> float:
    if len(classes) == 2:
        return float(roc_auc_score(y, scores))
    return float(roc_auc_score(y, scores, multi_class="ovr", average="macro", labels=list(classes)))


def _labels_from_scores(scores: np.ndarray, classes: Sequence) -> np.ndarray:
    classes = list(classes)
    if len(classes) == 2:
        return (scores >= 0.5).astype(int)
    return np.asarray(classes)[np.argmax(scores, axis=1)]


def _safe_group_splits(y: np.ndarray, groups: np.ndarray, requested: int) -> int:
    unique_groups = np.unique(groups)
    per_class_groups = [
        len(np.unique(groups[y == klass])) for klass in np.unique(y)
    ]
    return max(2, min(requested, len(unique_groups), min(per_class_groups)))


def _inner_auc(
    builder: Callable[[], object],
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    k: int,
    fdr: float,
    seed: int,
    feature_mode: str,
    classes: Sequence,
) -> float:
    splits = _safe_group_splits(y, groups, 3)
    cv = StratifiedGroupKFold(n_splits=splits, shuffle=True, random_state=seed)
    scores = []
    for train, valid in cv.split(X, y, groups):
        selected = (
            _select_in_fold(X[train], y[train], min(k, X.shape[1]), fdr)
            if feature_mode == "selected"
            else np.arange(X.shape[1])
        )
        model = builder()
        model.fit(X[train][:, selected], y[train])
        scores.append(_auc(y[valid], _predict_scores(model, X[valid][:, selected], classes), classes))
    return float(np.mean(scores))


def _expected_calibration_error(y: np.ndarray, probabilities: np.ndarray, bins: int = 8) -> float:
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        mask = (probabilities >= left) & (
            probabilities <= right if right == 1 else probabilities < right
        )
        if mask.any():
            error += (
                mask.mean()
                * abs(float(probabilities[mask].mean()) - float(y[mask].mean()))
            )
    return float(error)


def compare_models(
    X: np.ndarray,
    y: np.ndarray,
    groups: Sequence,
    *,
    k: int = 8,
    fdr: float = 0.05,
    outer_splits: int = 5,
    repeats: int = 2,
    seed: int = 0,
    feature_names: Optional[Sequence[str]] = None,
) -> Dict:
    """
    Nested, patient-grouped comparison of conservative linear and nonlinear models.
    Supports binary and multi-class outcomes.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    if X.ndim != 2 or len(X) != len(y) or len(groups) != len(y):
        raise ValueError("X, y and groups must describe the same samples.")
    classes = sorted(np.unique(y).tolist())
    if len(classes) < 2:
        raise ValueError("At least two outcome classes are required.")
    multiclass = len(classes) > 2

    names = list(feature_names or [f"feature_{i}" for i in range(X.shape[1])])
    specs = _model_specs(seed, multiclass=multiclass)
    model_outputs = {
        name: {
            "repeat_auc": [],
            "repeat_f1": [],
            "repeat_brier": [],
            "repeat_ece": [],
            "selected": Counter(),
            "inner_choice_counts": Counter(),
            "pooled_true": [],
            "pooled_pred": [],
        }
        for name in specs
    }
    n_splits = _safe_group_splits(y, groups, outer_splits)

    for repeat in range(repeats):
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + repeat
        )
        if multiclass:
            repeat_scores = {mn: np.full((len(y), len(classes)), np.nan) for mn in specs}
        else:
            repeat_scores = {mn: np.full(len(y), np.nan) for mn in specs}
        for fold, (train, test) in enumerate(cv.split(X, y, groups)):
            selected = _select_in_fold(
                X[train], y[train], min(k, X.shape[1]), fdr
            )
            for model_name, spec in specs.items():
                builders = spec["builders"]
                feature_mode = spec["feature_mode"]
                inner_scores = [
                    _inner_auc(
                        builder,
                        X[train],
                        y[train],
                        groups[train],
                        min(k, X.shape[1]),
                        fdr,
                        seed + repeat * 100 + fold,
                        feature_mode,
                        classes,
                    )
                    for builder in builders
                ]
                choice = int(np.argmax(inner_scores))
                model = builders[choice]()
                model_features = (
                    selected
                    if feature_mode == "selected"
                    else np.arange(X.shape[1])
                )
                model.fit(X[train][:, model_features], y[train])
                repeat_scores[model_name][test] = _predict_scores(
                    model, X[test][:, model_features], classes
                )
                if feature_mode == "selected":
                    model_outputs[model_name]["selected"].update(selected.tolist())
                model_outputs[model_name]["inner_choice_counts"][choice] += 1

        for model_name, scores in repeat_scores.items():
            model_outputs[model_name]["repeat_auc"].append(_auc(y, scores, classes))
            pred = _labels_from_scores(scores, classes)
            model_outputs[model_name]["repeat_f1"].append(
                f1_score(y, pred, average="macro" if multiclass else "binary")
            )
            model_outputs[model_name]["pooled_true"].extend(y.tolist())
            model_outputs[model_name]["pooled_pred"].extend(pred.tolist())
            if not multiclass:
                model_outputs[model_name]["repeat_brier"].append(
                    brier_score_loss(y, scores)
                )
                model_outputs[model_name]["repeat_ece"].append(
                    _expected_calibration_error(y, scores)
                )

    results = []
    total_folds = repeats * n_splits
    for model_name, output in model_outputs.items():
        selected_indices = (
            [
                index
                for index, count in output["selected"].most_common()
                if count >= total_folds / 2
            ]
            if specs[model_name]["feature_mode"] == "selected"
            else []
        )
        aucs = np.asarray(output["repeat_auc"])
        metrics = classification_metrics(
            output["pooled_true"], output["pooled_pred"], labels=classes
        )
        entry = {
            "model": model_name,
            "roc_auc_mean": round(float(aucs.mean()), 4),
            "roc_auc_std": round(float(aucs.std()), 4),
            "f1_mean": round(float(np.mean(output["repeat_f1"])), 4),
            "brier_mean": (
                round(float(np.mean(output["repeat_brier"])), 4)
                if output["repeat_brier"] else None
            ),
            "calibration_error_mean": (
                round(float(np.mean(output["repeat_ece"])), 4)
                if output["repeat_ece"] else None
            ),
            "classification_metrics": metrics,
            "confusion_matrix": metrics["confusion_matrix"],
            "probability_method": (
                "sigmoid-transformed decision score"
                if "svm" in model_name
                else "native predict_proba"
            ),
            "feature_mode": specs[model_name]["feature_mode"],
            "dimensionality_reduction": (
                "PCA retaining 95% variance, fitted inside each training fold"
                if model_name.startswith("pca_")
                else None
            ),
            "stable_panel": [names[index] for index in selected_indices],
            "selection_counts": {
                names[index]: int(output["selected"][index])
                for index in selected_indices
            },
        }
        results.append(entry)

    complexity_order = {
        "elastic_net_logistic": 0,
        "linear_svm": 1,
        "pca_logistic": 1,
        "pca_linear_svm": 2,
        "random_forest": 3,
        "hist_gradient_boosting": 3,
        "xgboost": 4,
    }
    best_auc = max(result["roc_auc_mean"] for result in results)
    competitive = [
        result for result in results if result["roc_auc_mean"] >= best_auc - 0.01
    ]
    recommended = min(
        competitive,
        key=lambda result: (
            complexity_order.get(result["model"], 99),
            result["brier_mean"] if result["brier_mean"] is not None else 1.0,
        ),
    )
    results.sort(key=lambda result: (-result["roc_auc_mean"],
                                     result["brier_mean"] if result["brier_mean"] is not None else 1.0))

    return {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_patients": int(len(np.unique(groups))),
        "task_type": "multiclass" if multiclass else "binary",
        "n_classes": int(len(classes)),
        "class_labels": [int(c) for c in classes],
        "validation": {
            "outer_cv": "repeated stratified patient-grouped",
            "inner_cv": "patient-grouped model tuning",
            "feature_selection": (
                "inside every training fold for sparse/raw-feature models"
            ),
            "pca": (
                "median imputation, scaling and PCA are fitted inside every "
                "training fold for PCA challengers"
            ),
            "outer_splits": n_splits,
            "repeats": repeats,
        },
        "recommended_model": recommended["model"],
        "recommendation_rule": (
            "Choose the simplest model within 0.01 ROC-AUC of the best challenger; "
            "external validation is still required."
        ),
        "models": results,
        "availability": dependency_status(),
    }
