"""
Leakage-safe multi-model evaluation for Domain 2.

Every outer test fold holds out complete patients. Feature selection, median
imputation and model tuning happen only inside the corresponding training data.
This keeps nonlinear challengers honest instead of rewarding leakage.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .biomarker_engine import _select_in_fold


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


def _model_specs(seed: int) -> Dict[str, Dict]:
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
                                objective="binary:logistic",
                                eval_metric="logloss",
                                tree_method="hist",
                                n_jobs=1,
                                random_state=seed,
                            ),
                        ),
                    ]
                )
                for depth, child in ((2, 5), (3, 8))
            ],
        }
    return specs


def _predict_probability(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))[:, 1]
    score = np.asarray(model.decision_function(X), dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(score, -30, 30)))


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
        scores.append(roc_auc_score(y[valid], _predict_probability(model, X[valid][:, selected])))
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
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    groups = np.asarray(groups)
    if X.ndim != 2 or len(X) != len(y) or len(groups) != len(y):
        raise ValueError("X, y and groups must describe the same samples.")
    if len(np.unique(y)) != 2:
        raise ValueError("Exactly two outcome classes are required.")

    names = list(feature_names or [f"feature_{i}" for i in range(X.shape[1])])
    specs = _model_specs(seed)
    model_outputs = {
        name: {
            "repeat_auc": [],
            "repeat_f1": [],
            "repeat_brier": [],
            "repeat_ece": [],
            "selected": Counter(),
            "inner_choice_counts": Counter(),
        }
        for name in specs
    }
    n_splits = _safe_group_splits(y, groups, outer_splits)

    for repeat in range(repeats):
        cv = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=seed + repeat
        )
        repeat_probabilities = {
            model_name: np.full(len(y), np.nan) for model_name in specs
        }
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
                repeat_probabilities[model_name][test] = _predict_probability(
                    model, X[test][:, model_features]
                )
                if feature_mode == "selected":
                    model_outputs[model_name]["selected"].update(selected.tolist())
                model_outputs[model_name]["inner_choice_counts"][choice] += 1

        for model_name, probabilities in repeat_probabilities.items():
            model_outputs[model_name]["repeat_auc"].append(
                roc_auc_score(y, probabilities)
            )
            model_outputs[model_name]["repeat_f1"].append(
                f1_score(y, probabilities >= 0.5)
            )
            model_outputs[model_name]["repeat_brier"].append(
                brier_score_loss(y, probabilities)
            )
            model_outputs[model_name]["repeat_ece"].append(
                _expected_calibration_error(y, probabilities)
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
        results.append(
            {
                "model": model_name,
                "roc_auc_mean": round(float(aucs.mean()), 4),
                "roc_auc_std": round(float(aucs.std()), 4),
                "f1_mean": round(float(np.mean(output["repeat_f1"])), 4),
                "brier_mean": round(float(np.mean(output["repeat_brier"])), 4),
                "calibration_error_mean": round(
                    float(np.mean(output["repeat_ece"])), 4
                ),
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
        )

    complexity_order = {
        "elastic_net_logistic": 0,
        "linear_svm": 1,
        "pca_logistic": 1,
        "pca_linear_svm": 2,
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
            result["brier_mean"],
        ),
    )
    results.sort(key=lambda result: (-result["roc_auc_mean"], result["brier_mean"]))

    return {
        "n_samples": int(len(y)),
        "n_features": int(X.shape[1]),
        "n_patients": int(len(np.unique(groups))),
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
