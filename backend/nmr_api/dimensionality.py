"""
Exploratory PCA and UMAP projections for Domain 2.

These projections are intentionally separated from predictive validation.
PCA/UMAP are fitted to the requested cohort for visualization, batch/outlier
review and hypothesis generation; they must not be quoted as classifier
performance or external biological validation.
"""

from __future__ import annotations

import importlib.util
import warnings
from typing import Dict, Optional, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


def dependency_status() -> Dict:
    return {
        "pca": {
            "available": True,
            "implementation": "scikit-learn",
            "role": "linear variance, loadings, batch and outlier inspection",
        },
        "umap": {
            "available": importlib.util.find_spec("umap") is not None,
            "implementation": "umap-learn",
            "role": "exploratory nonlinear neighborhood visualization",
        },
    }


def _top_loadings(
    components: np.ndarray,
    feature_names: Sequence[str],
    component_index: int,
    top_k: int,
) -> list[Dict]:
    if component_index >= len(components):
        return []
    weights = components[component_index]
    order = np.argsort(np.abs(weights))[::-1][: min(top_k, len(weights))]
    return [
        {
            "feature": str(feature_names[index]),
            "loading": round(float(weights[index]), 6),
            "absolute_loading": round(float(abs(weights[index])), 6),
        }
        for index in order
    ]


def project(
    X: np.ndarray,
    labels: Sequence,
    *,
    sample_names: Optional[Sequence[str]] = None,
    patient_ids: Optional[Sequence] = None,
    feature_names: Optional[Sequence[str]] = None,
    include_umap: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    seed: int = 42,
    top_loadings: int = 10,
) -> Dict:
    """Return PCA scores/loadings and an optional UMAP visualization."""
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)
    if X.ndim != 2 or len(X) != len(labels):
        raise ValueError("X and labels must describe the same two-dimensional cohort.")
    if len(X) < 3 or X.shape[1] < 2:
        raise ValueError("PCA/UMAP requires at least 3 samples and 2 features.")
    if not 0.0 <= min_dist <= 1.0:
        raise ValueError("min_dist must be between 0 and 1.")

    samples = (
        list(sample_names)
        if sample_names is not None
        else [f"sample_{i}" for i in range(len(X))]
    )
    patients = list(patient_ids) if patient_ids is not None else list(samples)
    features = (
        list(feature_names)
        if feature_names is not None
        else [f"feature_{i}" for i in range(X.shape[1])]
    )
    if len(samples) != len(X) or len(patients) != len(X):
        raise ValueError("sample_names and patient_ids must match the sample count.")
    if len(features) != X.shape[1]:
        raise ValueError("feature_names must match the feature count.")

    imputed = SimpleImputer(strategy="median").fit_transform(X)
    standardized = StandardScaler().fit_transform(imputed)
    component_count = min(standardized.shape)
    pca = PCA(n_components=component_count, svd_solver="full")
    scores = pca.fit_transform(standardized)
    pc1 = scores[:, 0]
    pc2 = scores[:, 1] if scores.shape[1] > 1 else np.zeros(len(scores))

    umap_coordinates = None
    umap_status = dependency_status()["umap"]
    effective_neighbors = min(max(2, int(n_neighbors)), len(X) - 1)
    if include_umap and umap_status["available"]:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Tensorflow not installed; ParametricUMAP will be unavailable",
                category=ImportWarning,
            )
            import umap

        # PCA pre-reduction makes high-resolution NMR matrices tractable while
        # preserving substantially more information than a 2-D PCA display.
        pre_dimensions = min(50, scores.shape[1], len(X) - 1)
        umap_input = scores[:, :pre_dimensions]
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=effective_neighbors,
            min_dist=float(min_dist),
            metric="euclidean",
            random_state=seed,
            n_jobs=1,
        )
        umap_coordinates = reducer.fit_transform(umap_input)

    points = []
    for index in range(len(X)):
        point = {
            "sample": str(samples[index]),
            "patient": str(patients[index]),
            "group": str(labels[index]),
            "pc1": round(float(pc1[index]), 6),
            "pc2": round(float(pc2[index]), 6),
        }
        if umap_coordinates is not None:
            point.update(
                {
                    "umap1": round(float(umap_coordinates[index, 0]), 6),
                    "umap2": round(float(umap_coordinates[index, 1]), 6),
                }
            )
        points.append(point)

    # A zero-variance cohort makes explained_variance_ratio_ NaN, which is not
    # JSON-compliant (Starlette rejects it → HTTP 500). Sanitize to 0.0.
    explained = np.nan_to_num(
        np.asarray(pca.explained_variance_ratio_, dtype=float), nan=0.0
    )
    return {
        "n_samples": int(len(X)),
        "n_features": int(X.shape[1]),
        "groups": sorted({str(value) for value in labels}),
        "preprocessing": [
            "median imputation",
            "feature-wise standardization",
        ],
        "pca": {
            "available": True,
            "explained_variance_ratio": [
                round(float(value), 6) for value in explained[:20]
            ],
            "pc1_explained_percent": round(float(explained[0] * 100), 3),
            "pc2_explained_percent": round(
                float(explained[1] * 100) if len(explained) > 1 else 0.0,
                3,
            ),
            "cumulative_first_two_percent": round(
                float(explained[:2].sum() * 100), 3
            ),
            "top_loadings": {
                "PC1": _top_loadings(
                    pca.components_, features, 0, top_loadings
                ),
                "PC2": _top_loadings(
                    pca.components_, features, 1, top_loadings
                ),
            },
        },
        "umap": {
            **umap_status,
            "computed": umap_coordinates is not None,
            "n_neighbors": effective_neighbors,
            "min_dist": float(min_dist),
            "random_state": seed,
            "input": (
                f"first {min(50, scores.shape[1], len(X) - 1)} PCA components"
                if umap_coordinates is not None
                else None
            ),
        },
        "points": points,
        "interpretation_warning": (
            "PCA and UMAP are exploratory cohort visualizations fitted on the "
            "displayed samples. Apparent separation is not classifier accuracy "
            "or external biological validation; patient-grouped cross-validation "
            "must be used for predictive claims."
        ),
    }
