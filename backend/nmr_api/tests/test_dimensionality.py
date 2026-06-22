from __future__ import annotations

import unittest

import numpy as np

from backend.nmr_api import dimensionality


class DimensionalityTests(unittest.TestCase):
    def test_pca_projection_returns_scores_variance_and_loadings(self):
        rng = np.random.default_rng(12)
        labels = np.repeat(["baseline", "follow-up"], 20)
        X = rng.normal(size=(40, 10))
        X[20:, 0] += 2.5
        X[20:, 1] -= 1.5

        result = dimensionality.project(
            X,
            labels,
            sample_names=[f"s{i}" for i in range(len(X))],
            patient_ids=np.repeat(np.arange(20), 2),
            feature_names=[f"metabolite_{i}" for i in range(X.shape[1])],
            include_umap=False,
        )

        self.assertEqual(len(result["points"]), 40)
        self.assertGreater(result["pca"]["pc1_explained_percent"], 10)
        self.assertFalse(result["umap"]["computed"])
        loaded = {
            item["feature"] for item in result["pca"]["top_loadings"]["PC1"]
        }
        self.assertIn("metabolite_0", loaded)
        self.assertTrue(all("pc1" in point and "pc2" in point for point in result["points"]))

    def test_umap_projection_is_available_and_reproducible(self):
        if not dimensionality.dependency_status()["umap"]["available"]:
            self.skipTest("umap-learn is not installed.")
        rng = np.random.default_rng(3)
        X = rng.normal(size=(24, 7))
        labels = np.repeat(["A", "B"], 12)
        X[12:, :2] += 1.8

        first = dimensionality.project(
            X, labels, n_neighbors=6, seed=9
        )
        second = dimensionality.project(
            X, labels, n_neighbors=6, seed=9
        )

        self.assertTrue(first["umap"]["computed"])
        coordinates_a = np.array(
            [[point["umap1"], point["umap2"]] for point in first["points"]]
        )
        coordinates_b = np.array(
            [[point["umap1"], point["umap2"]] for point in second["points"]]
        )
        np.testing.assert_allclose(coordinates_a, coordinates_b, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
