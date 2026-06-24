from __future__ import annotations

import unittest

import numpy as np

from backend.nmr_api import biomarker_engine as be


def _separable_cohort(n=80, p=12, seed=3):
    """Two classes with a real signal in the first 3 features."""
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    X = rng.normal(size=(n, p))
    X[y == 1, :3] += 2.0
    return X, y


class BiomarkerValidationTests(unittest.TestCase):
    def test_q2_permutation_vip_present_and_sane(self):
        X, y = _separable_cohort()
        names = [f"m{i}" for i in range(X.shape[1])]
        res = be.discover(X, y, k=5, repeats=2, feature_names=names, permutations=50)
        # Q² should be finite and positive for a separable cohort
        self.assertFalse(np.isnan(res["honest_q2"]))
        self.assertGreater(res["honest_q2"], 0.0)
        # permutation p-value should be small (real signal) and in (0, 1]
        self.assertIsNotNone(res["permutation_p_value"])
        self.assertLessEqual(res["permutation_p_value"], 0.2)
        self.assertGreater(res["permutation_p_value"], 0.0)
        # VIP present for stable-panel features, >1 for the true signal features
        self.assertTrue(res["vip_scores"])
        strong = [v for k, v in res["vip_scores"].items() if k in ("m0", "m1", "m2")]
        self.assertTrue(any(v > 1.0 for v in strong))

    def test_permutation_pvalue_high_for_null(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(60, 10))
        y = np.array([0, 1] * 30)          # no real association
        res = be.discover(X, y, k=4, repeats=2, feature_names=[f"m{i}" for i in range(10)],
                          permutations=50)
        # a null cohort should NOT be strongly significant
        self.assertGreater(res["permutation_p_value"], 0.05)

    def test_permutations_zero_skips_test(self):
        X, y = _separable_cohort(n=40, p=8)
        res = be.discover(X, y, k=4, repeats=1, permutations=0,
                          feature_names=[f"m{i}" for i in range(8)])
        self.assertIsNone(res["permutation_p_value"])
        self.assertEqual(res["n_permutations"], 0)


if __name__ == "__main__":
    unittest.main()
