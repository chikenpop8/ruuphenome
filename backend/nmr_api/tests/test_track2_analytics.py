"""Track-2 analytics: full metrics + confusion matrix, top-k panel sweep,
multi-class support, differential analysis, and correlation/GGM networks."""

from __future__ import annotations

import unittest

import numpy as np

from backend.nmr_api import biomarker_engine as be
from backend.nmr_api import correlation as co
from backend.nmr_api import differential as dv
from backend.nmr_api import model_suite as ms


def _binary(n=80, p=12, seed=3):
    rng = np.random.default_rng(seed)
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    X = rng.normal(size=(n, p))
    X[y == 1, :3] += 2.0
    return X, y


def _multiclass(n_per=25, p=15, seed=5):
    rng = np.random.default_rng(seed)
    y = np.array([0] * n_per + [1] * n_per + [2] * n_per)
    X = rng.normal(size=(3 * n_per, p))
    X[y == 0, 0:2] += 2.4
    X[y == 1, 2:4] += 2.4
    X[y == 2, 4:6] += 2.4
    return X, y


class MetricsTests(unittest.TestCase):
    def test_binary_confusion_metric_set(self):
        X, y = _binary()
        r = be.discover(X, y, k=5, repeats=2, permutations=0,
                        feature_names=[f"m{i}" for i in range(X.shape[1])])
        m = r["classification_metrics"]
        for key in ("accuracy", "sensitivity", "specificity", "precision", "recall", "f1"):
            self.assertIn(key, m)
            self.assertTrue(0.0 <= m[key] <= 1.0)
        self.assertEqual(np.array(m["confusion_matrix"]).shape, (2, 2))
        self.assertEqual(r["task_type"], "binary")

    def test_panel_size_sweep(self):
        X, y = _binary()
        r = be.discover(X, y, k=8, repeats=2, permutations=0,
                        feature_names=[f"m{i}" for i in range(X.shape[1])],
                        panel_sizes=(1, 3, 5, 10))
        ks = [e["k"] for e in r["panel_sweep"]]
        for want in (1, 3, 5, 10):
            self.assertIn(want, ks)
        for e in r["panel_sweep"]:
            self.assertTrue(0.0 <= e["honest_roc_auc"] <= 1.0)

    def test_bootstrap_ci_present_and_brackets_auc(self):
        X, y = _binary()
        r = be.discover(X, y, k=5, repeats=3, permutations=0, ci_boot=300,
                        feature_names=[f"m{i}" for i in range(X.shape[1])])
        ci = r["honest_roc_auc_ci95"]
        self.assertIsInstance(ci, list)
        self.assertEqual(len(ci), 2)
        self.assertLessEqual(ci[0], ci[1])
        # the point estimate should sit within (or at the edge of) its own CI
        self.assertLessEqual(ci[0] - 0.02, r["honest_roc_auc"])
        self.assertLessEqual(r["honest_roc_auc"], ci[1] + 0.02)

    def test_ci_disabled(self):
        X, y = _binary(n=40, p=8)
        r = be.discover(X, y, k=4, repeats=1, permutations=0, ci_boot=0,
                        feature_names=[f"m{i}" for i in range(8)])
        self.assertIsNone(r["honest_roc_auc_ci95"])

    def test_multiclass_discover(self):
        X, y = _multiclass()
        r = be.discover(X, y, k=6, repeats=2, permutations=0,
                        feature_names=[f"m{i}" for i in range(X.shape[1])])
        self.assertEqual(r["task_type"], "multiclass")
        self.assertEqual(r["n_classes"], 3)
        self.assertGreater(r["honest_roc_auc"], 0.8)      # macro OvR AUC
        m = r["classification_metrics"]
        self.assertIn("per_class_recall", m)
        self.assertEqual(len(m["per_class_recall"]), 3)
        self.assertEqual(np.array(m["confusion_matrix"]).shape, (3, 3))
        self.assertIsNone(r["honest_q2"])                 # Q² is binary-only


class ModelSuiteExtrasTests(unittest.TestCase):
    def test_random_forest_and_confusion(self):
        rng = np.random.default_rng(4)
        patients = np.repeat(np.arange(60), 2)
        y = np.repeat(rng.integers(0, 2, 60), 2)
        X = rng.normal(size=(120, 20))
        X[:, :4] += 1.2 * y[:, None]
        res = ms.compare_models(X, y, patients, k=8, outer_splits=3, repeats=1,
                                feature_names=[f"f{i}" for i in range(20)])
        names = {m["model"] for m in res["models"]}
        self.assertIn("random_forest", names)
        for m in res["models"]:
            self.assertEqual(np.array(m["confusion_matrix"]).shape, (2, 2))
            self.assertIn("classification_metrics", m)

    def test_multiclass_model_suite(self):
        rng = np.random.default_rng(6)
        patients = np.repeat(np.arange(60), 2)
        yp = rng.integers(0, 3, 60)
        y = np.repeat(yp, 2)
        X = rng.normal(size=(120, 18))
        for c in (0, 1, 2):
            X[y == c, c * 2:c * 2 + 2] += 1.8
        res = ms.compare_models(X, y, patients, k=8, outer_splits=3, repeats=1,
                                feature_names=[f"f{i}" for i in range(18)])
        self.assertEqual(res["task_type"], "multiclass")
        self.assertEqual(res["n_classes"], 3)
        for m in res["models"]:
            self.assertEqual(np.array(m["confusion_matrix"]).shape, (3, 3))


class DifferentialTests(unittest.TestCase):
    def test_binary_differential_volcano(self):
        rng = np.random.default_rng(1)
        y = np.array([0] * 30 + [1] * 30)
        X = rng.normal(5, 1, size=(60, 8))
        X[y == 1, :2] += 2.0
        res = dv.differential_analysis(X, y, [f"m{i}" for i in range(8)],
                                       class_names={0: "control", 1: "case"})
        self.assertEqual(res["task_type"], "binary")
        self.assertGreaterEqual(res["n_significant"], 2)
        self.assertIsNotNone(res["volcano"])
        top = res["table"][0]
        self.assertIsNotNone(top["log2_fold_change"])
        self.assertIsNotNone(top["q_value"])

    def test_multiclass_differential(self):
        X, y = _multiclass()
        res = dv.differential_analysis(X, y, [f"m{i}" for i in range(X.shape[1])])
        self.assertEqual(res["task_type"], "multiclass")
        self.assertIn("kruskal", res["test"])
        self.assertIsNone(res["volcano"])          # volcano is a 2-group concept
        self.assertGreaterEqual(res["n_significant"], 3)


class CorrelationTests(unittest.TestCase):
    def test_ggm_removes_indirect_edge(self):
        # x0 → x1 → x2 chain; partial correlation must drop the indirect x0–x2 edge
        rng = np.random.default_rng(0)
        x0 = rng.normal(size=200)
        x1 = x0 + rng.normal(0, 0.5, 200)
        x2 = x1 + rng.normal(0, 0.5, 200)
        X = np.column_stack([x0, x1, x2])
        P = co.partial_correlation(X)
        self.assertGreater(abs(P[0, 1]), 0.3)      # direct
        self.assertGreater(abs(P[1, 2]), 0.3)      # direct
        self.assertLess(abs(P[0, 2]), 0.2)         # indirect → conditionally independent

    def test_analyze_and_covariate(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(40, 6))
        X[:, 1] = X[:, 0] * 0.9 + rng.normal(0, 0.3, 40)
        names = [f"m{i}" for i in range(6)]
        res = co.analyze(X, names, method="spearman")
        self.assertIn("correlation_matrix", res)
        self.assertIn("network", res)
        self.assertEqual(res["network"]["kind"], "gaussian_graphical_model (Ledoit-Wolf shrinkage)")
        cov = X[:, 0] * 2 + rng.normal(0, 0.5, 40)
        cc = co.covariate_correlation(X, names, cov, "age")
        self.assertEqual(cc["table"][0]["metabolite"], "m0")   # strongest covariate link

    def test_pn_guard(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(30, 200))
        res = co.analyze(X, [f"b{i}" for i in range(200)])
        self.assertLessEqual(res["n_metabolites"], 80)         # capped
        self.assertTrue(res["warnings"])


if __name__ == "__main__":
    unittest.main()
