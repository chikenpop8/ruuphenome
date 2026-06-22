from __future__ import annotations

import os
import sys
import types
import unittest

import numpy as np

from backend.nmr_api import model_suite, nmrformer_backend


class ModelSuiteTests(unittest.TestCase):
    def test_patient_grouped_model_comparison_runs(self):
        rng = np.random.default_rng(4)
        patients = np.repeat(np.arange(60), 2)
        y_patient = rng.integers(0, 2, 60)
        y = np.repeat(y_patient, 2)
        X = rng.normal(size=(120, 24))
        X[:, :4] += 1.2 * y[:, None]

        result = model_suite.compare_models(
            X,
            y,
            patients,
            k=8,
            outer_splits=3,
            repeats=1,
            feature_names=[f"f{i}" for i in range(X.shape[1])],
        )

        names = {item["model"] for item in result["models"]}
        self.assertIn("elastic_net_logistic", names)
        self.assertIn("linear_svm", names)
        self.assertIn("pca_logistic", names)
        self.assertIn("pca_linear_svm", names)
        self.assertIn("hist_gradient_boosting", names)
        self.assertGreater(max(item["roc_auc_mean"] for item in result["models"]), 0.8)
        self.assertEqual(result["n_patients"], 60)
        pca_result = next(
            item for item in result["models"] if item["model"] == "pca_logistic"
        )
        self.assertEqual(pca_result["feature_mode"], "all")
        self.assertIn("inside each training fold", pca_result["dimensionality_reduction"])

    def test_nmrformer_adapter_and_hybrid_contract(self):
        module_name = "_ruuphenome_fake_nmrformer"
        fake = types.ModuleType(module_name)

        def predict_assignments(**_kwargs):
            return [{"metabolite": "citrate", "probability": 0.9}]

        fake.predict_assignments = predict_assignments
        sys.modules[module_name] = fake
        previous = os.environ.get(nmrformer_backend.ADAPTER_ENV)
        os.environ[nmrformer_backend.ADAPTER_ENV] = module_name
        try:
            neural = nmrformer_backend.predict(
                np.linspace(10, 0, 20), np.zeros(20), []
            )
            combined = nmrformer_backend.hybridize(
                [
                    {
                        "metabolite": "citrate",
                        "confidence": 80.0,
                        "confidence_label": "high",
                    }
                ],
                neural,
            )
            self.assertTrue(nmrformer_backend.status()["available"])
            self.assertEqual(neural[0]["confidence"], 90.0)
            self.assertTrue(combined[0]["neural_support"])
            self.assertAlmostEqual(combined[0]["confidence"], 83.5)
        finally:
            if previous is None:
                os.environ.pop(nmrformer_backend.ADAPTER_ENV, None)
            else:
                os.environ[nmrformer_backend.ADAPTER_ENV] = previous
            sys.modules.pop(module_name, None)


if __name__ == "__main__":
    unittest.main()
