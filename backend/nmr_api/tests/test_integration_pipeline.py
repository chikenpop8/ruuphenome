"""
End-to-end Track-1 → Track-2 integration + binned-loader robustness (network-free).

Mirrors the `/spectral/pipeline-file` endpoint spine (load_binned_matrix →
extract_embedded_labels → _run_cohort_pipeline) so a realistic organizer-style
binned file — ~20k ppm bins, either orientation, "ppm"-suffixed headers, an inline
label column — cannot silently break the demo. Also asserts the result is
JSON-encodable exactly as the FastAPI endpoint returns it.
"""

from __future__ import annotations

import io
import time
import unittest

import numpy as np
import pandas as pd
from fastapi.encoders import jsonable_encoder

from backend.nmr_api import main
from backend.nmr_api import spectral_cohort as sc

_PLANTED = {1.33: 2.0, 1.47: 2.2, 0.95: 1.8, 5.23: 2.6, 3.05: 1.6, 2.54: 1.5, 3.78: 1.4}


def _binned_csv(*, n_samples=16, n_bins=2600, lo=0.5, hi=9.5, sep=",",
                ppm_suffix=False, transpose=False, label_col=None, seed=0):
    """A realistic binned matrix as raw CSV/TSV bytes (+ the true labels)."""
    rng = np.random.default_rng(seed)
    bins = np.round(np.linspace(lo, hi, n_bins), 5)
    X = np.abs(rng.normal(0.02, 0.005, size=(n_samples, n_bins)))
    for sh, amp in _PLANTED.items():
        j = int(np.argmin(np.abs(bins - sh)))
        X[:, max(0, j - 1):j + 2] += amp
    y = np.array([1] * (n_samples // 2) + [0] * (n_samples - n_samples // 2))
    jb = int(np.argmin(np.abs(bins - 0.95)))          # BCAA up in cases (a real signal)
    X[y == 1, max(0, jb - 1):jb + 2] += 1.6
    cols = [f"{b}ppm" if ppm_suffix else f"{b}" for b in bins]
    df = pd.DataFrame(X, index=[f"sample_{i}" for i in range(n_samples)], columns=cols)
    if label_col:
        df.insert(0, label_col, ["case" if v else "control" for v in y])
    if transpose:
        df = df.T
    buf = io.StringIO()
    df.to_csv(buf, sep=sep)
    return buf.getvalue().encode(), y


class BinnedLoaderRobustnessTests(unittest.TestCase):
    def test_orientation_suffix_and_separator_agree(self):
        base_raw, _ = _binned_csv(n_bins=1500, seed=1)
        Xa, binsa = sc.load_binned_matrix(base_raw)
        # bins-as-rows, TSV, "ppm"-suffixed headers → must reduce to the same thing
        alt_raw, _ = _binned_csv(n_bins=1500, sep="\t", ppm_suffix=True, transpose=True, seed=1)
        Xb, binsb = sc.load_binned_matrix(alt_raw)
        self.assertEqual(Xa.shape, Xb.shape)
        self.assertTrue(np.allclose(binsa, binsb))
        self.assertEqual(list(Xa.index), list(Xb.index))     # samples, not bins

    def test_inline_label_column_dropped_from_matrix_but_found(self):
        raw, y = _binned_csv(n_bins=1200, label_col="Group", seed=2)
        X, bins = sc.load_binned_matrix(raw)
        self.assertEqual(X.shape[1], 1200)                    # label col not counted as a bin
        self.assertTrue(all(isinstance(c, float) for c in X.columns))
        label_map, info = sc.extract_embedded_labels(raw, [str(s) for s in X.index])
        self.assertIsNotNone(label_map)
        self.assertEqual(info["label_column"], "Group")
        self.assertEqual(len(set(label_map.values())), 2)


class EndToEndPipelineTests(unittest.TestCase):
    def test_one_file_pipeline_is_complete_and_json_safe(self):
        raw, _ = _binned_csv(n_samples=16, n_bins=2600, label_col="Condition", seed=3)
        X, bins = sc.load_binned_matrix(raw)
        label_map, info = sc.extract_embedded_labels(raw, [str(s) for s in X.index])
        self.assertIsNotNone(label_map)
        out = main._run_cohort_pipeline(X, bins, label_map=label_map,
                                        include_biomarkers=True)
        # every stage produced output
        ann = out["annotation"]
        self.assertGreater(ann["n_metabolites_annotated"], 3)
        m0 = ann["metabolites"][0]
        for f in ("msi_level", "d2o", "provenance", "robust_coverage"):
            self.assertIn(f, m0)                              # Track-1 quality carried through
        self.assertIn("quantification", out)
        self.assertIn("diagnostic_ppm", out)                  # feature selection ran
        self.assertGreater(out["diagnostic_ppm"]["n_selected"], 0)
        self.assertIn("biomarkers", out)                      # Track-2 ran (labels present)
        self.assertIn("ncd_relevance", out)
        # the endpoint returns this as JSON — it must encode without error
        encoded = jsonable_encoder(out)
        self.assertEqual(encoded["annotation"]["metabolites"][0]["metabolite"],
                         m0["metabolite"])

    def test_no_label_runs_annotation_only(self):
        raw, _ = _binned_csv(n_samples=6, n_bins=1200, seed=4)   # no label column
        X, bins = sc.load_binned_matrix(raw)
        label_map, info = sc.extract_embedded_labels(raw, [str(s) for s in X.index])
        self.assertIsNone(label_map)
        out = main._run_cohort_pipeline(X, bins, label_map=None, include_biomarkers=False)
        self.assertIn("annotation", out)
        self.assertIn("diagnostic_ppm", out)
        self.assertNotIn("biomarkers", out)                   # correctly skipped

    def test_twenty_thousand_bins_scales(self):
        """The organizer file is ~20k ppm bins — the pipeline must not choke."""
        raw, _ = _binned_csv(n_samples=6, n_bins=20000, seed=5)
        X, bins = sc.load_binned_matrix(raw)
        self.assertEqual(X.shape, (6, 20000))
        t0 = time.time()
        out = main._run_cohort_pipeline(X, bins, label_map=None, include_biomarkers=False)
        self.assertLess(time.time() - t0, 90)                 # F7 downsample keeps it tractable
        self.assertGreater(out["annotation"]["n_metabolites_annotated"], 3)
        self.assertIn("diagnostic_ppm", out)


if __name__ == "__main__":
    unittest.main()
