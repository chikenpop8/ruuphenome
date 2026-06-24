from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backend.nmr_api import spectral_cohort as sc


class SpectralCohortTests(unittest.TestCase):
    def test_demo_cohort_shapes_and_labels(self):
        X, bins, labels = sc.make_demo_binned(n_per_group=15)
        self.assertEqual(X.shape[0], 30)
        self.assertEqual(len(labels), 30)
        self.assertEqual(sum(labels.values()), 15)  # 15 cases

    def test_loader_round_trip_and_orientation(self):
        X, bins, _ = sc.make_demo_binned(n_per_group=5)
        # samples-as-rows layout
        X2, ppm2 = sc.load_binned_matrix(X.to_csv().encode())
        self.assertEqual(X2.shape, X.shape)
        # transposed (bins-as-rows) layout must be auto-corrected
        X3, ppm3 = sc.load_binned_matrix(X.T.to_csv().encode())
        self.assertEqual(X3.shape, X.shape)

    def test_pqn_normalization_preserves_shape(self):
        X, bins, _ = sc.make_demo_binned(n_per_group=8)
        Xn = sc.pqn_normalize(X)
        self.assertEqual(Xn.shape, X.shape)
        self.assertFalse(np.isnan(Xn.values).any())

    def test_annotation_recovers_known_metabolites(self):
        X, bins, _ = sc.make_demo_binned(n_per_group=10)
        res = sc.annotate(sc.pqn_normalize(X), bins)
        names = {m["metabolite"] for m in res["metabolites"]}
        # BCAAs and glucose are planted into every sample
        for expected in ("valine", "leucine", "isoleucine", "glucose"):
            self.assertIn(expected, names)
        self.assertEqual(res["annotated_matrix"].shape[0], X.shape[0])

    def test_metadata_join_explicit_column(self):
        ids = [f"s{i}" for i in range(8)]
        meta = pd.DataFrame({
            "Sample Name": ids,
            "Condition": ["control"] * 5 + ["disease"] * 3,
        })
        lm, info = sc.derive_labels(meta, ids, label_column="Condition")
        self.assertEqual(info["label_column"], "Condition")
        self.assertEqual(len(lm), 8)
        self.assertEqual(set(lm.values()), {0, 1})

    def test_parse_identified_peaks_csv_and_tsv(self):
        for sep in (",", "\t"):
            raw = (f"ppm{sep}metabolite\n1.33{sep}lactate\n"
                   f"3.04{sep}creatinine\n").encode()
            pins = sc.parse_identified_peaks(raw)
            self.assertEqual(pins.get(1.33), "lactate")
            self.assertEqual(pins.get(3.04), "creatinine")

    def test_identified_peaks_feed_annotation(self):
        X, bins, _ = sc.make_demo_binned(n_per_group=8)
        pins = {1.92: "acetate"}   # pin an extra assignment
        res = sc.annotate(sc.pqn_normalize(X), bins, identified_peaks=pins)
        self.assertIn("acetate", {m["metabolite"] for m in res["metabolites"]})

    def test_nnls_deconvolution_quantifies_and_fdr(self):
        X, bins, _ = sc.make_demo_binned(n_per_group=12)
        res = sc.deconvolve(sc.pqn_normalize(X), bins)
        # good fit and a sensible number passing FDR
        self.assertGreater(res["mean_fit_r2"], 0.5)
        self.assertGreater(res["n_passing_fdr"], 3)
        self.assertEqual(res["concentrations"].shape[0], X.shape[0])
        # planted core metabolites should be quantified and pass FDR
        passing = {m["metabolite"] for m in res["metabolites"] if m["passes_fdr"]}
        for expected in ("valine", "leucine", "isoleucine", "glucose"):
            self.assertIn(expected, passing)
        # decoys must never appear in target output
        self.assertFalse(any(m["metabolite"].startswith("decoy::")
                             for m in res["metabolites"]))

    def test_full_pipeline_recovers_planted_signal(self):
        X, bins, labels = sc.make_demo_binned(n_per_group=20)
        res = sc.annotate(sc.pqn_normalize(X), bins)
        M = res["annotated_matrix"]
        # BCAA columns should be higher in cases than controls (planted signal)
        cases = [s for s in M.index if str(s).startswith("case")]
        ctrls = [s for s in M.index if str(s).startswith("ctrl")]
        self.assertGreater(M.loc[cases, "valine"].mean(),
                           M.loc[ctrls, "valine"].mean())


if __name__ == "__main__":
    unittest.main()
