"""Track-1 F5 (diagnostic-ppm selector) + F7 (20k-bin NNLS scaling)."""

from __future__ import annotations

import os
import unittest

import numpy as np
import pandas as pd

from backend.nmr_api import spectral_cohort as sc


def _grid(lo, hi, step):
    return np.round(np.arange(lo, hi, step), 4)


class DiagnosticPpmSelectorTests(unittest.TestCase):
    def test_supervised_picks_discriminating_position(self):
        bins = _grid(0.4, 4.0, 0.004)                     # ~900 bins (native path)
        rng = np.random.default_rng(1)
        X = pd.DataFrame(np.abs(rng.normal(0.02, 0.005, size=(40, len(bins)))),
                         columns=bins, index=[f"s{i}" for i in range(40)])
        y = np.array([0] * 20 + [1] * 20)
        j = int(np.argmin(np.abs(bins - 1.33)))           # lactate CH3, up in cases
        X.iloc[20:, j - 2:j + 3] += 2.5
        sel = sc.select_diagnostic_ppm(X, bins, y=y, top_k=6)
        self.assertIn("supervised", sel["mode"])
        top = [p["ppm"] for p in sel["top_positions"]]
        self.assertTrue(any(abs(p - 1.33) < 0.02 for p in top))   # found the signal
        hit = [p for p in sel["top_positions"] if abs(p["ppm"] - 1.33) < 0.02][0]
        self.assertIsNotNone(hit["nearest_metabolite"])           # annotated

    def test_unsupervised_picks_high_signal_position(self):
        bins = _grid(0.4, 4.0, 0.004)
        rng = np.random.default_rng(2)
        X = pd.DataFrame(np.abs(rng.normal(0.02, 0.005, size=(10, len(bins)))),
                         columns=bins, index=[f"s{i}" for i in range(10)])
        j = int(np.argmin(np.abs(bins - 1.48)))           # alanine
        X.iloc[:, j - 2:j + 3] += 3.0
        sel = sc.select_diagnostic_ppm(X, bins, top_k=5)
        self.assertIn("unsupervised", sel["mode"])
        self.assertTrue(any(abs(p["ppm"] - 1.48) < 0.03 for p in sel["top_positions"]))

    def test_water_region_excluded(self):
        bins = _grid(0.4, 9.5, 0.01)
        rng = np.random.default_rng(3)
        X = pd.DataFrame(np.abs(rng.normal(0.02, 0.005, size=(8, len(bins)))),
                         columns=bins, index=[f"s{i}" for i in range(8)])
        j = int(np.argmin(np.abs(bins - 4.80)))           # residual water — must be skipped
        X.iloc[:, j - 2:j + 3] += 10.0
        sel = sc.select_diagnostic_ppm(X, bins, top_k=10)
        self.assertFalse(any(4.70 <= p["ppm"] <= 4.90 for p in sel["top_positions"]))


class AnnotateNoiseFloorTests(unittest.TestCase):
    """annotate()'s occupancy gate must be an ABSOLUTE noise floor — the old
    relative 0.75-quantile called ~25% of bins occupied on ANY input (103/578
    metabolites on pure noise, 575/578 on a sparse spectrum)."""

    def test_pure_noise_calls_almost_nothing(self):
        bins = _grid(0.5, 9.5, 0.0005)                    # ~18k bins
        rng = np.random.default_rng(0)
        X = pd.DataFrame(np.abs(rng.normal(0.02, 0.005, size=(8, len(bins)))),
                         columns=bins, index=[f"s{i}" for i in range(8)])
        a = sc.annotate(X, bins)
        self.assertLessEqual(a["n_metabolites_annotated"], 3)          # was 103
        d = sc.deconvolve(X, bins)
        self.assertEqual(sum(1 for m in d["metabolites"] if m["passes_fdr"]), 0)

    def test_sparse_spectrum_recovers_only_planted(self):
        bins = _grid(0.5, 9.5, 0.0008)
        rng = np.random.default_rng(1)
        X = pd.DataFrame(np.abs(rng.normal(0.004, 0.0015, size=(8, len(bins)))),
                         columns=bins, index=[f"s{i}" for i in range(8)])
        for shs in ([1.33, 4.11], [1.48, 3.78]):          # lactate, alanine (full coverage)
            for sh in shs:
                j = int(np.argmin(np.abs(bins - sh)))
                X.iloc[:, j - 1:j + 2] += 3.0
        a = sc.annotate(X, bins)
        called = {m["metabolite"] for m in a["metabolites"]}
        self.assertLess(a["n_metabolites_annotated"], 20)             # was 575/578
        self.assertIn("lactate", called)
        self.assertIn("alanine", called)


from backend.nmr_api import identification_quality as _idq


class D2OChemistryTests(unittest.TestCase):
    """The D2O assessment must be structure-aware: O/N/S-H protons exchange out in
    D2O and vanish; C-H persists. Grades must follow the real exchangeable inventory."""

    def test_exchangeable_proton_counts(self):
        # (name, exchangeable OH/NH/SH, non-exchangeable C-H)
        for name, ex, ch in [("glucose", 5, 7), ("urea", 4, 0), ("alanine", 3, 4),
                             ("citrate", 1, 4), ("creatinine", 2, 5)]:
            s = _idq.metabolite_exchangeable(name)
            self.assertIsNotNone(s, f"no structure for {name}")
            self.assertEqual(s["exchangeable"], ex, f"{name} exchangeable")
            self.assertEqual(s["nonexchangeable_ch"], ch, f"{name} C-H")

    def test_all_heteroatom_H_molecule_is_invisible(self):
        # urea: every proton is on N -> exchanges out -> not observable in D2O
        u = _idq.metabolite_exchangeable("urea")
        a = _idq.d2o_assessment([5.8], 1, structure=u)
        self.assertEqual(a["d2o_grade"], "invisible")
        self.assertFalse(a["usable_in_d2o"])

    def test_ch_bearing_molecule_is_reliable(self):
        g = _idq.metabolite_exchangeable("glucose")
        a = _idq.d2o_assessment([3.4, 3.8, 3.47], 5, structure=g)
        self.assertEqual(a["d2o_grade"], "reliable")
        self.assertTrue(a["usable_in_d2o"])
        self.assertEqual(a["structural_ch"], 7)
        self.assertAlmostEqual(a["observable_fraction"], round(7 / 12, 2))

    def test_no_structure_never_false_invisible(self):
        # unknown structure must degrade gracefully, never claim "invisible"
        a = _idq.d2o_assessment([3.4, 3.8], 3, structure=None)
        self.assertNotEqual(a["d2o_grade"], "invisible")
        self.assertIsNone(a["structural_exchangeable"])

    def test_annotate_propagates_structural_grade(self):
        bins = _grid(0.0, 9.99, 0.01)
        X = pd.DataFrame(np.zeros((4, len(bins))), columns=bins,
                         index=[f"s{i}" for i in range(4)])
        for p in (3.24, 3.40, 3.47, 3.71, 3.83, 4.63, 3.03, 4.05):
            j = int(np.argmin(np.abs(bins - p)))
            X.iloc[:, j] = 8.0
        a = sc.annotate(X, bins, min_coverage=0.05)
        graded = [m for m in a["metabolites"] if m["d2o"].get("d2o_grade")]
        self.assertTrue(graded, "no metabolite carried a structural d2o_grade")
        m = next(m for m in graded if m["metabolite"] == "glucose")
        self.assertEqual(m["d2o"]["structural_ch"], 7)
        self.assertIn(m["d2o"]["d2o_grade"], ("reliable", "caution"))


class SerumWhitelistTests(unittest.TestCase):
    """Matrix-gated physiological panel restricts the ~578 BMRB library to real
    serum/urine metabolites (cuts over-annotation); unknown matrix keeps the full lib."""

    def test_panel_restricts_known_matrix_only(self):
        serum = sc.panel_reference_shifts("serum")
        urine = sc.panel_reference_shifts("urine")
        self.assertIsNotNone(serum)
        self.assertIsNotNone(urine)
        self.assertLess(len(serum), len(sc.REFERENCE_SHIFTS))
        self.assertGreaterEqual(len(serum), 20)
        low = {k.lower() for k in serum}
        self.assertTrue({"glucose", "lactate", "alanine"} <= low)
        # unknown / blank / non-biofluid → full library (None)
        self.assertIsNone(sc.panel_reference_shifts("unknown"))
        self.assertIsNone(sc.panel_reference_shifts(""))
        self.assertIsNone(sc.panel_reference_shifts("plant extract"))

    def test_annotate_with_panel_cuts_candidates(self):
        bins = _grid(0.0, 9.99, 0.01)
        X = pd.DataFrame(np.zeros((4, len(bins))), columns=bins,
                         index=[f"s{i}" for i in range(4)])
        for p in (3.24, 3.40, 3.47, 3.71, 3.83, 1.33, 1.48):     # glucose/lactate/alanine-ish
            X.iloc[:, int(np.argmin(np.abs(bins - p)))] = 8.0
        full = sc.annotate(X, bins, min_coverage=0.05)
        serum = sc.annotate(X, bins, min_coverage=0.05,
                            reference_shifts=sc.panel_reference_shifts("serum"))
        self.assertLess(serum["reference_library_size"], full["reference_library_size"])
        self.assertLessEqual(serum["n_metabolites_annotated"], full["n_metabolites_annotated"])


class NnlsScalingTests(unittest.TestCase):
    def test_large_grid_downsamples_and_stays_correct(self):
        bins = np.round(np.linspace(0.4, 9.5, 4000), 5)   # > _NNLS_BIN_CAP → downsample
        rng = np.random.default_rng(4)
        X = pd.DataFrame(np.abs(rng.normal(0.01, 0.003, size=(6, 4000))),
                         columns=bins, index=[f"s{i}" for i in range(6)])
        for sh in (1.33, 0.98, 3.23, 5.23, 1.48):
            k = int(np.argmin(np.abs(bins - sh)))
            X.iloc[:, k - 10:k + 10] += 1.5
        res = sc.deconvolve(X, bins)
        self.assertGreater(res["mean_fit_r2"], 0.3)                 # sane fit
        self.assertEqual(res["concentrations"].shape[0], 6)
        self.assertFalse(any(m["metabolite"].startswith("decoy::")
                             for m in res["metabolites"]))          # no decoy leak
        self.assertEqual(len(res["fit_overlay"]["ppm"]),
                         len(res["fit_overlay"]["fitted"]))         # overlay grid consistent


from backend.nmr_api import pscnn as _pscnn
from backend.nmr_api import quantifier as _quant


class PSCNNTests(unittest.TestCase):
    @unittest.skipUnless(_pscnn.available(), "torch not available")
    def test_pscnn_trains_and_identifies(self):
        want = ["glucose", "lactate", "valine", "alanine", "citrate", "acetate"]
        panel = {n: sc.REFERENCE_SHIFTS[n] for n in want if n in sc.REFERENCE_SHIFTS}
        grid = _pscnn.make_grid(128)
        model, meta = _pscnn.train(panel, grid=grid, n_mixtures=100, epochs=20,
                                   lr=3e-3, batch_size=32, seed=0, save=False)
        h = meta["loss_history"]
        self.assertLess(h[-1], h[0] - 0.1)        # actually learned (not stuck at ln2)
        self.assertLess(h[-1], 0.55)              # deterministic now (torch seeded)
        present = ["glucose", "lactate", "valine"]
        samp = np.zeros(len(grid))
        for n in present:
            samp += _pscnn.fingerprint(panel[n], grid)
        samp += np.abs(np.random.default_rng(0).normal(0.02, 0.005, len(grid)))
        probs = _pscnn.identify((model, meta), grid, samp)
        self.assertEqual(set(probs), set(panel))
        self.assertGreaterEqual(sum(probs[n] > 0.5 for n in present), 2)  # recovers present


class BenchmarkHarnessTests(unittest.TestCase):
    def test_deterministic_baseline_runs(self):
        from backend.nmr_api import track1_benchmark as bm
        res = bm.run_baseline(n_cohorts=3, n_present=10, n_samples=4, n_bins=1000)
        self.assertIn("deconvolve_fdr_baseline", res)
        # FDR-controlled deconvolution is a strong baseline; annotate is permissive
        self.assertGreater(res["deconvolve_fdr_baseline"]["f1"], 0.6)
        self.assertGreater(res["annotate_baseline"]["recall"], 0.7)
        self.assertLess(res["annotate_baseline"]["precision"],
                        res["deconvolve_fdr_baseline"]["precision"])


class QuantifierTests(unittest.TestCase):
    @unittest.skipUnless(_quant.available(), "torch not available")
    def test_quantifier_trains_and_predicts(self):
        # network-free: use the reference library as a synthetic GISSMO-style corpus
        want = ["glucose", "lactate", "valine", "alanine", "citrate", "acetate", "tyrosine", "glycine"]
        corpus = {n: sc.REFERENCE_SHIFTS[n] for n in want if n in sc.REFERENCE_SHIFTS}
        grid = _pscnn.make_grid(256)
        model, meta = _quant.train(corpus, grid=grid, n_bins=256, epochs=8,
                                   steps_per_epoch=12, batch_size=32, patch=16, save=False)
        h = meta["loss_history"]
        self.assertLess(h[-1], h[0])                      # learned (MSE decreasing)
        present = list(corpus)[:3]
        spec = np.zeros(len(grid))
        for n in present:
            spec += _pscnn.fingerprint(corpus[n], grid)
        called = _quant.identify((model, meta), grid, spec, threshold=0.05)
        self.assertEqual(set(meta["names"]), set(corpus))
        self.assertGreaterEqual(len(set(called) & set(present)), 2)   # recovers present


class BmrbExperimentalBundleTests(unittest.TestCase):
    """Network-free checks on the bundled BMRB experimental peak lists."""

    def test_bundle_loads_with_real_landmarks(self):
        from backend.nmr_api import bmrb_experimental as be
        if not be.BUNDLE_PATH.exists():
            self.skipTest("BMRB experimental bundle not built (run be.build_bundle off-VM)")
        peaks = be.load_bundle()
        self.assertGreaterEqual(len(peaks), 12)               # solid real panel
        # every entry is [[ppm, intensity], ...] with sane ppm
        for name, pk in peaks.items():
            self.assertTrue(all(len(p) == 2 and 0.4 <= p[0] <= 10 for p in pk), name)

        def near(name, target, tol=0.06):
            return name in peaks and any(abs(p[0] - target) < tol for p in peaks[name])
        self.assertTrue(near("glucose", 5.23), "glucose anomeric ~5.23 missing")
        self.assertTrue(near("lactate", 1.33), "lactate CH3 ~1.33 missing")
        self.assertTrue(near("alanine", 1.47), "alanine CH3 ~1.47 missing")
        # BCAA triad present (the NCD/T2D signature)
        for bcaa, mth in (("valine", 0.99), ("leucine", 0.95), ("isoleucine", 0.94)):
            self.assertTrue(near(bcaa, mth, 0.08), f"{bcaa} methyl missing")


class BmrbValidationHarnessTests(unittest.TestCase):
    @unittest.skipUnless(_pscnn.available(), "torch not available")
    def test_bmrb_validation_runs_on_real_peaks(self):
        from backend.nmr_api import bmrb_experimental as be
        from backend.nmr_api import track1_benchmark as bm
        if not be.BUNDLE_PATH.exists():
            self.skipTest("BMRB experimental bundle not built")
        res = bm.run_bmrb_validation(n_test=4, n_bins=600, epochs=8, seed=1)
        self.assertNotIn("error", res)
        self.assertGreaterEqual(res["n_compounds"], 12)
        for cond, m in res["conditions"].items():
            for method in ("deterministic_permissive", "deterministic_fdr", "pscnn", "hybrid"):
                s = m[method]
                self.assertTrue(0.0 <= s["precision"] <= 1.0 and 0.0 <= s["f1"] <= 1.0, (cond, method))
        # real-data signature: FDR-gated deterministic is PRECISE but low-recall on real spectra
        clean = res["conditions"]["BMRB real spectra (clean)"]
        self.assertLess(clean["deterministic_fdr"]["recall"], clean["deterministic_permissive"]["recall"])


class F3FinetuneLoaderTests(unittest.TestCase):
    """F3 on-VM fine-tune loader — parses organizer annotations, is governance-gated,
    and the fine-tune path is callable on OPEN data (ready-to-use, not trained here)."""

    def test_parses_ppm_metabolite_annotation(self):
        from backend.nmr_api import finetune_loader as fl
        raw = b"ppm,metabolite\n5.23,glucose\n3.83,glucose\n1.33,lactate\n1.48,alanine\n"
        panel = fl.load_annotation_panel(raw)
        self.assertIn("glucose", panel)
        self.assertEqual(sorted(panel["glucose"]), [3.83, 5.23])
        self.assertIn("lactate", panel); self.assertIn("alanine", panel)

    def test_parses_compound_shift_table(self):
        from backend.nmr_api import finetune_loader as fl
        raw = b"metabolite\tshifts\nvaline\t0.99 1.04 2.28 3.62\ncitrate\t2.54 2.66\n"
        panel = fl.load_annotation_panel(raw)
        self.assertEqual(sorted(panel["valine"]), [0.99, 1.04, 2.28, 3.62])
        self.assertEqual(sorted(panel["citrate"]), [2.54, 2.66])

    def test_finetune_refuses_off_vm(self):
        from backend.nmr_api import finetune_loader as fl
        old = os.environ.pop("NMR_OFFLINE", None)
        try:
            with self.assertRaises(RuntimeError):
                fl.finetune_pscnn("/nonexistent.csv")   # gated before any file/train
        finally:
            if old is not None:
                os.environ["NMR_OFFLINE"] = old

    @unittest.skipUnless(_pscnn.available(), "torch not available")
    def test_finetune_path_runs_on_open_data(self):
        import tempfile
        from pathlib import Path
        from backend.nmr_api import finetune_loader as fl, pscnn as p
        # OPEN synthetic annotations derived from the reference library (no closed data)
        lines = ["ppm,metabolite"]
        for n in ("glucose", "lactate", "alanine", "valine"):
            for s in sc.REFERENCE_SHIFTS[n]:
                lines.append(f"{s},{n}")
        with tempfile.TemporaryDirectory() as tmp:
            ann = Path(tmp) / "annot.csv"; ann.write_bytes("\n".join(lines).encode())
            outp = Path(tmp) / "ft.pt"                       # NOT the serve checkpoint
            os.environ["NMR_OFFLINE"] = "1"
            try:
                rep = fl.finetune_pscnn(ann, out=outp, epochs=2, n_mixtures=80, seed=0)
            finally:
                os.environ.pop("NMR_OFFLINE", None)
            self.assertTrue(outp.exists())                  # wrote locally, to the temp path
            self.assertGreaterEqual(rep["n_annotated_compounds"], 4)
            model, meta = p.load_checkpoint(outp)           # produced a loadable checkpoint
            self.assertIn("glucose", meta["panel"])
        # the real serve checkpoint was untouched
        self.assertNotEqual(str(outp), str(p.CHECKPOINT_PATH))


class RealValidationWiringTests(unittest.TestCase):
    """Network-free checks for the GISSMO held-out validation wiring."""

    def test_library_mapping(self):
        from backend.nmr_api.track1_benchmark import _match_library
        refs = sc.REFERENCE_SHIFTS
        for c in ("glucose", "valine", "leucine", "alanine", "citrate", "tyrosine"):
            self.assertIsNotNone(_match_library(c, refs), c)

    def test_gissmo_ids_map_to_library(self):
        from backend.nmr_api import external_reference as ext
        from backend.nmr_api.track1_benchmark import _match_library
        refs = sc.REFERENCE_SHIFTS
        mapped = sum(1 for c in ext.GISSMO_IDS if _match_library(c, refs))
        self.assertGreaterEqual(mapped, 12)   # verified GISSMO compounds map to the library


if __name__ == "__main__":
    unittest.main()
