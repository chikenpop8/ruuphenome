"""LiCO loader tests — reproduce the EXACT dataset schema with synthetic fixtures so
the loader is verified off-VM (the real nmr-pattern/ data can only be read on the VM).

Fixtures mirror: transposed ppm×Var spectra, positional Var↔metadata-row join,
serum bracketed boolean labels, urine DUPLICATE 'Factor Value' columns, and QC +
dilution samples that must be excluded.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.nmr_api import lico_loader as L


def _write_spectra(path: Path, sample_ids, n_ppm=10, seed=0):
    """Write spectra_intensity_ppm.csv as (ppm rows × Var columns)."""
    rng = np.random.default_rng(seed)
    ppm = np.round(np.linspace(0.5, 9.5, n_ppm), 4)
    cols = ["ppm"] + list(sample_ids)
    lines = [",".join(cols)]
    for i in range(n_ppm):
        row = [f"{ppm[i]:.4f}"] + [f"{abs(rng.normal(1, 0.2)):.4f}" for _ in sample_ids]
        lines.append(",".join(row))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class SerumLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "nmr-pattern"
        sd = root / "Human_Serum"
        sd.mkdir(parents=True)
        # 6 samples, Var1..Var6 (order == metadata row order)
        _write_spectra(sd / "spectra_intensity_ppm.csv", [f"Var{i}" for i in range(1, 7)])
        meta = ("Sample Name,Characteristics[Organism part],"
                "Factor Value[Rheumatoid arthritis],Factor Value[Anti-TNF therapy]\n"
                "1,Serum,TRUE,TRUE\n"
                "2,Serum,TRUE,FALSE\n"
                "3,Serum,TRUE,TRUE\n"
                "4,Serum,FALSE,FALSE\n"
                "5,Serum,FALSE,TRUE\n"
                "6,Serum,FALSE,FALSE\n")
        (sd / "MTBLS6213_Metadata.csv").write_text(meta, encoding="utf-8")
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def test_serum_loads_transposed_and_labels(self):
        ds = L.load_serum(self.root)
        self.assertEqual(ds["matrix"], "serum")
        self.assertEqual(ds["n_samples"], 6)                 # transposed correctly
        self.assertEqual(ds["sample_ids"][0], "Var1")
        # RA task: TRUE→1 (has RA), 3 TRUE / 3 FALSE
        ra = ds["tasks"]["rheumatoid_arthritis"]
        self.assertEqual("Factor Value[Rheumatoid arthritis]", ra["label_column"])
        self.assertEqual(ra["label_map"]["Var1"], 1)
        self.assertEqual(ra["label_map"]["Var4"], 0)
        self.assertEqual(sum(ra["label_map"].values()), 3)
        # second task available
        self.assertIn("anti_tnf_therapy", ds["tasks"])
        self.assertEqual(ds["tasks"]["anti_tnf_therapy"]["label_map"]["Var2"], 0)

    def test_positional_count_mismatch_fails_loud(self):
        # drop a metadata row → 5 rows vs 6 samples → must raise, not silently mis-align
        p = self.root / "Human_Serum" / "MTBLS6213_Metadata.csv"
        lines = p.read_text().splitlines()
        p.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            L.load_serum(self.root)


class UrineLoaderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "nmr-pattern"
        ud = root / "Human_Urine"
        ud.mkdir(parents=True)
        _write_spectra(ud / "spectra_intensity_ppm.csv", [f"Var{i}" for i in range(1, 9)])
        # DUPLICATE 'Factor Value' columns: col1 = sex/QC, col2 = condition
        meta = ("Sample Name,Characteristics[Organism part],Factor Value,Factor Value\n"
                "u1,urine,Male,Control Group\n"
                "u2,urine,Female,diabetes mellitus\n"
                "u3,urine,Male,diabetes mellitus\n"
                "u4,urine,Female,Control Group\n"
                "u5,urine,Study Pool 01,Study Pool 01\n"          # QC
                "u6,urine,External Reference 01,External Reference 01\n"  # QC
                "u7,urine,Male,Dilution_50_%\n"                   # dilution
                "u8,urine,Female,Control Group\n")
        (ud / "MTBLS1_MTBLS694_metadata.csv").write_text(meta, encoding="utf-8")
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def test_urine_excludes_qc_and_labels_condition(self):
        ds = L.load_urine(self.root)
        self.assertEqual(ds["matrix"], "urine")
        # Var5 (pool), Var6 (external ref), Var7 (dilution) excluded → 5 kept
        self.assertEqual(ds["n_qc_excluded"], 3)
        self.assertEqual(ds["n_samples"], 5)
        # condition labelled on retained samples: 3 Control / 2 diabetes, diabetes=1
        diab = ds["tasks"]["diabetes"]
        self.assertEqual(diab["positive_class"], "diabetes mellitus")
        self.assertEqual(sum(diab["label_map"].values()), 2)
        self.assertEqual(diab["label_map"]["Var2"], 1)
        self.assertEqual(diab["label_map"]["Var1"], 0)
        # sex covariate retained for kept samples only
        self.assertEqual(ds["sex"]["Var1"], "Male")
        self.assertNotIn("Var5", ds["sex"])

    def test_build_supervised_set_shapes(self):
        ds = L.load_urine(self.root)
        X, y, ppm, feats = L.build_supervised_set(ds, "diabetes")
        self.assertEqual(X.shape[0], len(y))
        self.assertEqual(X.shape[0], 5)
        self.assertEqual(len(ppm), X.shape[1])
        self.assertTrue(set(y) <= {0, 1})


class SerumResolutionTests(unittest.TestCase):
    """Serum can arrive at a DIFFERENT ppm range/resolution than urine — the loader
    must report each grid faithfully (bin width, ppm range) without assuming one grid."""

    def _serum_root(self, ppm_lo, ppm_hi, n_ppm):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name) / "nmr-pattern"
        sd = root / "Human_Serum"
        sd.mkdir(parents=True)
        ids = [f"Var{i}" for i in range(1, 7)]
        ppm = np.round(np.linspace(ppm_lo, ppm_hi, n_ppm), 5)
        lines = [",".join(["ppm"] + ids)]
        rng = np.random.default_rng(0)
        for i in range(n_ppm):
            lines.append(",".join([f"{ppm[i]:.5f}"] + [f"{abs(rng.normal(1, .2)):.4f}" for _ in ids]))
        (sd / "spectra_intensity_ppm.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (sd / "MTBLS6213_Metadata.csv").write_text(
            "Sample Name,Characteristics[Organism part],Factor Value[Rheumatoid arthritis]\n"
            + "".join(f"{i},Serum,{'TRUE' if i <= 3 else 'FALSE'}\n" for i in range(1, 7)),
            encoding="utf-8")
        return root

    def test_coarse_vs_fine_grid_reported(self):
        coarse = L.load_serum(self._serum_root(0.2, 10.0, 250))     # ~0.039 ppm/bin
        fine = L.load_serum(self._serum_root(0.5, 9.5, 4000))       # ~0.0022 ppm/bin
        self.assertGreater(coarse["profile"]["bin_width_est"], 0.02)
        self.assertTrue(coarse["profile"]["coarse_bins"])
        self.assertLess(fine["profile"]["bin_width_est"], 0.005)
        self.assertFalse(fine["profile"]["coarse_bins"])
        self.assertEqual(coarse["profile"]["ppm_range"], [0.2, 10.0])
        self.assertEqual(fine["n_samples"], 6)


class SerumTrack1RunTests(unittest.TestCase):
    """Run the real Track-1 pipeline on serum and check SAFE metrics + pSCNN/hybrid
    status (fallback-safe: passes whether or not a checkpoint/torch is present)."""

    def setUp(self):
        try:
            from backend.nmr_api import spectral_cohort as sc  # noqa
        except Exception as e:  # pragma: no cover
            self.skipTest(f"pipeline deps unavailable: {e}")
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name) / "nmr-pattern"
        sd = root / "Human_Serum"
        sd.mkdir(parents=True)
        from backend.nmr_api import spectral_cohort as sc
        ids = [f"Var{i}" for i in range(1, 21)]                     # 20 samples
        ppm = np.round(np.arange(0.5, 9.5, 0.006), 5)              # ~1500 fine bins
        rng = np.random.default_rng(3)
        M = np.abs(rng.normal(0.02, 0.005, size=(len(ppm), len(ids))))
        for name in ("glucose", "lactate", "alanine", "citrate", "creatinine", "glycine"):
            for sh in list(sc.REFERENCE_SHIFTS.get(name, []))[:4]:
                j = int(np.argmin(np.abs(ppm - sh)))
                if abs(ppm[j] - sh) < 0.006:
                    M[j, :] += rng.uniform(3.0, 6.0)
        lines = [",".join(["ppm"] + ids)]
        for i in range(len(ppm)):
            lines.append(",".join([f"{ppm[i]:.5f}"] + [f"{M[i, k]:.4f}" for k in range(len(ids))]))
        (sd / "spectra_intensity_ppm.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
        (sd / "MTBLS6213_Metadata.csv").write_text(
            "Sample Name,Characteristics[Organism part],Factor Value[Rheumatoid arthritis],Factor Value[Anti-TNF therapy]\n"
            + "".join(f"{i},Serum,{'TRUE' if i <= 10 else 'FALSE'},{'TRUE' if i % 2 else 'FALSE'}\n"
                      for i in range(1, 21)), encoding="utf-8")
        self.root = root

    def tearDown(self):
        self.tmp.cleanup()

    def test_run_track1_safe_metrics_and_hybrid(self):
        ds = L.load_serum(self.root)
        m = L.run_track1(ds, task="rheumatoid_arthritis")
        # SAFE metrics present and well-typed
        self.assertEqual(m["matrix"], "serum")
        self.assertEqual(m["n_samples"], 20)
        self.assertIsInstance(m["n_ppm_bins"], int)
        self.assertEqual(m["label_counts"], {"FALSE": 10, "TRUE": 10})   # literal, faithful to source
        self.assertIsInstance(m["n_fdr_confirmed"], int)
        self.assertIn(m["confidence"], ("low", "high"))
        self.assertIsInstance(m["hybrid_active"], bool)
        self.assertIsInstance(m["pscnn_status"], dict)     # present whether or not loaded
        self.assertTrue(m["concentration_csv_export"]["ok"])
        # NO raw data leaked in the safe metrics
        for k in ("X", "spectra", "concentrations", "metadata"):
            self.assertNotIn(k, m)
        # fallback-safe: if hybrid not active, method is deterministic-only
        if not m["hybrid_active"]:
            self.assertIn("deterministic", (m["identification_method"] or "").lower())

    def test_training_check_no_compound_annotations(self):
        ds = L.load_serum(self.root)
        chk = L.training_check(ds)
        self.assertFalse(chk["has_compound_annotations"])          # only phenotype labels
        self.assertIn("rheumatoid_arthritis", chk["phenotype_labels"])
        self.assertTrue(any("pSCNN" in s for s in chk["do_not_train"]))
        self.assertTrue(any("NMR_OFFLINE" in chk["governance"] for _ in [0]))


class Mtbls6213AdapterTests(unittest.TestCase):
    """Real public MTBLS6213 is ISA-Tab (s_MTBLS6213.txt) — NOT the competition
    spectra_intensity_ppm.csv. The adapter must extract the competition metadata schema
    from the real ISA-Tab layout (bracketed factors + ontology 'plumbing' columns +
    a bonus Treatment-response factor), tab-separated."""

    # exact real MTBLS6213 sample-sheet column layout (values are illustrative)
    ISATAB = (
        "Source Name\tCharacteristics[Organism]\tTerm Source REF\tTerm Accession Number\t"
        "Characteristics[Organism part]\tTerm Source REF\tTerm Accession Number\t"
        "Protocol REF\tSample Name\t"
        "Factor Value[Rheumatoid arthritis]\tUnit\tTerm Source REF\tTerm Accession Number\t"
        "Factor Value[Anti-TNF therapy]\tTerm Source REF\tTerm Accession Number\t"
        "Factor Value[Treatment response]\tTerm Source REF\tTerm Accession Number\n"
        "101\tHomo sapiens\tNCBITAXON\t9606\tSerum\tUBERON\t1\tprep\t101\tFALSE\t\t\t\tFALSE\t\t\tn/a\t\t\n"
        "201\tHomo sapiens\tNCBITAXON\t9606\tSerum\tUBERON\t1\tprep\t201\tTRUE\t\t\t\tTRUE\t\t\tTRUE\t\t\n"
        "202\tHomo sapiens\tNCBITAXON\t9606\tSerum\tUBERON\t1\tprep\t202\tTRUE\t\t\t\tFALSE\t\t\tFALSE\t\t\n"
    )

    def test_isatab_to_competition_metadata(self):
        from backend.nmr_api import mtbls_adapter as A
        meta = A.isatab_to_serum_metadata(self.ISATAB.encode())
        # exactly the competition columns, plumbing dropped
        self.assertEqual(list(meta.columns), [
            "Characteristics[Organism part]", "Sample Name",
            "Factor Value[Rheumatoid arthritis]", "Factor Value[Anti-TNF therapy]"])
        self.assertEqual(list(meta["Sample Name"]), ["101", "201", "202"])
        self.assertEqual(list(meta["Factor Value[Rheumatoid arthritis]"]), ["FALSE", "TRUE", "TRUE"])
        self.assertTrue((meta["Characteristics[Organism part]"] == "Serum").all())

    def test_adapted_metadata_loads_and_labels_via_loader(self):
        """The adapted ISA-Tab must flow through the SAME loader as competition data."""
        from backend.nmr_api import mtbls_adapter as A
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nmr-pattern"
            sd = root / "Human_Serum"
            sd.mkdir(parents=True)
            # spectra for 3 samples, Var order == metadata row order
            _write_spectra(sd / "spectra_intensity_ppm.csv", ["Var1", "Var2", "Var3"])
            A.isatab_to_serum_metadata(self.ISATAB.encode()).to_csv(
                sd / "MTBLS6213_Metadata.csv", index=False)
            ds = L.load_serum(root)
            ra = ds["tasks"]["rheumatoid_arthritis"]
            self.assertEqual(ra["label_map"], {"Var1": 0, "Var2": 1, "Var3": 1})  # FALSE=0, TRUE=1


@unittest.skipUnless(
    Path("/private/tmp/claude-501/-Applications-Vibing-coding-Noom-copy-cat/"
         "bc24c460-d0b7-49e3-b37a-5d6c2f1613c3/scratchpad/mtbls6213/zips/101.zip").exists(),
    "cached real MTBLS6213 Bruker zips not present")
class Mtbls6213RealSpectraTests(unittest.TestCase):
    """Real Bruker → ppm×Var reconstruction (runs only when the public zips are cached)."""

    def test_bruker_zip_reconstructs_ppm_spectrum(self):
        from backend.nmr_api import mtbls_adapter as A
        zp = ("/private/tmp/claude-501/-Applications-Vibing-coding-Noom-copy-cat/"
              "bc24c460-d0b7-49e3-b37a-5d6c2f1613c3/scratchpad/mtbls6213/zips/101.zip")
        ppm, inten = A.bruker_zip_to_spectrum(zp)
        self.assertGreater(len(ppm), 10000)                     # real high-res spectrum
        self.assertTrue(ppm[0] < ppm[-1])                       # ascending
        c, b = A.bin_to_grid(ppm, inten, 0.5, 9.5, 0.005)
        self.assertEqual(len(c), len(b))
        # residual water dominates serum in D2O → strong signal near 4.7 ppm
        wj = int(np.argmin(np.abs(c - 4.7)))
        self.assertGreater(b[wj], np.median(b))


class IngestionReportTests(unittest.TestCase):
    def test_report_runs_on_both_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "nmr-pattern"
            for arm, meta_name, meta in (
                ("Human_Serum", "MTBLS6213_Metadata.csv",
                 "Sample Name,Characteristics[Organism part],Factor Value[Rheumatoid arthritis]\n"
                 + "".join(f"{i},Serum,{'TRUE' if i % 2 else 'FALSE'}\n" for i in range(1, 7))),
                ("Human_Urine", "MTBLS1_MTBLS694_metadata.csv",
                 "Sample Name,Characteristics[Organism part],Factor Value,Factor Value\n"
                 + "".join(f"u{i},urine,{'Male' if i % 2 else 'Female'},"
                          f"{'diabetes mellitus' if i % 2 else 'Control Group'}\n" for i in range(1, 7))),
            ):
                d = root / arm
                d.mkdir(parents=True)
                _write_spectra(d / "spectra_intensity_ppm.csv", [f"Var{i}" for i in range(1, 7)])
                (d / meta_name).write_text(meta, encoding="utf-8")
            rep = L.ingestion_report(root)
            self.assertEqual(rep["serum"]["matrix"], "serum")
            self.assertEqual(rep["urine"]["matrix"], "urine")
            self.assertIn("rheumatoid_arthritis", rep["serum"]["tasks"])


if __name__ == "__main__":
    unittest.main()
