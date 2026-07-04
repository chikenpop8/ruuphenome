"""Track-1 provenance & condition-handling tests.

A binned matrix carries only sample IDs, ppm bins, and intensities — pH/solvent/
temperature/prep come from elsewhere and change ¹H chemical shifts. These tests pin:
  * matrix auto-profile + QC warnings (coarse bins, negative/transformed, missing),
  * solvent normalization and the condition-aware D₂O guard decision,
  * never-fail provenance assembly with condition-aware warnings,
  * the D₂O guard actually gating annotate/d2o_assessment.
"""
from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backend.nmr_api import provenance as P
from backend.nmr_api import spectral_cohort as sc
from backend.nmr_api import identification_quality as idq


def _grid(lo, hi, step):
    return np.round(np.arange(lo, hi, step), 5)


class SolventGateTests(unittest.TestCase):
    def test_normalization_and_guard(self):
        # aqueous/D2O → guard ON
        for s in ("D2O", "d2o", "aqueous", "aqueous D2O", "H2O+D2O", "phosphate buffer"):
            self.assertTrue(P.applies_d2o_guard(s), s)
        # named organic solvents → guard OFF (exchangeable protons stay visible)
        for s in ("DMSO-d6", "CDCl3", "CD3OD", "CD3CN", "acetone-d6", "benzene-d6", "pyridine-d5"):
            self.assertFalse(P.applies_d2o_guard(s), s)
        # missing/unknown → default to aqueous rule (assumed), never crash
        for s in ("", None, "unknown", "something weird"):
            self.assertTrue(P.applies_d2o_guard(s), s)
        self.assertEqual(P.normalize_solvent("methanol-d4"), "cd3od")


class ProvenanceBuildTests(unittest.TestCase):
    def test_never_fails_on_missing_and_warns(self):
        prov = P.build_provenance({"warnings": []}, {})   # nothing provided
        c = prov["conditions"]
        self.assertEqual(c["temperature_c"], "not provided")
        self.assertIsNone(c["ph"])
        self.assertEqual(c["internal_standard"], "unknown")   # advanced/optional default
        self.assertTrue(c["apply_d2o_guard"])                 # unknown → assumed aqueous
        joined = " ".join(prov["warnings"]).lower()
        self.assertIn("ph not provided", joined)
        self.assertIn("solvent not specified", joined)
        self.assertIn("temperature not provided", joined)
        # honesty + capabilities must state the limits
        self.assertFalse(prov["capabilities"]["ph_solvent_temperature_curated_library"])
        self.assertFalse(prov["capabilities"]["chenomx_style_condition_aware_fitting"])

    def test_organic_solvent_disables_guard_and_warns(self):
        prov = P.build_provenance({"warnings": []}, {"solvent": "DMSO-d6", "ph": "7.4",
                                                     "temperature": "25", "field_mhz": "600 MHz"})
        c = prov["conditions"]
        self.assertFalse(c["apply_d2o_guard"])
        self.assertEqual(c["ph"], 7.4)
        self.assertEqual(c["field_mhz"], 600.0)
        self.assertTrue(any("not applied" in w.lower() for w in prov["warnings"]))

    def test_condition_file_parse(self):
        raw = b"solvent: CDCl3\npH = 3.0\ntemperature: 37\nreference compound: TMS\n"
        fields = P.parse_condition_file(raw)
        self.assertEqual(P.normalize_solvent(fields["solvent"]), "cdcl3")
        self.assertEqual(fields["internal_standard"].strip(), "TMS")   # reference → advanced field


class MatrixProfileTests(unittest.TestCase):
    def test_warns_on_coarse_negative_missing(self):
        bins = _grid(0.5, 9.5, 0.05)                       # 0.05 ppm → coarse
        X = pd.DataFrame(np.ones((5, len(bins))), columns=bins)
        X.iloc[0, 0] = -2.0                                # negative/transformed
        X.iloc[1, 1] = np.nan                              # missing
        prof = sc.profile_matrix(X, bins)
        self.assertGreater(prof["bin_width_est"], 0.01)
        self.assertTrue(prof["coarse_bins"])
        self.assertTrue(prof["has_negative_intensity"])
        self.assertEqual(prof["n_missing"], 1)
        text = " ".join(prof["warnings"]).lower()
        self.assertIn("coarse", text)
        self.assertIn("negative", text)
        self.assertIn("missing", text)

    def test_fine_matrix_no_coarse_warning(self):
        bins = _grid(0.5, 9.5, 0.001)                      # fine
        X = pd.DataFrame(np.ones((4, len(bins))), columns=bins)
        prof = sc.profile_matrix(X, bins)
        self.assertFalse(prof["coarse_bins"])
        self.assertFalse(prof["has_negative_intensity"])
        self.assertEqual(prof["n_missing"], 0)

    def test_loader_returns_profile(self):
        bins = _grid(0.5, 9.5, 0.01)
        df = pd.DataFrame(np.ones((3, len(bins))), columns=[f"{b:.3f}" for b in bins],
                          index=["s0", "s1", "s2"])
        raw = df.to_csv().encode()
        X, bin_ppm, profile = sc.load_binned_matrix_profiled(raw)
        self.assertEqual(profile["n_samples"], 3)
        self.assertEqual(profile["n_features"], len(bins))


class D2OGuardGatingTests(unittest.TestCase):
    def _spiked(self):
        bins = _grid(0.0, 10.0, 0.01)
        X = pd.DataFrame(np.zeros((4, len(bins))), columns=[f"{b:.2f}" for b in bins])
        for p in (3.24, 3.40, 3.47, 3.71, 3.83, 4.63, 3.03, 4.05):
            X.iloc[:, int(round(p / 0.01))] = 8.0
        return X, bins

    def test_annotate_records_guard_flag(self):
        X, bins = self._spiked()
        aq = sc.annotate(X, bins, min_coverage=0.05, apply_d2o_guard=True)
        org = sc.annotate(X, bins, min_coverage=0.05, apply_d2o_guard=False)
        self.assertTrue(aq["apply_d2o_guard"])
        self.assertFalse(org["apply_d2o_guard"])
        # organic-solvent path must not apply the D₂O disappearance downgrade
        gm = next(m for m in org["metabolites"] if m["metabolite"] == "glucose")
        self.assertEqual(gm["d2o"]["d2o_grade"], "not_applicable")
        self.assertTrue(gm["d2o"]["usable_in_d2o"])

    def test_d2o_assessment_guard_off_keeps_all_shifts(self):
        # a shift in the water/HDO window would normally be excluded under the guard
        off = idq.d2o_assessment([3.4, 4.80], 3, apply_guard=False)
        on = idq.d2o_assessment([3.4, 4.80], 3, apply_guard=True)
        self.assertEqual(off["n_water_hdo"], 0)            # not flagged when guard off
        self.assertEqual(off["d2o_grade"], "not_applicable")
        self.assertGreaterEqual(off["n_nonexchangeable"], on["n_nonexchangeable"])


if __name__ == "__main__":
    unittest.main()
