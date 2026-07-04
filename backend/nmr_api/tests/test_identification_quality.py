"""Track-1 honesty layer: MSI identification levels + D2O exchangeable-proton
guard + authoritative organizer pins."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from backend.nmr_api import identification_quality as idq
from backend.nmr_api import spectral_cohort as sc


class MSIAndD2OTests(unittest.TestCase):
    def test_classify_shift(self):
        self.assertEqual(idq.classify_shift(4.80), "water_hdo")      # residual HDO
        self.assertEqual(idq.classify_shift(11.0), "exchangeable_risk")  # COOH/NH
        self.assertEqual(idq.classify_shift(1.33), "non_exchangeable")   # lactate CH3
        self.assertEqual(idq.classify_shift(5.23), "non_exchangeable")   # glucose anomeric

    def test_msi_never_level1_without_standard(self):
        self.assertEqual(idq.msi_level(0.8, 3)["msi_level"], 2)  # library match → L2
        self.assertEqual(idq.msi_level(0.3, 1)["msi_level"], 3)  # single resonance → L3
        self.assertEqual(idq.msi_level(0.0, 0)["msi_level"], 4)  # no non-exch evidence
        # only an authentic standard reaches Level 1
        self.assertEqual(idq.msi_level(0.8, 3, has_authentic_standard=True)["msi_level"], 1)

    def test_d2o_excludes_water_and_exchangeable(self):
        a = idq.d2o_assessment([1.33, 4.80, 11.0], expected_shifts=3)
        self.assertEqual(a["n_nonexchangeable"], 1)   # only 1.33 survives
        self.assertEqual(a["n_water_hdo"], 1)
        self.assertEqual(a["n_exchangeable_risk"], 1)
        self.assertTrue(a["usable_in_d2o"])
        self.assertEqual(a["robust_shifts"], [1.33])
        self.assertIsNotNone(a["d2o_caveat"])

    def test_d2o_rejects_water_only(self):
        a = idq.d2o_assessment([4.80], expected_shifts=1)
        self.assertFalse(a["usable_in_d2o"])
        self.assertEqual(a["n_nonexchangeable"], 0)


class AnnotateHonestyIntegrationTests(unittest.TestCase):
    def _flat_matrix(self, value=1.0):
        bins = np.round(np.arange(0.0, 10.0, 0.01), 2)
        X = pd.DataFrame(np.full((6, len(bins)), value),
                         columns=bins, index=[f"s{i}" for i in range(6)])
        return X, bins

    def test_annotate_emits_msi_and_d2o_fields(self):
        X, bins, _ = sc.make_demo_binned(n_per_group=8)
        res = sc.annotate(sc.pqn_normalize(X), bins)
        self.assertIn("identification_standard", res)
        self.assertIn("d2o_guard", res)
        for m in res["metabolites"]:
            self.assertIn(m["msi_level"], (1, 2, 3))       # never 4 among the "present"
            self.assertTrue(m["d2o"]["usable_in_d2o"])      # guard: non-exchangeable evidence
            self.assertEqual(m["provenance"], "reference_match")

    def test_water_only_metabolite_rejected(self):
        X, bins = self._flat_matrix()
        res = sc.annotate(X, bins, reference_shifts={"phantom_water": [4.80]})
        self.assertNotIn("phantom_water", {m["metabolite"] for m in res["metabolites"]})

    def test_authoritative_pin_bypasses_occupancy(self):
        # 1.92 ppm kept LOW (below the occupancy gate); a pin must still force it in
        X, bins = self._flat_matrix(value=10.0)
        j = int(np.argmin(np.abs(bins - 1.92)))
        X.iloc[:, j] = 0.1
        no_pin = sc.annotate(X, bins, reference_shifts={"acetate": [1.92]})
        with_pin = sc.annotate(X, bins, reference_shifts={"acetate": [1.92]},
                               identified_peaks={1.92: "acetate"})
        self.assertNotIn("acetate", {m["metabolite"] for m in no_pin["metabolites"]})
        pinned = [m for m in with_pin["metabolites"] if m["metabolite"] == "acetate"]
        self.assertTrue(pinned)
        self.assertEqual(pinned[0]["provenance"], "organizer_pin")
        self.assertEqual(with_pin["n_organizer_pins"], 1)


if __name__ == "__main__":
    unittest.main()
