from __future__ import annotations

import asyncio
import io
import unittest

from fastapi import UploadFile

from backend.nmr_api import main, signal_processing
from backend.nmr_api.shifts_db import HMDB_KNOWN_SHIFTS


class ProfileWorkflowEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        names = {
            "CC(O)=O": "acetate",
            "C[C@H](N)C(O)=O": "L-alanine",
            "OC(=O)CC(O)(CC(O)=O)C(O)=O": "citrate",
        }
        shifts = {key: HMDB_KNOWN_SHIFTS[key] for key in names}
        demo = signal_processing.demo_spectrum(shifts, compound_names=names)
        rows = ["ppm,intensity"]
        rows.extend(
            f"{ppm},{intensity}"
            for ppm, intensity in zip(demo["ppm"], demo["intensity"])
        )
        cls.raw = ("\n".join(rows) + "\n").encode()

    def _upload(self) -> UploadFile:
        return UploadFile(filename="demo.csv", file=io.BytesIO(self.raw))

    def _run_auto(self, *, bootstrap_iterations: int = 0):
        return asyncio.run(
            main.profile_auto(
                self._upload(),
                snr_threshold=5.0,
                tolerance_ppm=0.04,
                assignment_backend="pattern-matcher",
                fdr=0.05,
                hi=0.85,
                lo=0.5,
                bootstrap_iterations=bootstrap_iterations,
            )
        )

    def test_profile_qc_endpoint_passes_good_demo(self):
        qc = asyncio.run(main.profile_qc(self._upload(), snr_threshold=5.0))

        self.assertEqual(qc["verdict"], "pass")
        self.assertGreater(qc["snr"], 10)
        self.assertIn("baseline_score", qc)
        self.assertIn("water_residual", qc)

    def test_profile_auto_endpoint_returns_metabolite_contract(self):
        result = self._run_auto()

        self.assertGreaterEqual(len(result["metabolites"]), 1)
        for item in result["metabolites"]:
            self.assertGreaterEqual(item.confidence, 0)
            self.assertLessEqual(item.confidence, 1)
            self.assertIn(item.status, {"accept", "review", "reject"})
            self.assertGreaterEqual(item.fdr, 0)
            self.assertTrue(item.provenance.model_version)
            self.assertIsInstance(item.provenance.flags, list)

    def test_profile_triage_partitions_latest_auto_profile(self):
        auto = self._run_auto()
        triaged = main.profile_triage(hi=0.85, lo=0.5)

        total = sum(len(bucket) for bucket in triaged.values())
        names = [
            item.name
            for bucket in triaged.values()
            for item in bucket
        ]
        self.assertEqual(total, len(auto["metabolites"]))
        self.assertEqual(len(names), len(set(names)))

    def test_profile_auto_fills_bootstrap_uncertainty(self):
        result = self._run_auto(bootstrap_iterations=4)

        for item in result["metabolites"]:
            concentration = item.concentration
            self.assertIsNotNone(concentration.ci_low)
            self.assertIsNotNone(concentration.ci_high)
            self.assertLessEqual(concentration.ci_low, concentration.value)
            self.assertLessEqual(concentration.value, concentration.ci_high)

    def test_profile_report_and_csv_endpoints(self):
        self._run_auto(bootstrap_iterations=2)
        original_ncd_screen = main.ncd_screen
        main.ncd_screen = lambda repeats=1: {"panel": [{"ncd": "demo"}], "note": "test"}
        try:
            report = main.profile_report(signed_by="Reviewer")
            csv_response = main.profile_report_csv()
        finally:
            main.ncd_screen = original_ncd_screen

        self.assertIn("qc", report)
        self.assertIn("metabolites", report)
        self.assertIn("triage", report)
        self.assertIn("ncd_panel", report)
        self.assertEqual(report["signed_by"], "Reviewer")
        self.assertTrue(report["signed_at"])
        self.assertIn("name,concentration,unit", csv_response.body.decode())


if __name__ == "__main__":
    unittest.main()
