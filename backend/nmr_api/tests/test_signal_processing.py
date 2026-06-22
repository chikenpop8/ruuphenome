from __future__ import annotations

import io
import unittest
import zipfile
from pathlib import Path

import numpy as np

from backend.nmr_api import signal_processing
from backend.nmr_api.shifts_db import HMDB_KNOWN_SHIFTS


NAMES = {
    "CC(O)=O": "acetate",
    "C[C@H](N)C(O)=O": "L-alanine",
    "OC(=O)CC(O)(CC(O)=O)C(O)=O": "citrate",
    "CO": "methanol",
    "N[C@@H](Cc1ccc(O)cc1)C(O)=O": "L-tyrosine",
}
TEST_SHIFTS = {key: HMDB_KNOWN_SHIFTS[key] for key in NAMES}


class Domain1ProcessingTests(unittest.TestCase):
    def test_demo_recovers_known_metabolites_with_qc(self):
        result = signal_processing.demo_spectrum(TEST_SHIFTS, compound_names=NAMES)
        assigned = {item["metabolite"]: item for item in result["assignments"]}

        self.assertGreaterEqual(result["quality_control"]["score"], 75)
        self.assertLess(result["quality_control"]["peaks_total"], 150)
        self.assertTrue(set(NAMES.values()).issubset(assigned))
        # Single-resonance compounds are intentionally capped/tentative.
        self.assertTrue(all(assigned[name]["confidence"] >= 45 for name in NAMES.values()))
        self.assertTrue(
            all(assigned[name]["mean_ppm_error"] <= 0.01 for name in NAMES.values())
        )

    def test_baseline_correction_flattens_curved_background(self):
        rng = np.random.default_rng(7)
        x = np.linspace(-1, 1, 4000)
        baseline = 0.7 + 0.4 * x + 0.25 * x**2
        peaks = (
            4.0 * np.exp(-((x + 0.35) / 0.012) ** 2)
            + 2.5 * np.exp(-((x - 0.42) / 0.02) ** 2)
        )
        y = baseline + peaks + rng.normal(0, 0.01, len(x))
        corrected = signal_processing.baseline_correct(y)
        quiet = (np.abs(x + 0.35) > 0.08) & (np.abs(x - 0.42) > 0.1)

        self.assertLess(abs(float(np.median(corrected[quiet]))), 0.01)
        self.assertGreater(float(np.max(corrected)), 2.0)

    def test_robust_peak_picker_rejects_noise(self):
        rng = np.random.default_rng(11)
        ppm = np.linspace(10, 0, 20000)
        y = rng.normal(0, 0.002, len(ppm))
        truth = [1.48, 1.92, 3.36, 6.89]
        for shift in truth:
            y += np.exp(-((ppm - shift) / 0.004) ** 2)
        peaks = signal_processing.pick_peaks(ppm, y, snr=5)
        observed = [peak["ppm"] for peak in peaks]

        self.assertLessEqual(len(peaks), 8)
        for shift in truth:
            self.assertLess(min(abs(value - shift) for value in observed), 0.002)

    def test_processed_csv_parser(self):
        raw = b"ppm,intensity\n1.0,2.0\n2.0,3.0\n" + b"\n".join(
            f"{3 + i / 10},{i}".encode() for i in range(20)
        )
        ppm, intensity = signal_processing.parse_processed_spectrum(raw, "spectrum.csv")
        self.assertEqual(len(ppm), 22)
        self.assertEqual(len(intensity), 22)
        self.assertAlmostEqual(float(ppm[0]), 1.0)

    def test_internal_standard_referencing(self):
        ppm = np.linspace(1.0, -1.0, 8000)
        y = np.exp(-((ppm + 0.17) / 0.004) ** 2)
        referenced, info = signal_processing.reference_to_internal_standard(ppm, y)
        peak_ppm = referenced[int(np.argmax(y))]

        self.assertEqual(info["method"], "internal standard")
        self.assertAlmostEqual(peak_ppm, 0.0, places=4)
        self.assertAlmostEqual(info["offset_ppm"], 0.17, places=3)

    def test_real_bmrb_leucine_recovers_all_reference_resonances(self):
        archive = (
            Path(__file__).resolve().parents[1]
            / "open_data"
            / "bmrb_raw"
            / "bmse000042_1.zip"
        )
        if not archive.exists():
            self.skipTest("Curated BMRB raw corpus has not been downloaded.")

        fid, acquisition = signal_processing.read_bruker_zip(archive.read_bytes())
        result = signal_processing.process_fid(
            fid,
            acquisition["sw_hz"],
            acquisition["sf_mhz"],
            carrier_ppm=acquisition["carrier_ppm"],
            reference_shifts={
                "leucine": [0.96, 1.70, 1.72, 1.73, 3.73],
            },
            compound_names={"leucine": "L-leucine"},
            assignment_backend="pattern-matcher",
        )
        assignment = next(
            item
            for item in result["assignments"]
            if item["metabolite"] == "L-leucine"
        )

        self.assertEqual(
            result["quality_control"]["reference"]["method"],
            "internal standard",
        )
        self.assertEqual(assignment["matched_count"], 5)
        self.assertEqual(assignment["coverage"], 1.0)
        self.assertLess(assignment["mean_ppm_error"], 0.01)
        self.assertGreater(assignment["confidence"], 90)
        self.assertEqual(
            result["self_supervised_matches"][0]["metabolite"],
            "L-leucine",
        )

    def test_zip_path_traversal_is_rejected(self):
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr("../fid", b"unsafe")
        with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
            with self.assertRaisesRegex(ValueError, "unsafe path"):
                signal_processing._validate_zip(archive)


if __name__ == "__main__":
    unittest.main()
