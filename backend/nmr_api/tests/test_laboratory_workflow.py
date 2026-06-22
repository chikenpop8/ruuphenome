from __future__ import annotations

import unittest

from backend.nmr_api import laboratory_workflow


class LaboratoryWorkflowTests(unittest.TestCase):
    def test_workflow_has_ordered_unique_release_stages(self):
        stages = laboratory_workflow.workflow()["stages"]
        self.assertEqual([stage["order"] for stage in stages], list(range(1, 14)))
        self.assertEqual(len({stage["id"] for stage in stages}), len(stages))
        self.assertTrue(all(stage["release_gate"] for stage in stages))
        self.assertTrue(all(stage["required_records"] for stage in stages))

    def test_complete_qc_pass_releases_sample(self):
        result = laboratory_workflow.evaluate_qc(
            qc_score=92,
            max_snr=80,
            negative_area_fraction=0.04,
            reference_method="internal standard",
            pooled_qc_cv_percent=8,
            drift_percent=6,
            blank_contamination=False,
            instrument_suitability_passed=True,
            sample_identity_verified=True,
        )
        self.assertEqual(result["decision"], "pass")
        self.assertTrue(result["release_allowed"])

    def test_failed_identity_or_spectrum_blocks_release(self):
        result = laboratory_workflow.evaluate_qc(
            qc_score=60,
            max_snr=80,
            negative_area_fraction=0.04,
            reference_method="internal standard",
            pooled_qc_cv_percent=8,
            drift_percent=6,
            blank_contamination=False,
            instrument_suitability_passed=True,
            sample_identity_verified=False,
        )
        self.assertEqual(result["decision"], "fail")
        self.assertFalse(result["release_allowed"])

    def test_missing_batch_evidence_requires_review(self):
        result = laboratory_workflow.evaluate_qc(
            qc_score=92,
            max_snr=80,
            negative_area_fraction=0.04,
            reference_method="internal standard",
        )
        self.assertEqual(result["decision"], "needs_review")
        self.assertFalse(result["release_allowed"])


if __name__ == "__main__":
    unittest.main()
