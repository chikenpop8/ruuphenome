from __future__ import annotations

import json
import unittest

import numpy as np

from backend.nmr_api import open_data, self_supervised


class SelfSupervisedTests(unittest.TestCase):
    def test_open_corpus_has_provenance(self):
        self.assertTrue(open_data.CORPUS_PATH.exists())
        self.assertTrue(open_data.PROVENANCE_PATH.exists())
        provenance = json.loads(open_data.PROVENANCE_PATH.read_text())
        data = np.load(open_data.CORPUS_PATH)

        self.assertEqual(len(provenance["entries"]), 12)
        self.assertEqual(data["spectra"].shape, (12, 4096))
        self.assertTrue(all(len(item["sha256"]) == 64 for item in provenance["entries"]))

    def test_trained_encoder_retrieves_reference(self):
        status = self_supervised.status()
        self.assertTrue(status["trained"])
        spectra, labels, ppm = self_supervised._load_corpus()
        matches = self_supervised.identify(ppm, spectra[0], top_k=3)

        self.assertEqual(matches[0]["metabolite"], labels[0])
        self.assertGreater(matches[0]["cosine_similarity"], 0.9)


if __name__ == "__main__":
    unittest.main()
