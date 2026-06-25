from __future__ import annotations

import os
import unittest

from backend.nmr_api import enrich


class OfflineModeTests(unittest.TestCase):
    def setUp(self):
        self._prev = os.environ.get("NMR_OFFLINE")

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("NMR_OFFLINE", None)
        else:
            os.environ["NMR_OFFLINE"] = self._prev

    def test_offline_flag_parses(self):
        for val in ("1", "true", "TRUE", "yes"):
            os.environ["NMR_OFFLINE"] = val
            self.assertTrue(enrich.offline_mode())
        for val in ("0", "", "no", "false"):
            os.environ["NMR_OFFLINE"] = val
            self.assertFalse(enrich.offline_mode())

    def test_offline_blocks_network_get(self):
        os.environ["NMR_OFFLINE"] = "1"
        # _get must short-circuit to None without any outbound call
        self.assertIsNone(enrich._get("https://pubchem.ncbi.nlm.nih.gov/rest/pug/x"))

    def test_offline_enrich_returns_fallback(self):
        os.environ["NMR_OFFLINE"] = "1"
        out = enrich.enrich("definitely-not-cached-compound-zzz", "", "")
        self.assertIsNone(out["pubchem_cid"])
        # name-based fallback links must still be provided
        self.assertTrue(out["external_refs"]["hmdb"])


if __name__ == "__main__":
    unittest.main()
