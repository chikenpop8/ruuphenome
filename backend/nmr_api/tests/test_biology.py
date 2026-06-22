from __future__ import annotations

import unittest

from backend.nmr_api import biology


class BiologyTests(unittest.TestCase):
    def test_name_normalization_handles_stereo_and_acid_forms(self):
        self.assertEqual(biology.normalize("L-Alanine"), "alanine")
        self.assertEqual(biology.normalize("L-Lactic acid"), "lactate")
        self.assertEqual(biology.normalize("(R)-3-Hydroxybutyric acid"), "3-hydroxybutyrate")
        self.assertEqual(biology.normalize("Pyruvic acid"), "pyruvate")

    def test_annotate_returns_curated_biology(self):
        bio = biology.annotate("Creatinine")
        self.assertIsNotNone(bio)
        self.assertIn("kidney", bio["disease_associations"].lower())
        self.assertTrue(bio["pathways"])
        self.assertEqual(bio["source"], "HMDB 5.0 (curated)")

    def test_annotate_unknown_returns_none(self):
        self.assertIsNone(biology.annotate("not-a-real-metabolite-xyz"))

    def test_pathway_enrichment_finds_glycolysis(self):
        # An energy-metabolism panel should enrich Glycolysis most strongly.
        panel = ["L-Lactic acid", "Pyruvic acid", "D-Glucose"]
        background = list(biology.METABOLITE_BIOLOGY.keys())
        result = biology.pathway_enrichment(panel, background)
        self.assertTrue(result)
        top = result[0]
        self.assertEqual(top["pathway"], "Glycolysis / Gluconeogenesis")
        self.assertLess(top["p_value"], 0.05)
        self.assertGreaterEqual(top["overlap"], 2)

    def test_pathway_enrichment_pvalues_are_sorted(self):
        panel = ["lactate", "pyruvate", "citrate", "alanine", "glucose"]
        result = biology.pathway_enrichment(panel, list(biology.METABOLITE_BIOLOGY.keys()))
        pvals = [r["p_value"] for r in result]
        self.assertEqual(pvals, sorted(pvals))

    def test_interpret_panel_combines_cards_and_enrichment(self):
        panel = ["L-Lactic acid", "Pyruvic acid", "D-Glucose", "L-Valine"]
        out = biology.interpret_panel(panel)
        self.assertEqual(out["coverage"]["total"], 4)
        self.assertEqual(out["coverage"]["annotated"], 4)
        self.assertIsNotNone(out["top_pathway"])
        self.assertTrue(out["metabolite_biology"])
        self.assertTrue(out["pathway_enrichment"])


if __name__ == "__main__":
    unittest.main()
