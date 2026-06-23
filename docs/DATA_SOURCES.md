# RuuPhenome — Integrated Data Sources

All sources below are **free, public, and run with no training**. "Load → run"
means: point the app at it and biomarkers + biology produce results immediately.
GPU/H100 is only ever used for the *optional* SSL-encoder upgrade (see
`H100_TRAINING.md`) — none of these datasets require it.

## Bundled & ready to run (in the repo)

| ID | Source | Task | Samples × metabolites | Demo result | Needs training |
|----|--------|------|------------------------|-------------|----------------|
| `mtbls242` | MetaboLights MTBLS242 (¹H NMR, serum) | gastric-bypass time-point 0 vs 4 | 465 × 21 | AUC ≈ 0.97 | ❌ |
| `mtbls1` | MetaboLights MTBLS1 (¹H NMR, urine) | type-2 diabetes vs control | 132 × 220 | AUC ≈ 0.92, BCAA pathway ✔ | ❌ |
| `mtbls424` | MetaboLights MTBLS424 (¹H NMR, serum) | breast-cancer relapse vs no-relapse | 590 labeled × 22 | AUC ≈ 0.57 (honest — relapse is hard) | ❌ |

Switch between them in the UI (**Tools → Select Domain 2 dataset**) or via the
`?dataset=` query param on `/biomarkers-safe`, `/biomarkers-model-suite`,
`/biomarkers-projection`. The `GET /datasets` endpoint lists them.

> **On MTBLS424's 0.57 AUC:** this is a *feature, not a bug*. Predicting cancer
> relapse from 22 serum metabolites is genuinely hard; the leakage-safe engine
> reports honest, modest performance instead of an inflated number. It shows the
> tool generalizes to a second disease and does not fabricate signal.

## Reference knowledge (lookup, never trained)

| Source | Role in RuuPhenome |
|--------|--------------------|
| **HMDB 5.0** | ¹H reference shifts (annotation fingerprints) + curated biology (49 metabolites, 29 pathways) |
| **BMRB** | 12 pure-compound ¹H reference spectra (SSL encoder references) |
| **PubChem / ChEBI / KEGG** | Reference-card enrichment (IUPAC name, CAS, CID, InChIKey, resolving links), disk-cached |

## Drop-in (same format, load with no code change)

| Source | How |
|--------|-----|
| **Any other MetaboLights NMR study** | Download its `m_*.tsv` (MAF) + `s_*.txt` (sample sheet) → upload via `/track2/discover-with-metadata`, or bundle it: drop the TSV + a labels JSON in `open_data/` and add one entry to `DATASETS` in `main.py` |
| **Metabolomics Workbench studies** | Export as a metabolite × sample table (CSV/TSV) → the robust loader (`read_results_table`) reads it directly; supply metadata separately |
| **Organizer-provided identified peaks (Track 1)** | Upload alongside the binned matrix to `/spectral/annotate` or `/spectral/pipeline` |

## How to add a new bundled dataset (≈5 min, no training)

1. Put the metabolite × sample table at `open_data/demo_<id>.tsv`.
2. Put a sample→label map at `open_data/demo_<id>_labels.json` (`{"sample": 0|1}`).
3. Add one entry to the `DATASETS` registry in `main.py`:
   ```python
   "<id>": {
       "label": "…", "kind": "labeled",
       "tsv": _OPEN / "demo_<id>.tsv",
       "labels": _OPEN / "demo_<id>_labels.json",
       "task": "X vs Y", "class_names": {0: "X", 1: "Y"},
       "source": "MetaboLights <ID> (¹H NMR)",
   }
   ```
4. Done — the UI switcher, all endpoints, and `/datasets` pick it up automatically.

_All MetaboLights data is publicly available under EMBL-EBI's open terms; BMRB
metabolomics standards are free to use._
