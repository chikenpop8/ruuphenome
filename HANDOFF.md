# RuuPhenome — Current Project Handoff

Last verified: **June 26, 2026**  
Active app root: `/Applications/Vibing coding/Noom copy cat/ruuphenome`

RuuPhenome is an open-source NMR metabolomics profiler for the Thailand
National Phenome Institute / BDI Hackathon 2026 Track Phenome work. The product
goal is an explainable alternative to closed NMR profiling tools such as
Chenomx: import NMR evidence, annotate metabolites, quantify/visualize them,
then connect the metabolite table to leakage-safe biomarker discovery and
biological interpretation.

The project has three connected layers:

- **Domain 1 single-spectrum profiling:** raw Bruker ZIP or processed 1D ¹H
  spectrum → preprocessing/QC/peak picking/metabolite assignment evidence.
- **Track 1 cohort pipeline:** preprocessed/binned NMR matrix → normalization →
  metabolite annotation → overlap-aware quantification → visualization/export.
- **Domain 2 biomarker discovery:** sample × metabolite/ppm matrix → PCA/UMAP,
  leakage-safe model comparison, stable biomarker panel and pathway biology.

## Repository and runtime

- Parent workspace: `/Applications/Vibing coding/Noom copy cat`
- Active app: `/Applications/Vibing coding/Noom copy cat/ruuphenome`
- Main package: `backend/nmr_api`
- Frontend: `backend/nmr_api/static/profiler.html`
- This active app **is a Git repository**. Check `git status --short` before
  editing; the workspace may be dirty.
- Reuse the existing virtual environment at `backend/nmr_api/.venv`.

From the active app root:

```bash
backend/nmr_api/.venv/bin/python -m uvicorn \
  backend.nmr_api.main:app --host 127.0.0.1 --port 8100
```

Or use the project launcher:

```bash
bash backend/nmr_api/run.sh
```

Then open:

- UI: http://127.0.0.1:8100/
- Swagger: http://127.0.0.1:8100/docs
- Health: http://127.0.0.1:8100/healthz

Important runtime nuance: `run.sh` exports
`NMRFORMER_ADAPTER_MODULE=nmr_api.nmrformer_adapter` and runs from `backend/`.
Direct `uvicorn backend.nmr_api.main:app` does **not** automatically set that
environment variable.

## Verified environment

| Package | Version / status |
|---|---:|
| Python environment | `backend/nmr_api/.venv` |
| FastAPI | 0.138.0 |
| Uvicorn | 0.49.0 |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| SciPy | 1.18.0 |
| scikit-learn | 1.9.0 |
| nmrglue | 0.11 |
| XGBoost | 3.3.0 |
| CatBoost | not installed |
| PyTorch | 2.12.1 |
| umap-learn | 0.5.12 |

`NMRTransformer` is **not installed**. Arbitrary-SMILES shift prediction uses
the HMDB known-shift fallback table unless `backend/nmr_api/setup_nmrtransformer.sh`
is run successfully.

`NMRformer` files and a local adapter are bundled in
`backend/nmr_api/NMRformer` and `backend/nmr_api/nmrformer_adapter.py`. The
adapter reports its model files as present, but production claims should still
say: **NMRformer support is optional and must be validated on the target sample
type before being used as a primary assignment claim.** Pattern matching remains
the explainable default.

## Important modules

| Module | Current responsibility |
|---|---|
| `main.py` | FastAPI app, endpoint wiring and demo dataset registry |
| `spectral_cohort.py` | Track 1 binned-spectra pipeline: load/orient matrix, normalize, annotate, deconvolve, export and derive labels |
| `signal_processing.py` | Safe Bruker import, digital-filter removal, FFT, phase/baseline correction, referencing, peak picking and spectrum QC |
| `pipeline.py` | Parse MetaboLights MAF/results tables and bridge Domain 2 compounds to Domain 1 peaks |
| `shifts_db.py` | NMRTransformer adapter and HMDB fallback shifts |
| `nmrformer_backend.py` | Optional NMRformer adapter contract and hybrid scoring |
| `nmrformer_adapter.py` | Local wrapper for bundled `zza1211/NMRformer` files |
| `open_data.py` | Public BMRB corpus/provenance handling |
| `build_bmrb_library.py` | Builder for expanded BMRB reference-shift library |
| `self_supervised.py` | Masked 1D convolutional autoencoder and BMRB reference embeddings |
| `biomarker_engine.py` | Leakage-safe p≫n feature screening, grouped CV, stability, Q², VIP and permutation checks |
| `model_suite.py` | Patient-grouped raw/PCA logistic, SVM, HistGradientBoosting and XGBoost comparison |
| `dimensionality.py` | PCA scores/loadings and exploratory UMAP coordinates |
| `biomarkers.py` | Older/simple MetaboLights biomarker workflow and sample/patient parsing |
| `biology.py` | Curated metabolite biology cards and pathway enrichment |
| `laboratory_workflow.py` | Thirteen-stage real-laboratory workflow and conservative QC release evaluator |
| `library.py`, `enrich.py` | Compound cards, cached PubChem enrichment and UI data |
| `validate.py` | Validation CLI against reference peak/ID/quantification files |
| `static/profiler.html` | Single-file Chenomx-style UI |

## Current data and trained assets

### Bundled demo datasets

`main.py` defaults to the bundled MTBLS242 table:

```text
backend/nmr_api/open_data/demo_mtbls242.tsv
```

Override with `NMR_DEFAULT_TSV` if needed.

The app registry currently exposes:

- `mtbls242` — gastric-bypass longitudinal time-point task, serum NMR.
- `mtbls1` — type-2 diabetes vs control, urine NMR.
- `mtbls424` — breast-cancer relapse vs no-relapse, serum NMR.

The older user-supplied files in `/Users/bigray/Downloads/` are still useful
test material, but the app no longer depends on them as the default runtime
source.

### Public BMRB / model assets

- BMRB pure-compound corpus: `backend/nmr_api/open_data/bmrb_1h_corpus.npz`
- Open-data provenance: `backend/nmr_api/open_data/provenance.json`
- Expanded reference-shift library: `backend/nmr_api/open_data/bmrb_reference_shifts.json`
- Masked-spectrum encoder: `backend/nmr_api/models/masked_nmr_encoder.pt`
- NMRformer folder: `backend/nmr_api/NMRformer`

The self-supervised model is trained and available, but its retrieval benchmark
uses augmented versions of the same small pure-reference collection. Do not
present it as independent serum-mixture identification accuracy.

## What is actually implemented

### Domain 1 single-spectrum profiling

- Safe zipped Bruker FID ingestion with path-traversal checks.
- Bruker digital-filter removal and direct-dimension orientation.
- DC correction, apodization, zero filling, FFT, automatic phase correction and
  asymmetric least-squares baseline correction.
- Internal DSS/TSP/TMS referencing when confidently detected.
- MAD-based noise estimation and prominence/width-aware peak picking.
- Peak SNR, FWHM, area and artifact flags.
- Complete-pattern metabolite matching with coverage, ppm error, ambiguity and
  confidence.
- Optional NMRformer hybrid support through a validated adapter.
- Self-supervised nearest-reference evidence as supporting evidence only.
- Per-spectrum QC and real-laboratory release-rule evaluation.
- Processing from either Bruker ZIP or two-column processed CSV/TSV.

### Track 1 binned-spectra cohort pipeline

This is the biggest update versus the older handoff. A working binned cohort
pipeline now exists.

Implemented flow:

```text
preprocessed/binned matrix
→ orientation detection
→ PQN / total-area / no normalization
→ reference-shift annotation
→ optional organizer identified-peak pins
→ sample × metabolite table
→ NNLS linear-combination deconvolution
→ target-decoy/FDR-style filtering
→ overlay visualization
→ concentration CSV export
→ optional biomarker discovery and biology if labels are available
```

Key routes:

- `GET /spectral/demo-pipeline`
- `POST /spectral/annotate`
- `POST /spectral/pipeline`
- `POST /spectral/pipeline-file`
- `GET /spectral/demo-concentrations.csv`
- `POST /spectral/export-concentrations`

The one-file route accepts a binned CSV/TSV with an inline label column such as
`Class`, `Group`, `Condition`, `Diagnosis`, `Phenotype` or `Status`. It works
with string labels like `control/case`. There is a known bug with numeric
`0/1` labels; see “Known gaps” below.

### Domain 2 biomarkers and dimensionality

- MetaboLights result parsing into sample × metabolite matrices.
- Dataset switcher for `mtbls242`, `mtbls1`, `mtbls424`.
- Metadata-driven task creation:
  - `POST /track2/metadata-columns`
  - `POST /track2/discover-with-metadata`
- Leakage-safe variance/FDR/Top-k feature selection inside CV folds.
- Repeated stratified patient-grouped nested CV.
- Elastic-net logistic, linear SVM, PCA logistic, PCA linear SVM,
  HistGradientBoosting and XGBoost challengers.
- Median imputation, scaling and PCA fitted separately inside training folds.
- ROC-AUC, F1, Brier score, calibration error and stable-panel reporting.
- Q², VIP scores and permutation p-values in the safe biomarker engine.
- Exploratory full-cohort PCA scores, explained variance and loadings.
- Exploratory UMAP fitted to the PCA representation with fixed seed.
- Biological interpretation via curated metabolite cards and pathway enrichment.

### UI

The Tools menu exposes the major workflows:

- Track 1 → Track 2 automated pipeline for binned spectra.
- Last imported pipeline result viewer.
- Real laboratory workflow.
- Current-spectrum laboratory QC evaluation.
- Domain 2 biomarker/model comparison.
- Domain 2 biological interpretation.
- Domain 2 PCA/UMAP visualization.
- Dataset switcher.
- Bruker FID processing and processed CSV/TSV spectrum analysis.
- Domain 1 QC/assignments, open-data status, plugin/backend status.

Recent UI work includes:

- Imported Track 1 data populates the main spectrum page, not only a popup.
- Dynamic zoom/pan, pinch/trackpad zoom and drag-to-pan navigation.
- Pinned multi-compound overlays.
- Per-row pin controls and per-compound overlay colors.
- Custom drag color picker with RGB/HEX entry, Enter-to-apply, Copy button and
  optional browser `EyeDropper` support.

## Verified scan results from June 26, 2026

These are engineering smoke-test results, not final scientific validation.

### Local checks

```bash
backend/nmr_api/.venv/bin/python -m unittest discover -s backend/nmr_api/tests -q
```

Result: **38 tests passing** in about 12 seconds.

Additional checks:

- Python modules compile successfully with `py_compile`.
- `bash -n backend/nmr_api/run.sh` passes.
- Inline frontend JavaScript parses with the macOS JavaScript engine.
- Node is not installed on this machine, so `node --check` is unavailable.

### API smoke tests

A local server was started on `127.0.0.1:8102` and these routes responded:

- `GET /healthz`
- `GET /plugins`
- `GET /datasets`
- `GET /open-data`
- `GET /self-supervised/status`
- `GET /spectral/demo-pipeline`
- `POST /spectral/pipeline-file`
- `GET /biology`
- `GET /biomarkers-model-suite?dataset=mtbls1&repeats=1`
- `GET /biomarkers-projection?dataset=mtbls1&include_umap=false`
- `GET /`
- `GET /openapi.json`

### Track 1 demo numbers

`GET /spectral/demo-pipeline`:

- 60 samples
- 900 ppm bins
- PQN normalization
- 268 annotated metabolites
- 578-entry reference-shift library
- synthetic task: case/control with planted BCAA signal
- biomarker AUC: `0.9263`

`POST /spectral/pipeline-file` using a generated demo binned file with string
labels:

- label column detected: `Class`
- classes: `control`, `case`
- mean deconvolution fit R²: `0.9331`
- FDR level: `0.05`
- quantified metabolites: `126`
- passing FDR: `35`
- biomarker AUC: `0.9263`

The same upload with numeric `Class = 0/1` did **not** detect labels. This is a
known implementation bug, not a failure of the whole pipeline.

### Domain 2 model-suite smoke test

`GET /biomarkers-model-suite?dataset=mtbls1&repeats=1`:

| Model | ROC-AUC |
|---|---:|
| PCA logistic | 0.9869 |
| PCA linear SVM | 0.9712 |
| Linear SVM | 0.9236 |
| Elastic-net logistic | 0.9142 |
| HistGradientBoosting | 0.9065 |
| XGBoost | 0.8777 |

These are smoke-test values on a bundled dataset with `repeats=1`, not final
benchmark claims.

### Biology smoke test

`GET /biology?metabolites=lactate,alanine,glucose,pyruvate,citrate,leucine`
returns curated metabolite cards and pathway enrichment. Top pathway in the
smoke test was Glycolysis / Gluconeogenesis with glucose, lactate and pyruvate.

### Self-supervised status

`GET /self-supervised/status` reports:

- available: true
- trained: true
- device: `mps`
- embedding dimension: 64
- final loss: about `0.0127`
- augmented retrieval top-1 accuracy: `0.975`
- top-5 accuracy: `1.0`

Again, this is an internal augmented retrieval check, not independent mixture
validation.

## API endpoints

### Core and UI

- `GET /`
- `GET /healthz`
- `GET /plugins`
- `GET /datasets`
- `GET /domain1-peaks`
- `POST /analyze`
- `GET /compounds`
- `POST /compounds-upload`

### Domain 1 single-spectrum / open data

- `GET /demo-spectrum`
- `POST /process-fid`
- `POST /process-spectrum`
- `GET /open-data`
- `POST /open-data/build`
- `GET /self-supervised/status`
- `POST /self-supervised/train`

### Track 1 binned spectral cohort

- `GET /spectral/demo-pipeline`
- `POST /spectral/annotate`
- `POST /spectral/pipeline`
- `POST /spectral/pipeline-file`
- `GET /spectral/demo-concentrations.csv`
- `POST /spectral/export-concentrations`

### Track 2 / Domain 2 biomarkers

- `GET /biomarkers`
- `POST /biomarkers-upload`
- `GET /biomarkers-safe`
- `GET /biomarkers-model-suite`
- `POST /biomarkers-model-suite-upload`
- `GET /biomarkers-projection`
- `POST /biomarkers-projection-upload`
- `GET /biomarkers-benchmark`
- `POST /track2/metadata-columns`
- `POST /track2/discover-with-metadata`
- `GET /biology`
- `GET /enrich-names`

### Laboratory workflow

- `GET /laboratory-workflow`
- `POST /laboratory-workflow/evaluate-qc`

## Known gaps and bugs — do not overclaim

1. **Inline numeric labels are missed.**  
   `spectral_cohort.extract_embedded_labels()` currently skips numeric columns,
   so `Class = 0/1` is ignored even when the column name is clearly a label.
   String labels such as `control/case` work. Fix by allowing numeric label
   columns when the column name matches a label synonym or when cardinality is
   small and the values are not ppm bins.

2. **Annotation is currently over-permissive.**  
   The demo annotates 268 metabolites from a 578-entry reference library. That
   is useful for showing coverage, but real lab claims need stricter scoring,
   duplicate/synonym collapsing, ambiguity handling, mixture validation and
   comparison against manual/Chenomx-reviewed ground truth.

3. **The strongest cohort pipeline assumes preprocessed/binned data.**  
   It matches the workshop request for binned NMR peak/spectral files. It is
   not yet a full raw multi-sample FID → alignment → binning production
   pipeline. Single-spectrum raw processing exists, but batch raw cohort
   alignment/binning is still a separate future feature.

4. **Concentration export is not yet laboratory-validated.**  
   NNLS deconvolution and CSV export work, and the UI labels the table in µM
   when internal-standard calibration is present. Treat these as Chenomx-style
   estimates until validated with standards, internal calibration and manual
   review on the target instrument/matrix.

5. **NMRformer is bundled but not a free pass.**  
   The files and adapter exist. Direct startup may not activate it. Even when
   active, use it as supporting evidence until target-matrix validation proves
   it improves assignments safely.

6. **PCA/UMAP are exploratory.**  
   Full-cohort PCA/UMAP separation is not classifier performance. Predictive
   metrics must keep imputation, scaling, feature selection and PCA inside
   training folds.

7. **Clinical/disease claims are limited by labels.**  
   MTBLS242 is longitudinal surgery time points, not disease versus control.
   MTBLS1 and MTBLS424 provide labeled demos, but final clinical claims need
   independent validation.

8. **The laboratory workflow is RUO design only.**  
   There is no full LIMS, authentication, electronic signature or durable
   append-only audit store.

9. **Frontend is a large single-file app.**  
   It works for a hackathon demo, but long-term maintainability would benefit
   from modularization.

## Recommended next work

1. Fix numeric inline label detection in `spectral_cohort.extract_embedded_labels`.
2. Tighten Track 1 annotation: synonym collapsing, minimum unique resonances,
   ambiguity scoring, matrix-specific exclusions and external validation.
3. Test `/spectral/pipeline` and `/spectral/pipeline-file` on the actual
   organizer binned files and metadata, then record the expected file format.
4. Validate quantification against internal standards/manual Chenomx-style
   review before making strong concentration claims.
5. Add a true raw cohort path if needed: batch import → common ppm grid →
   alignment → water/artifact masking → adaptive/fixed binning → matrix export.
6. Clarify NMRformer startup in docs/tests and add a small integration smoke
   test when the adapter is active.
7. Update this handoff after any major code or validation change.

## Instructions for the next AI agent

- Read this file before making completeness claims.
- Prefer the active app root: `/Applications/Vibing coding/Noom copy cat/ruuphenome`.
- Check `git status --short` before editing; preserve unrelated dirty files.
- Use `apply_patch` for file edits.
- Use the existing venv; do not recreate the environment unless explicitly
  asked.
- After code changes, run:

```bash
backend/nmr_api/.venv/bin/python -m unittest discover -s backend/nmr_api/tests -q
backend/nmr_api/.venv/bin/python -m py_compile $(find backend/nmr_api -name '*.py' -not -path '*/.venv/*')
bash -n backend/nmr_api/run.sh
```

- For frontend changes, parse the inline JS with macOS JavaScript engine if Node
  is unavailable.
- Keep synthetic, pure-reference, longitudinal and clinical validation clearly
  separated.
- Do not report exploratory PCA/UMAP as predictive accuracy.
- Do not claim NMRformer/NMRTransformer as active unless the runtime status
  actually confirms it for the startup mode being used.
