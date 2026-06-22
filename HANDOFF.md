# RuuPhenome — Current Project Handoff

Last verified: **June 22, 2026**

RuuPhenome is an open-source NMR metabolomics profiler for the Thailand
National Phenome Institute / BDI Hackathon 2026 Track 1. The aim is to provide
an explainable alternative to closed tools such as Chenomx.

The project has two related domains:

- **Domain 1:** raw or processed 1D ¹H NMR spectrum → preprocessing, QC, peak
  detection and metabolite-identification evidence.
- **Domain 2:** sample × metabolite/ppm matrix → exploratory visualization,
  leakage-safe model comparison and a stable biomarker panel.

## Repository and runtime

- Workspace: `/Applications/Vibing coding/Noom copy cat`
- Main package: `backend/nmr_api`
- Frontend: `backend/nmr_api/static/profiler.html`
- Additional older prototype: `nmr_pipeline`
- This workspace is **not a Git repository**.
- Reuse the existing environment at `backend/nmr_api/.venv`; do not recreate it.

Start from the workspace root:

```bash
backend/nmr_api/.venv/bin/python -m uvicorn \
  backend.nmr_api.main:app --host 127.0.0.1 --port 8100
```

Alternatively:

```bash
bash backend/nmr_api/run.sh
```

Then open:

- UI: http://127.0.0.1:8100/
- Swagger: http://127.0.0.1:8100/docs
- Health: http://127.0.0.1:8100/healthz

The server is not guaranteed to already be running when a new agent starts.

## Verified environment

| Package | Version |
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
| PyTorch | 2.12.1 |
| umap-learn | 0.5.12 |

`NMRTransformer` is **not installed**. Arbitrary-SMILES shift prediction
therefore uses the small HMDB fallback table. The setup script is
`backend/nmr_api/setup_nmrtransformer.sh`.

`NMRformer` is also **not active** unless a validated adapter is supplied via
`NMRFORMER_ADAPTER_MODULE`. Hybrid assignment safely falls back to transparent
pattern matching.

## Important modules

| Module | Current responsibility |
|---|---|
| `main.py` | FastAPI application and endpoint wiring |
| `signal_processing.py` | Safe Bruker import, digital-filter removal, DC/apodization/FFT, phase and baseline correction, referencing, peak picking, assignments and spectrum QC |
| `pipeline.py` | Parse MetaboLights MAF results and match Domain 2 compounds against Domain 1 peaks |
| `shifts_db.py` | NMRTransformer adapter and HMDB known-shift fallback |
| `nmrformer_backend.py` | Optional validated NMRformer adapter contract and hybrid scoring |
| `open_data.py` | Download, checksum and process the curated public BMRB corpus |
| `self_supervised.py` | Masked 1D convolutional autoencoder and BMRB reference embeddings |
| `biomarker_engine.py` | Leakage-safe p≫n feature screening, grouped CV and stability |
| `model_suite.py` | Patient-grouped raw/PCA logistic, SVM, HistGradientBoosting and XGBoost comparison |
| `dimensionality.py` | PCA scores/loadings and exploratory UMAP coordinates |
| `biomarkers.py` | Older/simple n>p biomarker workflow and sample/patient parsing |
| `laboratory_workflow.py` | Thirteen-stage real-laboratory workflow and conservative QC release evaluator |
| `library.py`, `enrich.py` | Compound cards, cached PubChem enrichment and UI data |
| `models.py` | Pydantic request/response models |
| `validate.py` | Validation CLI against reference peak/ID/quantification files |
| `static/profiler.html` | Single-file Chenomx-style UI |

## Current data and trained assets

### Default Domain 2 table

`main.py` defaults to:

```text
/Users/bigray/Downloads/Domain_2_NMR_results_MTBLS242 (2).tsv
```

Override it with `NMR_DEFAULT_TSV`.

The file currently resolves and contains:

- 465 sample columns
- 21 identified metabolites
- longitudinal group counts: 0=106, 1=98, 2=98, 3=92, 4=71
- default binary task: time point 0 versus time point 4
- selected binary task size: 177 samples

The matching sample table is:

```text
/Users/bigray/Downloads/Domain_2_sample_table_MTBLS242 (2).tsv
```

The supplied Domain 1 reference is a PDF image/plot, not a numeric cohort:

```text
/Users/bigray/Downloads/Domain_1_processed_NMR_spectrum (2).pdf
```

### Public BMRB corpus

- 12 raw Bruker 1D ¹H pure-compound experiments
- fixed 4096-point 10→0 ppm representation
- corpus: `backend/nmr_api/open_data/bmrb_1h_corpus.npz`
- provenance/checksums: `backend/nmr_api/open_data/provenance.json`
- raw/open-data directory size: approximately 6.6 MB
- trained encoder: `backend/nmr_api/models/masked_nmr_encoder.pt`
- checkpoint size: approximately 480 KB

Compounds: alanine, glutamine, histidine, leucine, phenylalanine, tyrosine,
valine, citrate, glycine, creatinine, lactate and glucose.

## What is actually implemented

### Domain 1

- Safe zipped Bruker FID ingestion with path-traversal checks.
- Bruker digital-filter removal and correct direct-dimension orientation.
- DC correction, exponential apodization, zero fill and FFT.
- Zero/first-order automatic phase correction.
- Asymmetric least-squares baseline correction.
- Internal DSS/TSP/TMS referencing when confidently detected.
- MAD-based noise estimation and prominence/width-aware peak picking.
- Peak SNR, FWHM, area and artifact flags.
- Complete-pattern metabolite matching with coverage, ppm error, ambiguity and
  confidence.
- Optional NMRformer adapter; not falsely reported as active.
- Self-supervised BMRB nearest-reference evidence; it does not override the
  chemically explainable assignment.
- Per-spectrum QC and real-laboratory release-rule evaluation.
- Processing from either a Bruker ZIP or two-column processed CSV/TSV.

### Domain 2

- MetaboLights MAF parsing into sample × metabolite matrices.
- Patient extraction from longitudinal sample IDs.
- Leakage-safe variance/FDR/Top-k feature selection inside CV folds.
- Repeated stratified patient-grouped nested CV.
- Elastic-net logistic, linear SVM, PCA logistic, PCA linear SVM,
  HistGradientBoosting and XGBoost challengers.
- Median imputation, scaling and PCA fitted separately inside every training
  fold for PCA models.
- ROC-AUC, F1, Brier score, calibration error and stable-panel reporting.
- Exploratory full-cohort PCA scores, explained variance and loadings.
- Exploratory UMAP fitted to the PCA representation with a fixed seed.
- Upload endpoints for model comparison and PCA/UMAP.

### UI

The Tools menu currently exposes:

- Real laboratory workflow
- Current-spectrum laboratory QC evaluation
- Domain 2 biomarker/model comparison
- Domain 2 PCA/UMAP visualization
- Bruker FID processing
- Processed CSV/TSV spectrum analysis
- Domain 1 QC and assignments
- Plugin/backend status

## Verified results

Treat these as engineering results with the caveats immediately below.

### Domain 1

- Synthetic stress benchmark, 30 spectra:
  - original peak F1: 0.22
  - upgraded peak F1: 0.74
  - true-resonance recall: 0.96
  - average peak calls: 422 → 28
- Real public BMRB leucine raw FID:
  - internal standard detected and referenced
  - 5/5 expected resonances recovered
  - mean ppm error: 0.00404
  - assignment confidence: 95.7
  - self-supervised reference rank: leucine first
- Self-supervised masked reconstruction:
  - loss: 0.0242 → 0.0127
  - augmented reference retrieval: top-1 0.975, top-5 1.000

The SSL retrieval benchmark uses augmented versions of the same pure-reference
collection. It is not independent serum-mixture identification accuracy.

### Domain 2

MTBLS242 time point 0 versus time point 4, two repeated patient-grouped outer
CV runs:

| Model | ROC-AUC |
|---|---:|
| PCA linear SVM | 0.9852 ± 0.0005 |
| PCA logistic | 0.9816 ± 0.0010 |
| Elastic-net logistic | 0.9652 ± 0.0053 |
| Linear SVM | 0.9606 ± 0.0019 |
| HistGradientBoosting | 0.9603 ± 0.0002 |
| XGBoost | 0.9588 ± 0.0066 |

The recommendation rule selects **PCA logistic** because it is the simpler
model within 0.01 ROC-AUC of the best challenger.

Exploratory projection:

- 177 samples × 21 metabolites
- PC1: 22.893%
- PC2: 12.021%
- first two PCs: 34.915%

These labels represent longitudinal surgery time points, **not disease versus
control**. PCA/UMAP plots are exploratory and are not validation metrics.

The older p≫n simulation benchmark reports AUC 0.988 ± 0.006 over 100 simulated
datasets. Its null control was honest 0.497 versus leaky 0.914.

## API endpoints

### Core and UI

- `GET /`
- `GET /healthz`
- `GET /plugins`
- `GET /domain1-peaks`
- `POST /analyze`
- `GET /compounds`
- `POST /compounds-upload`

### Domain 1

- `GET /demo-spectrum`
- `POST /process-fid`
- `POST /process-spectrum`
- `GET /open-data`
- `POST /open-data/build`
- `GET /self-supervised/status`
- `POST /self-supervised/train`

### Domain 2

- `GET /biomarkers`
- `POST /biomarkers-upload`
- `GET /biomarkers-safe`
- `GET /biomarkers-model-suite`
- `POST /biomarkers-model-suite-upload`
- `GET /biomarkers-projection`
- `POST /biomarkers-projection-upload`
- `GET /biomarkers-benchmark`

### Laboratory workflow

- `GET /laboratory-workflow`
- `POST /laboratory-workflow/evaluate-qc`

## Tests

Last verified command:

```bash
backend/nmr_api/.venv/bin/python -m unittest discover \
  -s backend/nmr_api/tests -v
```

Current result: **17 tests passing**.

Test modules:

- `test_signal_processing.py`
- `test_self_supervised.py`
- `test_model_suite.py`
- `test_dimensionality.py`
- `test_laboratory_workflow.py`

## Critical limitations — do not overclaim

1. The default UI spectrum is synthetic.
2. Real raw-FID validation currently covers public pure-compound BMRB data,
   especially leucine. It does **not** yet establish performance on real serum
   mixtures.
3. Concentrations shown in the UI are pseudo/scaled abundance, not validated
   µM concentrations.
4. The current MTBLS242 table has 21 metabolite features, not the expected raw
   ~20,000-ppm-feature cohort.
5. Time point 0 versus 4 is not diabetes/hypertension versus control.
6. NMRTransformer and NMRformer are not currently active.
7. UMAP is exploratory; attractive clusters are not proof of biology.
8. The laboratory workflow is a machine-readable RUO design. There is no LIMS,
   authentication, electronic signature or durable append-only audit store.
9. This is research-use software, not a validated medical diagnostic.

## Important missing feature from the latest workshop screenshots

The application still does **not** implement a true multi-sample raw/processed
spectral cohort pipeline. Specifically missing:

- batch upload of many numeric spectra or Bruker experiments;
- interpolation onto one common ppm grid;
- cross-sample peak/spectral alignment;
- fixed-width or adaptive/intelligent spectral binning;
- cohort normalization such as PQN/total-area normalization;
- sample × ppm-bin matrix export;
- stacked/overlaid multi-sample spectrum visualization;
- PCA/UMAP directly from those binned raw spectra.

Current PCA/UMAP operates on the 21-metabolite Domain 2 results table, not on a
newly generated spectral-bin matrix.

If implementing the screenshot requirement, the desired flow is:

```text
multiple spectra
→ common ppm grid
→ global/segment alignment
→ water/artifact masking
→ quantitative normalization
→ fixed and/or adaptive binning
→ sample × bin matrix
→ multi-spectrum overlay
→ PCA/UMAP
→ leakage-safe statistics/model comparison
```

This is the highest-priority missing computational feature.

## Recommended next work

1. Implement and validate the multi-spectrum alignment/binning pipeline above.
2. Test it on genuinely independent serum/plasma spectra with manual or
   Chenomx-reviewed ground truth.
3. Add matrix-specific internal-standard calibration for real concentrations.
4. Obtain true disease/control labels and the full raw ~20k-feature matrix.
5. Add batch/QC metadata to PCA/UMAP and test run-order drift.
6. Connect sample identity, QC decisions and report authorization to a LIMS and
   durable audit system.
7. Configure NMRTransformer/NMRformer only after installing and validating the
   actual runtimes.

## Instructions for the next AI agent

- Read this file before making claims about completeness.
- Preserve the distinction between synthetic, pure-standard, longitudinal and
  clinical validation.
- Do not report pseudo-abundance as µM.
- Keep all feature selection, scaling, imputation and PCA inside training folds
  for predictive metrics.
- Do not use full-cohort UMAP/PCA separation as classifier performance.
- Preserve existing user changes; the workspace may be dirty and has no Git
  history to recover from.
- Use `apply_patch` for edits and the existing virtual environment for tests.
- After code changes run the full 17-test suite, compile Python, validate
  `run.sh`, and check frontend JavaScript syntax.
