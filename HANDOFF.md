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

The app registry (`DATASETS` in `main.py`) currently exposes:

- `mtbls242` — gastric-bypass longitudinal time-point task, serum NMR (`kind: longitudinal`).
- `mtbls1` — type-2 diabetes vs control, urine NMR (`kind: labeled`).
- `mtbls424` — breast-cancer relapse vs no-relapse, serum NMR (`kind: labeled`).
- `mtbls147` — healthy reference cohort, plasma NMR (`kind: reference`; no group
  contrast — only `/reference-ranges`, not biomarker discovery).
- `mtbls356` — antiphospholipid syndrome (thrombotic vascular disease) vs healthy
  donor, serum NMR (`kind: labeled`).

The `NCD_PANEL` in `main.py` maps Thailand's major NCDs to these labeled cohorts;
`/ncd-screen` reports the honest leakage-safe AUC for each. Current honest AUCs
(from `biomarker_engine.discover`, the exact path `/ncd-screen` uses): diabetes
**0.925**, cardiovascular/APS **0.764** (perm-p 0.0099), cancer-relapse **0.573**.
See "How to add a new NCD cohort" below for the procedure that produced `mtbls356`.

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
- Confidence-gated profile backend for processed CSV/TSV spectra:
  - `POST /profile/qc`
  - `POST /profile/auto`
  - `GET /profile/triage`
  - `GET /profile/report`
  - `GET /profile/report.csv`
  The profile result contract lives in `profile_schema.py`; orchestration lives
  in `profile_workflow.py`. Auto-profile reuses peak picking, pattern/NMRformer
  assignment, NNLS deconvolution, target-decoy FDR, bootstrap concentration
  intervals, and full per-metabolite provenance.

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
   MTBLS1, MTBLS424 and MTBLS356 provide labeled demos, but final clinical claims
   need independent validation. Name each cohort's disease as it actually is —
   e.g. MTBLS356 is antiphospholipid syndrome (a thrombotic vascular disease),
   used as the cardiovascular panel slot, not a general ischemic-heart-disease
   cohort. A true AMI cohort (MTBLS395) exists but its outcome labels are
   ethically withheld and cannot be used. See "How to add a new NCD cohort".

8. **The laboratory workflow is RUO design only.**  
   There is no full LIMS, authentication, electronic signature or durable
   append-only audit store.

9. **Frontend is a large single-file app.**  
   It works for a hackathon demo, but long-term maintainability would benefit
   from modularization.

## How to add a new NCD cohort (data → registry → AUC test)

This is the exact, reproducible procedure used to add `mtbls356` (the
cardiovascular/APS cohort). Follow it to wire any new MetaboLights NMR
disease cohort into the NCD panel and AUC-test it honestly. **Open data only —
never the closed competition dataset.**

### 0. What a usable cohort needs (both gates must pass)

A candidate study must have **both**:

1. A **populated per-sample MAF** — `m_*_maf.tsv` whose sample columns (column 19
   onward) hold real numeric concentrations, one column per sample. Many NMR
   depositions ship an *empty* MAF (group-summary columns only, or all blank) —
   reject these.
2. **Open binary labels** — a `Factor Value[...]` column in `s_*.txt` with a real
   two-class contrast (disease vs control). Many clinical outcomes are
   **ethically withheld** (`NA` / "not free available for ethical restrictions") —
   reject these (e.g. `MTBLS395`, a rich AMI cohort whose death labels are
   withheld and therefore unusable).

Both gates exist because most cardiovascular NMR studies fail one of them.

### 1. Find candidates (EBI Search + FTP probe)

```bash
# Candidate accessions for a disease area:
curl -sS "https://www.ebi.ac.uk/ebisearch/ws/rest/metabolights?query=<DISEASE_TERMS>&size=40&fields=name,technology_type&format=json"
# Then for each accession, probe the real files on the EBI FTP:
BASE="https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public"
curl -sS "$BASE/<ACC>/"                       # list m_/s_/a_ files
curl -sS "$BASE/<ACC>/m_<ACC>..._maf.tsv"     # check sample columns are numeric
curl -sS "$BASE/<ACC>/s_<ACC>.txt"            # check Factor Value distribution
```

Verify the MAF sample-column headers (row 1, cols 19+) **match** the `Sample Name`
column in `s_*.txt`, and that the chosen factor splits cleanly into two classes.

### 2. Two real gotchas to clean (both bit us on `mtbls356`)

- **European comma-decimals.** Some MAFs store `26,2` instead of `26.2`;
  `biomarkers.build_matrix` then silently coerces ~90% of values to `NaN`.
  Fix when building the demo TSV: in **sample columns only** (index ≥ 18),
  replace `,`→`.` for cells matching `^-?\d+,\d+$`. Leave annotation columns
  (SMILES/InChI/chemical_shift) untouched. Confirm with: parsed matrix has
  near-zero `NaN` afterward.
- **Missing cells.** A few genuinely-blank cells are normal. The discovery
  pipeline now tolerates them (`biomarker_engine._impute_median`, leakage-safe
  per-fold). Do **not** fabricate values in the stored TSV to "fix" this.

### 3. Build the two project files

Drop into `backend/nmr_api/open_data/`:

- `demo_<acc>.tsv` — the cleaned MAF (decimals normalized as above).
- `demo_<acc>_labels.json` — `{ "<SampleName>": 0|1, ... }`, derived from the
  **authoritative `Factor Value` column** in `s_*.txt` (not from guessing on the
  sample-name prefix). Map disease→1, control→0.

### 4. Register it (two edits in `main.py`)

```python
# DATASETS:
"<acc>": {
    "label": "<ACC> — <disease> vs <control> (<matrix> NMR)",
    "kind": "labeled",
    "tsv": _OPEN / "demo_<acc>.tsv",
    "labels": _OPEN / "demo_<acc>_labels.json",
    "task": "<disease> vs <control>",
    "class_names": {0: "<control>", 1: "<disease>"},
    "source": "MetaboLights <ACC> (¹H NMR, <matrix>; <n_case> vs <n_ctrl>)",
},
# NCD_PANEL:
"<ncd_key>": {"ncd": "<honest disease name>", "dataset": "<acc>",
              "thai_burden": "<honest burden context>"},
```

Honesty rule: name the disease the cohort **actually** contains. `mtbls356` is
labeled "antiphospholipid syndrome (thrombotic vascular disease)", not "general
cardiovascular disease", because that is what the data is.

### 5. AUC-test it (must reproduce a known cohort first)

Run the leakage-safe nested-CV that `/ncd-screen` uses. The eval imports only the
light modules (`biomarkers`, `biomarker_engine`) — no FastAPI/torch needed:

```python
from nmr_api import biomarkers, biomarker_engine
raw = open("backend/nmr_api/open_data/demo_<acc>.tsv","rb").read()
lab = {k:int(v) for k,v in json.load(open(".../demo_<acc>_labels.json")).items()}
X,_,_ = biomarkers.build_matrix(raw)
rows  = [s for s in X.index if s in lab]
M = X.loc[rows].dropna(axis=1, how="all")
y = np.array([lab[s] for s in rows]); groups = np.array(rows)  # 1 sample = 1 patient
res = biomarker_engine.discover(M.values, y, k=8, repeats=3,
                                feature_names=list(M.columns), groups=groups)
# res: honest_roc_auc, honest_q2, permutation_p_value, stable_panel, leaky_roc_auc
```

**Sanity gate:** always run `mtbls1` through the same harness in the same pass and
confirm it still reports **AUC ≈ 0.925** (diabetes). If that drifts, your harness
or an edit is wrong — fix before trusting the new number.

### 6. Interpret the AUC honestly (do not chase a big number)

- Compare `honest_roc_auc` to `leaky_roc_auc`. A large gap = the honest CV is
  correctly refusing an inflated estimate. **A near-zero gap with a low AUC means
  the biological signal is genuinely weak — report it, do not "fix" it.**
  (`mtbls424` relapse: honest 0.573, leaky 0.571 → the signal isn't there; that is
  the truth, not a bug.)
- `permutation_p_value < 0.05` is the real test that a moderate AUC is a true
  signal (`mtbls356`: 0.764 with p=0.0099 is a genuine, defensible result on n=54).
- Signal flag thresholds used by `/ncd-screen`: `≥0.85 strong`, `≥0.70 moderate`,
  else `weak`.
- An AUC near 1.0 on a small cohort is a **red flag for leakage**, not a success.

## Profiler workflow — train on H100, reuse, or neither?

The proposed confidence-gated profiler flow (QC gate → auto-profile → triage →
quantify → NCD panel → report; AI proposes, human verifies only the flagged
cases) needs **almost no new training**. Most stages are deterministic math or
models the repo already ships. The H100 is required **only** for the optional
"wow" upgrades, and every one of those trains on open or simulated data.

### A. Already a trained open-source model in the repo — REUSE, no training

- **Metabolite assignment (Stage 2)** — `NMRformer/onedTrans_0.9782` (bundled,
  72 classes, ~97.8% top-1, inference-only). Open-source, pretrained.
- **Spectrum representation / fingerprint** — masked-spectrum SSL encoder
  `models/masked_nmr_encoder.pt` (trained on open BMRB). Feeds confidence and the
  Track-2 fingerprint.

### B. No model at all — deterministic / statistical (no training, no H100)

- **Stage 1 QC gate** — SNR, baseline flatness, residual-water integral, TSP/DSS
  referencing: rule-based thresholds.
- **Stage 2 deconvolution** — NNLS against the reference library: convex
  optimization, no learned weights.
- **Stage 3 confidence + FDR** — target-decoy FDR plus fit-residual and the
  NMRformer assignment probability, combined arithmetically. (Optional: a tiny
  logistic/isotonic *calibration* of that score — CPU, seconds, not H100.)
- **Stage 4 quantify** — NNLS concentrations + bootstrap uncertainty.
- **Stage 6 report/provenance** — templating only.

### C. Trained at runtime, but CPU not H100

- **Stage 5 NCD panel** — classical ML (elastic-net / SVM / HistGB / XGBoost from
  `model_suite.py`) fit per-cohort on the small open MTBLS tables in seconds. No
  deep learning; never touches the GPU.

### D. Optional NEW H100 training on open data (the upgrades that win)

Ship the whole workflow without these; add them only to beat the baseline.

1. **Low→high-field upconverter** (the hero demo) — genuinely needs the H100.
   Data: degrade open high-field MTBLS/BMRB spectra to benchtop (~60 MHz)
   resolution, train a recovery model. No off-the-shelf NMR field-upconverter
   exists to reuse.
2. **NN-based quantifier** — optional upgrade over NNLS. Data: GISSMO-simulated
   mixtures with known concentrations (open). Supervised regression on the H100.
3. **NMRformer fine-tune** — only to expand beyond its 72 metabolites or to
   instrument-match the competition matrix. Open BMRB/GISSMO data on the H100.

**Bottom line:** the human-in-the-loop profiler ships with **zero new training** —
reuse NMRformer + the SSL encoder, plus deterministic math and CPU classical ML.
Spend the H100 only on the optional field-upconverter / NN-quantifier that turn a
solid tool into a hackathon-winning one. All training stays open-data-only; the
closed competition dataset is never used for training (see data-governance notes).

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
