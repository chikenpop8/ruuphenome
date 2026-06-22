# RuuPhenome NMR API

FastAPI backend for NMR metabolite recognition at the National Phenome Institute.
It predicts ¹H NMR chemical shifts using the open-source **NMRTransformer** model
and matches them against annotated spectrum peaks to identify and rank metabolites.

## Architecture

```
backend/nmr_api/
├── main.py                  FastAPI app + HTTP endpoints
├── pipeline.py              Parse Domain-2 table → predict → score
├── shifts_db.py             NMRTransformer integration + HMDB fallback shifts
├── signal_processing.py     Domain 1 — nmrglue + scipy (FID → spectrum → peaks)
├── biomarkers.py            Domain 2 — scikit-learn biomarker discovery
├── dimensionality.py        Domain 2 — PCA loadings + exploratory UMAP
├── library.py               Compound library for the Profiler UI
├── models.py                Pydantic request/response schemas
├── static/profiler.html     Chenomx-style web UI
├── requirements.txt         Runtime dependencies
├── run.sh                   Start the dev server (port 8100)
└── setup_nmrtransformer.sh  Optional: install the full NMRTransformer model
```

### Connected plugins (the two-domain pipeline from the project doc)

| Stage | Plugin | Role |
|-------|--------|------|
| **Domain 1 — signal processing** | **nmrglue** + **scipy** | Safe Bruker import → digital-filter removal → FFT → 0th/1st-order phase → ALS baseline → robust peaks + QC |
| **Domain 1 — assignment** | Hybrid NMRformer adapter + pattern matcher | Neural support when configured, plus shift coverage, ppm error, SNR and ambiguity |
| **Domain 1 — representation learning** | PyTorch masked autoencoder | Self-supervised embeddings from unlabeled augmented open BMRB spectra |
| **Shift prediction** | **NMRTransformer** (open-source) + HMDB fallback | Predicted ¹H shifts per metabolite |
| **Domain 2 — biomarkers** | **scikit-learn + XGBoost** | Nested patient-grouped logistic/SVM/boosting comparison → calibrated performance + stable panel |
| **Domain 2 — dimensionality** | **scikit-learn PCA + umap-learn** | PCA scores/loadings and exploratory nonlinear neighborhoods |
| **Structure rendering** | **SmilesDrawer** (frontend) | 2D molecule from SMILES |
| **Web stack** | FastAPI · pandas · numpy | API + data handling |

`GET /plugins` reports the live status/version of every one. The active shift
backend is also reported in every `/healthz` and `/analyze` response, so results
stay auditable. This replaces the closed, subscription **Chenomx** software the
project sets out to retire.

## Quick start

```bash
# 1. Run the API (auto-creates venv, installs deps)
bash backend/nmr_api/run.sh

# 2. Open the interactive docs
open http://127.0.0.1:8100/docs

# 3. (Optional) Enable full NMRTransformer prediction
bash backend/nmr_api/setup_nmrtransformer.sh
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET`  | `/healthz` | Health + which prediction backend is active |
| `GET`  | `/plugins` | Live status/version of every connected plugin |
| `GET`  | `/domain1-peaks` | The default Domain 1 annotated peak list |
| `GET`  | `/laboratory-workflow` | Thirteen-stage real-laboratory workflow, roles, records and release gates |
| `POST` | `/laboratory-workflow/evaluate-qc` | Evaluate spectrum/batch evidence for pass, fail or needs-review |
| `POST` | `/analyze` | Upload a Domain 2 NMR results TSV → ranked metabolite matches |
| `GET`  | `/compounds` | Compound library for the Profiler UI |
| `POST` | `/compounds-upload` | Same, from an uploaded TSV |
| `GET`  | `/demo-spectrum` | **Domain 1**: synthesize + process a serum spectrum (nmrglue/scipy) |
| `POST` | `/process-fid` | **Domain 1**: upload a zipped Bruker FID → spectrum, peaks, assignments + QC |
| `POST` | `/process-spectrum` | **Domain 1**: upload processed CSV/TSV (`ppm`, `intensity`) → peaks, assignments + QC |
| `GET`  | `/open-data` | Open BMRB corpus manifest, provenance, checksums and status |
| `POST` | `/open-data/build` | Download and process the curated raw BMRB corpus |
| `GET`  | `/self-supervised/status` | Masked-spectrum encoder status and training report |
| `POST` | `/self-supervised/train` | Retrain the self-supervised encoder |
| `GET`  | `/biomarkers` | **Domain 2**: biomarker discovery (scikit-learn) on the demo dataset |
| `POST` | `/biomarkers-upload` | **Domain 2**: discovery on an uploaded TSV |
| `GET`  | `/biomarkers-safe` | Patient-grouped leakage-safe biomarker panel |
| `GET`  | `/biomarkers-model-suite` | Nested comparison: raw-feature and fold-internal PCA logistic/SVM, gradient boosting and XGBoost |
| `POST` | `/biomarkers-model-suite-upload` | Run the same model suite on an uploaded TSV |
| `GET`  | `/biomarkers-projection` | Exploratory PCA scores/loadings and UMAP coordinates |
| `POST` | `/biomarkers-projection-upload` | PCA/UMAP for an uploaded results TSV |
| `GET`  | `/` | The Chenomx-style Profiler web UI |

### Example: `/analyze`

```bash
curl -X POST http://127.0.0.1:8100/analyze \
  -F 'nmr_results_tsv=@"/Users/bigray/Downloads/Domain_2_NMR_results_MTBLS242 (2).tsv"' \
  -F 'tolerance_ppm=0.05'
```

Response (truncated):

```json
{
  "summary": {
    "total_metabolites": 21,
    "total_samples": 465,
    "metabolites_with_smiles": 20,
    "prediction_backend": "HMDB-fallback",
    "tolerance_ppm": 0.05
  },
  "matches": [
    {
      "metabolite": "histidine",
      "chebi_id": "CHEBI:...",
      "predicted_shifts": [3.19, 3.28, 7.08, 7.78],
      "peaks_matched": 5,
      "match_score": 100.0,
      "mean_abundance": 0.0123,
      "cv_percent": 18.4
    }
  ]
}
```

## How it fits RuuPhenome

1. **Domain 1** (pattern identification) — the annotated peak list a spectroscopist
   would normally read by hand. Served at `/domain1-peaks`, used as the match target.
2. **Domain 2** (study table) — the MetaboLights metabolite + abundance table.
   Uploaded to `/analyze`.
3. The engine replaces manual peak reading: predicts shifts for each candidate and
   scores the overlap, returning a ranked, explainable identification table plus
   per-metabolite abundance statistics (mean, CV%, detection rate) for biomarker work.

## Domain 1 processing quality

The upgraded Domain 1 path returns:

- acquisition-aware ppm coordinates instead of assuming every spectrum is centered identically;
- Bruker digital-filter removal and safe temporary extraction;
- automatic zero- and first-order phase correction;
- asymmetric least-squares baseline correction;
- robust MAD noise estimation and prominence/width-aware peak picking;
- artifact flags for the residual-water region;
- confidence-ranked metabolite assignments based on complete shift patterns;
- a 0–100 QC score with SNR, negative-area, peak-count and phase diagnostics.

On a 30-spectrum synthetic stress benchmark containing receiver phase errors,
first-order phase distortion, DC offsets and noise, the original prototype
averaged 422 detected peaks with peak F1 0.22. The upgraded path averaged 28
detected peaks, retained 96% of true resonances and reached peak F1 0.74.
This is a synthetic engineering benchmark, not a substitute for validation on
real Bruker serum spectra.

Reproduce it with:

```bash
backend/nmr_api/.venv/bin/python -m backend.nmr_api.benchmark_domain1 --spectra 30
```

NMRformer is **not** currently claimed as active. The assignment field reports
`reference-pattern-matcher` until a trained NMRformer runtime is installed and
validated against appropriate serum ground truth.

### Optional NMRformer adapter

Domain 1 now supports `assignment_backend=hybrid|pattern-matcher|nmrformer`.
Hybrid mode is the default and falls back safely to the pattern matcher. To
activate NMRformer, provide a validated local Python adapter:

```bash
export NMRFORMER_ADAPTER_MODULE=my_nmrformer_adapter
```

The module must expose:

```python
def predict_assignments(ppm, intensity, peaks):
    return [{"metabolite": "citrate", "probability": 0.92}]
```

The hybrid score currently weights transparent pattern evidence at 65% and
NMRformer support at 35% until serum-specific external validation justifies a
larger neural contribution.

### Open-data self-supervised encoder

The repository includes a curated corpus of 12 raw Bruker 1D ¹H experiments
from the Biological Magnetic Resonance Bank for serum-relevant metabolites.
Every file records its BMRB entry, DOI, source page, download URL, checksum,
acquisition metadata and processing QC in
`open_data/provenance.json`. BMRB states that its metabolomics standards data
are publicly available free of charge.

Build or refresh the corpus:

```bash
backend/nmr_api/.venv/bin/python -m backend.nmr_api.open_data
```

Train the masked-spectrum convolutional autoencoder:

```bash
backend/nmr_api/.venv/bin/python -m backend.nmr_api.self_supervised
```

Pretraining creates random unlabeled mixtures, masks contiguous spectral
regions and learns to reconstruct them. Compound names are not supplied during
pretraining. Afterward, embeddings of the known pure standards form an
explainable nearest-reference index.

Current training corpus:

- alanine, glutamine, histidine, leucine, phenylalanine, tyrosine and valine;
- citrate, glycine, creatinine, lactate and glucose;
- 4096 points from 10 to 0 ppm, DSS/TSP/TMS referenced when detected.

The masked-reconstruction loss decreased from 0.0242 to 0.0127. On 120
shifted/noisy augmented queries, reference retrieval achieved 0.975 top-1 and
1.000 top-5 accuracy. This is an invariance/retrieval check against the same
reference collection, not external biological-mixture accuracy.

The API returns these similarities as supporting evidence under
`self_supervised_matches`; they do not silently override chemically
explainable assignments.

### Real-laboratory operation

The complete accession-to-archive design is documented in
[`../../LABORATORY_PIPELINE.md`](../../LABORATORY_PIPELINE.md). The API exposes
the same workflow as structured JSON at `GET /laboratory-workflow`.

`POST /laboratory-workflow/evaluate-qc` combines the existing per-spectrum
metrics with pooled-QC precision, drift, blanks, instrument suitability and
identity checks. Missing evidence returns `needs_review`, so a good-looking
spectrum cannot be released without its laboratory context. The bundled limits
are visible research defaults and must be replaced by validated matrix-specific
limits before regulated or clinical use.

### Domain 2 model policy

- Feature selection and parameter choice happen inside training folds.
- PCA challengers use all available features, with imputation, scaling and PCA
  fitted independently inside every training fold.
- Entire patients are held out together in outer cross-validation.
- The simplest model within 0.01 ROC-AUC of the best model is recommended.
- Full-cohort PCA/UMAP coordinates are visualization only. They are not used as
  evidence of classifier accuracy or external biological separation.
- CatBoost is registered but intentionally ineligible until categorical
  clinical metadata is available.
- A Domain 2 neural network is intentionally disabled for the current small
  cohort; deep learning is concentrated in pretrained Domain 1 assignment.

Current MTBLS242 time-point 0 vs 4 result (two repeated patient-grouped outer
CV runs): PCA linear SVM ROC-AUC 0.9852, PCA logistic 0.9816, elastic-net
logistic 0.9652. The recommendation rule selects PCA logistic because it is the
simpler model within 0.01 ROC-AUC of the best challenger.

The exploratory cohort projection contains 177 samples and 21 metabolites.
PC1 explains 22.893% and PC2 12.021% of standardized variance. Open the UI
Tools menu → **PCA / UMAP visualization (Domain 2)** for score plots and loading
tables.
