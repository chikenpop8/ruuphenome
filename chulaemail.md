# RuuPhenome — Developer Handoff (`chulaemail.md`)

> **Read this before touching the code.** This document is written for a technical
> teammate with **zero prior context**. It explains what the project is, what
> works, how to run it, what broke before, and what not to change carelessly.
> For even deeper detail see [`HANDOFF.md`](HANDOFF.md) (module-by-module) and
> [`docs/H100_TRAINING.md`](docs/H100_TRAINING.md) (cluster training).

**Last updated:** 2026-07-02 · **Branch:** `feat/open-data-corpus-builder`

---

## 0. TL;DR (the 60-second version)

- **What:** An open, auditable ¹H-NMR metabolomics tool (a free alternative to
  Chenomx) for the **BDI Young Innovator Hackathon 2026, Track "Phenome."**
- **Two tracks:**
  - **Track 1** = turn an NMR spectrum into a **metabolite profile** (names + concentrations).
  - **Track 2** = take two tables (concentrations + metadata) and find the
    **biomarkers** that separate sample conditions.
- **Most of the app needs NO training.** Track 2 is classical ML that fits fresh
  on each upload. The **only** trainable model is an *optional* self-supervised
  (SSL) encoder that enriches Track 1 identification.
- **The competition dataset is closed & governed.** Train only on **open data**.
  Never copy the closed dataset off its VM. Keep `NMR_OFFLINE=1` on that VM.
- **Biggest current limitation:** the SSL training corpus is **12 spectra**. That
  is the quality ceiling — see §7 and §8.

---

## 1. Project overview

### What this project is
RuuPhenome is a browser-based, on-premise, fully auditable pipeline for ¹H-NMR
metabolomics. It replaces the closed desktop tool **Chenomx** with something that
is free, runs inside your own VM (no data leaves the host), and exposes every
annotation / fit / score for inspection.

### The main scientific / technical goal
Go **end-to-end** on NMR evidence:

1. **Track 1 (Domain 1) — profiling:** clean a spectrum (phase, baseline, peak
   pick, QC), assign peaks to metabolites, and report identities + concentrations.
2. **Track 2 (Domain 2) — biomarker discovery:** from a cohort's metabolite
   concentration table + metadata, find the metabolites that indicate sample
   condition, using **leakage-safe** statistics with honest performance metrics.

Chenomx stops at Track 1. RuuPhenome carries through to Track 2 (biomarkers +
pathway biology), which is the differentiator.

### What we are currently focusing on
- **Track 2 robustness** for the *real* competition data (arrives ~2026-07-03):
  tolerant parsing + a pre-flight preview so a mis-read file is caught instantly.
- **Track 1 identification quality** via the optional SSL encoder — specifically
  whether to invest an **H100 run** in it, and how to keep it **solvent-consistent**
  (the competition matrix is blood run in **D2O**).

---

## 2. Current project status

### Implemented & working
- **Track 1 single-spectrum profiling:** raw Bruker FID → digital-filter removal
  → auto-phase → ALS baseline → peak picking → QC → reference-pattern assignment
  → metabolite list + concentrations. (nmrglue-based, deterministic.)
- **Track 1 binned-cohort pipeline:** upload one binned matrix (samples × ppm
  bins, optional inline `Class`/`Group` column) → annotate → quantify → (if a
  label exists) biomarker discovery + biology.
- **Track 2 two-table biomarker discovery** (the exact hackathon spec):
  `POST /track2/biomarkers` takes Table 1 (metabolites × samples) + Table 2
  (metadata) → leakage-safe nested-CV discovery.
- **Track 2 pre-flight preview** (`POST /track2/preview`): reports detected shape,
  condition column, class balance, sample-ID match rate, and warnings — before
  running anything.
- **Per-biomarker effect sizes:** direction (up/down), fold-change, single-marker
  AUC, selection stability — surfaced in the results UI.
- **Biological interpretation:** pathway over-representation (hypergeometric)
  enrichment on the discovered panel.
- **PCA / UMAP** exploratory projections (visualization only).
- **Self-supervised (SSL) encoder:** a small masked-spectrum autoencoder trained
  on open BMRB spectra; adds an optional similarity score to Track 1 IDs.
- **Browser UI** (`static/profiler.html`): menus, spectrum viewer, Track 2 upload +
  preview + results, cohort chooser, reference card, etc.

### Tested / verified
- **43 unit/integration tests pass** (`unittest`, see §4). Covers signal
  processing, spectral cohort, biomarker validation, model suite, biology,
  dimensionality, offline guard, profile workflow, SSL.
- **Track 2 verified end-to-end against 6 real, external MetaboLights studies**
  (not just synthetic data) during development.
- The **SSL retrieval benchmark** runs and reports top-1 / top-5 / MRR — but see
  the honesty caveat in §7 (it is *in-distribution / augmented*, not external).

### Incomplete / uncertain
- **SSL encoder is data-limited:** trained on **12 D2O reference compounds** only.
  It already converges; more epochs alone won't help (see §7, §8).
- **Per-shift solvent/pH provenance is "unverified":** reference shifts come from
  HMDB 5.0 (aqueous/D2O by convention) but are **not** verified per compound. We
  deliberately do **not** fabricate solvent corrections.
- **Solvent selector / guard for Track 1** is planned but not yet built.
- **The real competition data has not arrived yet** — parsing is hardened against
  plausible surprises but not proven on the actual file.

---

## 3. Repository / code structure

```
ruuphenome/
├── README.md                     Project pitch + quick start
├── HANDOFF.md                    Deep module-by-module handoff (read alongside this)
├── chulaemail.md                 <-- this file
├── LABORATORY_PIPELINE.md        Lab-ops / release-gate notes
├── PROFILER_BACKEND_BUILD.md     Backend build spec
├── docs/
│   ├── H100_TRAINING.md          Exact LiCO/H100 job steps + command
│   └── DATA_SOURCES.md           Open-data source list + licenses
├── backend/nmr_api/              THE APP (FastAPI backend + static UI)
│   ├── main.py                   All HTTP endpoints (FastAPI app = `app`)
│   ├── signal_processing.py      Track 1 DSP: phase, baseline, peaks, QC (nmrglue)
│   ├── pipeline.py               Table readers, name aliases, robust parsing helpers
│   ├── biomarkers.py             build_matrix(): Table-1 → samples×metabolites matrix
│   ├── biomarker_engine.py       Track 2 core: leakage-safe nested-CV discover()
│   ├── model_suite.py            Multi-model comparison (elastic-net/SVM/HGB/XGB)
│   ├── spectral_cohort.py        Metadata parse + label derivation + solvent notes
│   ├── biology.py                Pathway enrichment + metabolite biology cards
│   ├── dimensionality.py         PCA / UMAP projections (viz only)
│   ├── shifts_db.py              HMDB reference shifts + optional NMRTransformer
│   ├── nmrformer_adapter.py      Optional 3rd-party NMRformer neural assignment
│   ├── nmrformer_backend.py      Hybrid pattern+neural contract
│   ├── self_supervised.py        SSL masked-spectrum encoder (THE trainable model)
│   ├── train_on_h100.py          H100/LiCO training entry point (wraps SSL train)
│   ├── open_data.py              Open BMRB corpus download + processing
│   ├── build_open_corpus.py      Expanded open-corpus builder (SSL pretraining data)
│   ├── build_bmrb_library.py     Build reference-shift library from BMRB
│   ├── library.py                Reference metabolite library (cards, shifts)
│   ├── enrich.py                 PubChem enrichment (offline-guarded)
│   ├── laboratory_workflow.py    LIMS-style release-gate description (not core)
│   ├── models.py                 Pydantic response models
│   ├── static/profiler.html      THE UI (single file: HTML+CSS+JS)
│   ├── models/                   Trained SSL checkpoints + training reports
│   ├── open_data/                Bundled corpus (.npz), provenance, demo labels
│   ├── requirements.txt          Python deps
│   ├── run.sh                    Dev server launcher (uvicorn --reload)
│   └── tests/                    43 tests (unittest)
└── nmr_pipeline/                 Standalone CLI experiment (peripheral)
```

### Training / evaluation / checkpoint / reporting flow (SSL encoder)

This is the only "ML training" in the project. Everything else fits fresh.

```
open_data/bmrb_1h_corpus.npz          (12 D2O reference spectra, 4096-pt grid, 10→0 ppm)
        │
        ▼
self_supervised._load_corpus()        loads spectra/labels/ppm
        │
        ▼
self_supervised._augment_batch()      random mixtures + shift + noise + masking
        │                              (labels NOT used → truly self-supervised)
        ▼
self_supervised.train()               masked-reconstruction loss; AdamW
        │
        ├─► models/masked_nmr_encoder.pt          (checkpoint: weights + reference embeddings)
        └─► models/masked_nmr_training.json       (loss history + config)
        │
        ▼
self_supervised.benchmark_retrieval() top-1 / top-5 / MRR on augmented queries
        │
        ▼
train_on_h100.main()                  wraps all of the above for the cluster:
        ├─ ensures corpus exists (download if missing; skip on cluster)
        ├─ trains with H100-sized args
        ├─ benchmarks retrieval
        ├─► models/h100_training_report.json      (full run report)
        └─► auto-saves CONFIG-TAGGED COPIES of all three files
             e.g. masked_nmr_encoder_200ep_cuda_b256.pt   (see §5, §6)
```

- **Inference use:** `signal_processing.py` calls `self_supervised.status()` and,
  only if a trained checkpoint exists, `self_supervised.identify()` to append a
  `self_supervised_similarity` score to Track 1 assignments. It is wrapped in
  try/except and **degrades gracefully** — no checkpoint = feature silently off.

---

## 4. Important commands

> Python venv lives at `backend/nmr_api/.venv`. All commands below assume it.

### Environment setup
```bash
cd ruuphenome/backend/nmr_api
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run the server (dev)
```bash
bash backend/nmr_api/run.sh
# → http://127.0.0.1:8100         (UI)
# → http://127.0.0.1:8100/docs    (Swagger API)
# → http://127.0.0.1:8100/healthz (health)
# Expose on the VM for judges:  NMR_HOST=0.0.0.0 bash backend/nmr_api/run.sh
```
`run.sh` uses `uvicorn --reload`, so **edits hot-reload** in this mode.
⚠️ The IDE/preview launch config (`.claude/launch.json`) does **not** use
`--reload` — under that path, Python changes require a manual restart (see §6).

### Run the test suite (43 tests)
```bash
cd ruuphenome
backend/nmr_api/.venv/bin/python -m unittest discover \
  -s backend/nmr_api/tests -p "test_*.py" -t .
# Expected: "Ran 43 tests ... OK"
```
Tests import as `backend.nmr_api.*`, so **run from the `ruuphenome/` root** with
`-t .` (running from elsewhere gives `ModuleNotFoundError: No module named 'backend'`).

### Train the SSL encoder
```bash
# From backend/  (so the nmr_api package resolves)
cd ruuphenome/backend
python -m nmr_api.train_on_h100 \
  --epochs 200 --steps-per-epoch 128 \
  --batch-size 256 --embedding-dim 128
```
- **Runs anywhere:** auto-selects CUDA → MPS → CPU. On a laptop it falls back
  cleanly (slower). On the H100 use the args above.
- `--rebuild-corpus` re-downloads the open BMRB corpus (needs internet;
  **do NOT use on the closed cluster** — upload the bundled `.npz` instead).
- Outputs land in `backend/nmr_api/models/` (see the flow diagram in §3).

### Reproduce the current bundled result
The bundled checkpoint was a small **dev** run (MPS, `embedding_dim=64`, 20
epochs). To reproduce it locally:
```bash
cd ruuphenome/backend
python -m nmr_api.train_on_h100 --epochs 20 --embedding-dim 64 --batch-size 16
```
Current recorded numbers (`models/masked_nmr_training.json`):
`initial_loss 0.0242 → final_loss 0.0127`, retrieval `top1 0.975 / top5 1.0 /
MRR 0.9875` — **augmented/in-distribution only** (see §7).

### GPU / CPU / memory notes
- The whole serving app boots at ~**190 MB RAM**; every heavy lib (torch, xgboost,
  umap) is lazy-loaded. It runs comfortably on a small VM (8 GB is plenty).
- `run.sh` sets `OMP_NUM_THREADS=1` etc. + `KMP_DUPLICATE_LIB_OK=TRUE` to avoid a
  macOS OpenMP crash from XGBoost/sklearn. Harmless on Linux — leave it.

---

## 5. Details of recent fixes (this cycle)

All verified live and against the **53-test suite** (was 43; +10 Track-2 tests).
Nothing here is committed yet unless you were told otherwise — check `git status`.

### Track 2 analytics expansion — 2026-07-02 (items 1–5)

Built the five requested Track-2 capabilities. **No training / SSL / H100 needed** —
all classical statistics + classical ML that fit fresh per upload. The binary path
is byte-for-byte unchanged (prior AUC/Q²/panel numbers reproduce exactly).

1. **Full metrics + confusion matrix** — `biomarker_engine.discover()` and
   `model_suite.compare_models()` now return accuracy/sensitivity/specificity/
   precision/recall/F1 (+ confusion matrix) from the pooled out-of-fold predictions
   (macro forms + per-class recall for multi-class). New reusable helper
   `biomarker_engine.classification_metrics()`.
2. **Top-1/3/5/10 minimal-panel sweep** — `discover(panel_sizes=(1,3,5,10))` reruns
   the leakage-safe CV at each k and returns per-k AUC/F1/stable-panel
   (`result["panel_sweep"]`), answering "how few metabolites are enough".
3. **Multi-class (≥3 group) support** — end-to-end: `screen_features()` (ANOVA-F for
   >2), multinomial LR, macro one-vs-rest ROC-AUC + macro-F1 + per-class recall.
   `spectral_cohort.derive_labels(multiclass=True)` keeps ALL classes (0..K-1) instead
   of collapsing to the two largest. Added **RandomForest** to the model suite.
   The two-table UI/preview now request `multiclass=true` (2-class degrades to binary).
4. **Differential analysis** — new `differential.py` + `POST /track2/differential`:
   per-metabolite group means, log2 fold-change, Mann-Whitney U + Welch t (2-group) /
   Kruskal-Wallis + ANOVA (>2), **BH q-values across all metabolites**, effect size,
   and a **volcano** array. Descriptive full-cohort (no leakage concern — no accuracy claimed).
5. **Correlation analysis** — new `correlation.py` + `POST /track2/correlation`: pairwise
   Pearson/Spearman (FDR) **plus a partial-correlation Gaussian Graphical Model network**
   (Ledoit-Wolf shrinkage precision → partial correlations; Krumsiek et al. 2011,
   BMC Syst Biol 5:21) so edges are *direct* associations, not indirect co-variation.
   Optional metabolite-vs-numeric-covariate correlation. Guarded for p≫n (caps at 80
   most-variable features on raw spectra).

UI: `renderTrack2Results` now shows the metric set, confusion matrix, panel sweep, and
multi-class per-class recall, plus **Differential / Correlation / Panel-CSV** buttons that
call the new endpoints. Verified live in the browser preview (binary + 3-class).

### Earlier fixes (previous cycle)

1. **Removed `rdkit`** — it was only used for a version string in `/plugins`.
   Uninstalled (~104 MB reclaimed) and dropped from `requirements.txt`. Structures
   render via SmilesDrawer (frontend CDN). No functional loss.
2. **Removed the "Advanced" UI menu** — lab-workflow/plugin-status/engine-info/
   Swagger/health links were noise for the deliverable. **The endpoints still
   exist**; Swagger is at `/docs`, health at `/healthz`.
3. **Robust Table 1 / Table 2 parsing** (`pipeline.py`):
   - `sniff_separator()` — picks tab / `;` / `,` (handles EU-style CSVs).
   - `NUMERIC_MISSING_TOKENS` — treats `n.d.`, `<LOD`, `-`, etc. as missing in the
     **concentration** table (not the metadata table, where they may be categories).
   - `coerce_numeric()` — rescues European decimals (`"12,34"` → `12.34`) *only*
     when the whole cell matches `^\d+,\d+$`, so clean data is never corrupted.
4. **New pre-flight preview** — `POST /track2/preview` + UI step: reports shape,
   condition column, class balance, sample-ID match rate, and warnings
   (transpose, mostly-empty, ID mismatch). The Run button is disabled until the
   inputs are actually usable.
5. **Per-biomarker effect sizes** — `biomarker_engine._panel_effect_sizes()` adds
   `panel_stats` to `discover()` output: `direction`, `fold_change`,
   `log2_fold_change`, `univariate_auc`, `selected_in_folds`. Rendered as a table
   in Track 2 results ("Lactate ▲ higher in case, 1.69×, AUC 1.0").
6. **Cohort chooser (select-then-run)** — "Run biomarker discovery" now opens a
   landing screen; clicking a cohort selects it (green check + Run button
   appears), then Run executes. JS: `window.__selectCohort`,
   `window.__runSelectedCohort`, `runBiomarkerModelSuite()`.
7. **ppm-axis label fix** in `fitOverlaySvg()` (label collided with the leftmost tick).
8. **JSON serialization fix** in `discover()` — `n_splits` was a `numpy.int64`
   and 500'd on serialization; now wrapped in `int()`.
9. **Two-layer label-column auto-detection fix** (`derive_labels()`): (a) restrict
   candidate scoring to the samples actually present in Table 1 *before* choosing;
   (b) prefer real ISA-Tab semantic columns (`Characteristics[…]`/`Factor Value[…]`)
   over ontology-plumbing companions (`Term Accession Number`, etc.).
10. **Empty-data crash guard** in `discover()` — a `has_signal` check fails fast
    with a clear message instead of a cryptic sklearn crash when every
    concentration cell is missing.

---

## 6. Mistakes / problems that happened before (be honest, avoid repeating)

### 6.1 Checkpoint mislabeling by hand — *the classic one*
- **What happened:** someone ran a short (10-epoch) job, then manually
  `cp masked_nmr_encoder.pt masked_nmr_encoder_200ep_cuda_b256.pt` — silently
  overwriting a real long-run checkpoint with a mislabeled short one.
- **Why:** hand-typed filenames don't reflect what actually ran.
- **How it's prevented now:** `train_on_h100._checkpoint_suffix()` derives the
  tag **from the actual args + device** (`{epochs}ep_{device}_b{batch}`) and
  `_save_named_checkpoint()` writes the tagged copies automatically.
- **Rule:** **never hand-name or `cp` checkpoints.** Let the trainer name them.

### 6.2 Stale server: fresh UI, old backend
- **What happened:** a long-running server showed the *new* UI but a clicked
  button 404'd because the backend was stale.
- **Why:** the UI is served with `FileResponse`, which re-reads `profiler.html`
  from disk on every request (so **HTML/JS edits appear instantly**). But **Python
  code does not hot-reload** unless uvicorn was started with `--reload`. The IDE/
  preview launch config does not pass `--reload`.
- **How to avoid:** after editing **Python** under the preview/launch path,
  **restart the process**. `run.sh` includes `--reload`, so use it for dev. Verify
  new endpoints are live: `curl -s localhost:8100/openapi.json | grep track2`.

### 6.3 Wrong label-column auto-detection (real MetaboLights file)
- **What happened:** on MTBLS1497 the auto-detector picked "Organism part"
  (balanced across the *full* metadata) which collapsed to one class once
  restricted to Table 1's samples; then it picked an ISA-Tab *plumbing* column.
- **Why:** it scored columns on the full metadata before restricting to the
  analyzable samples, and didn't distinguish semantic columns from ontology
  companions.
- **Fixed** (see §5.9). **Rule:** always eyeball the pre-flight preview's chosen
  condition column before trusting a Track 2 result.

### 6.4 `numpy.int64` not JSON-serializable → 500
- Fixed with `int()` (§5.8). **Rule:** cast numpy scalars to Python types before
  putting them in an API response.

### 6.5 EU-decimal dtype assumption
- **What happened:** the EU-decimal rescue didn't fire — the guard checked
  `col.dtype == object`, but pandas 2.x reads those columns as the newer `str`
  dtype.
- **Fixed:** check `not pd.api.types.is_numeric_dtype(col)` instead.
- **Rule:** don't assume `object` dtype for text columns on modern pandas.

### 6.6 Stale absolute paths in recorded JSON
- `models/masked_nmr_training.json` records an **absolute** checkpoint path from
  an older directory layout (it points at `.../Noom copy cat/backend/...`, missing
  `ruuphenome/`). Harmless (informational) but don't trust those paths — recompute
  from the repo root.

---

## 7. Warnings & rules for future work

### Do not change carelessly
- **The leakage-safe CV structure** in `biomarker_engine.py` / `model_suite.py`.
  Feature selection, imputation, scaling, and PCA are fit **inside each training
  fold**. Moving any of them outside the fold reintroduces data leakage and
  inflates AUC. `leaky_roc_auc` exists **only** to show the gap — never quote it
  as performance.
- **The offline guard** (`enrich.offline_mode()` / `NMR_OFFLINE`). It blocks all
  outbound calls on the closed VM. Don't add un-guarded network calls.
- **Checkpoint naming** (§6.1). Don't hand-name.

### Common failure points
- Editing Python but not restarting under the preview config (§6.2).
- Running tests from the wrong directory (§4).
- Trusting an auto-detected condition column without checking the preview (§6.3).

### Dataset-size limitation (important)
- The SSL corpus is **12 compounds** (`open_data/bmrb_1h_corpus.npz`). Data
  augmentation makes unlimited *synthetic mixtures*, but the *information ceiling*
  is 12 compounds' peak patterns. **More epochs / a bigger model cannot exceed
  that ceiling.** Expanding the corpus is the real lever, not GPU time.

### Overfitting & scientific-interpretation risks
- **Loss ↓ while retrieval ↓ is possible.** Masked-reconstruction loss dropping
  does not guarantee better metabolite discrimination. **Watch retrieval, not just
  loss.**
- **The retrieval benchmark is in-distribution / augmented** — it queries against
  the *same* reference collection it trained on. `top1 = 0.975` is **not** an
  external mixture-identification accuracy. Do **not** put it on a slide as
  "97.5% identification accuracy." A real external/held-out eval is needed before
  any gain is trustworthy (see §8).
- **PCA/UMAP separation is not classifier accuracy** — it's fitted on the shown
  samples for visualization only. Quote `honest_roc_auc` + `permutation_p_value`
  + `honest_q2` for predictive claims.
- **Solvent / exchangeable protons:** references are D2O-conventional but per-shift
  solvent/pH is **"unverified."** In D2O, exchangeable protons (COOH/OH/NH) can
  shift or vanish. **Do NOT fabricate per-compound solvent corrections** — surface
  the caveat honestly (that's the standing rule for this project).
- **The SSL encoder is OPTIONAL.** Keep it optional. Do not make Track 1 or Track 2
  hard-depend on a checkpoint being present.

### Data governance (non-negotiable)
- **Train on open data only.** The closed hackathon dataset is touched **only at
  inference**, on its own VM, and **never copied off**. Keep `NMR_OFFLINE=1` there.
- Do not send the closed dataset (or its rows) to any external service or agent.
  Metadata/schema-level discussion is fine; data rows are not.

---

## 8. Current research focus

### What we're trying to improve
- **Track 1 identification quality** via the SSL encoder — while keeping the
  training solvent (**D2O**) consistent with the competition matrix (blood in D2O).

### Metrics that matter (in priority order)
1. **External / held-out retrieval accuracy** for the SSL encoder — currently
   **missing**; the only retrieval number we have is in-distribution.
2. **Track 2 honesty metrics:** `honest_roc_auc`, `permutation_p_value`,
   `honest_q2`, and panel **stability** (Jaccard across folds).
3. Reconstruction **loss** — necessary but **not sufficient**; never the headline.

### Experiments to prioritize
- **Expand the open D2O corpus** (more BMRB ¹H metabolite references) → *then*
  train on the H100. This is the highest-leverage ML task.
- **Add an external retrieval eval** (query real mixtures / held-out spectra the
  encoder never saw) so SSL gains can actually be trusted.

### Experiments NOT to trust without proper validation
- Any SSL improvement measured only by **augmented/in-distribution retrieval**.
- Any AUC quoted from **`leaky_roc_auc`** or from **PCA/UMAP separation**.
- Any gain from **training longer on the current 12 spectra** (it already converges).

---

## 9. Recommended next steps

### Short-term engineering
- Add a **solvent-aware checkpoint field** (record training solvent = D2O) +
  a **Track 1 solvent selector/guard** that warns on non-D2O input.
- Add a **`.dockerignore`** (`.venv/`, `**/.git`, `*.ipynb`, `NMRformer/*.zip`)
  and use CPU-only torch to keep any image lean (~700 MB vs ~2.5 GB).
- Keep verifying the preview against odd file shapes before the real data lands.

### Short-term scientific / ML
- Build/refresh the **expanded open D2O corpus** via `build_open_corpus.py`.
- Implement a **held-out retrieval benchmark** (external, not augmented).
- Only then run the **H100 job** (`--epochs 200 --embedding-dim 128 --batch-size 256`).

### Cleanup
- Remove bundled dev cruft from any shipped image: `NMRformer/.git` (~38 MB),
  `NMRformer/metabolites_spectra.zip` (~24 MB), stray `__pycache__`.
- Fix the stale absolute paths recorded in `models/*.json` (§6.6).

### Documentation
- Keep `HANDOFF.md`, `docs/H100_TRAINING.md`, and this file in sync when the
  training flow or endpoints change.

---

## 10. Handoff summary (plain language)

RuuPhenome is a **free, transparent NMR analysis web app** for a Thai hackathon.
It does two jobs:

- **Track 1:** you give it an NMR spectrum, it tells you **which molecules are in
  the sample and how much** of each.
- **Track 2:** you give it a **table of molecule concentrations** plus a **table
  describing each sample** (e.g. sick vs healthy), and it finds **which molecules
  best tell the groups apart** — with honest statistics, not inflated numbers.

**Read these before you touch anything:**

1. **Track 2 needs no training.** It's classical machine learning that runs fresh
   on each upload in seconds. Don't "train a model" for it.
2. **The only trainable thing is an optional helper** (the SSL encoder for Track 1).
   It's trained on **12 open spectra** — that's the ceiling. Expanding the data
   matters far more than a bigger GPU. It's optional; the app works without it.
3. **HTML edits appear instantly; Python edits need a server restart** (unless you
   launched via `run.sh --reload`). If a button 404s after your change, you forgot
   to restart.
4. **Never hand-name checkpoints** — the trainer names them from the real run
   config. Hand-naming already destroyed a good checkpoint once.
5. **Never quote inflated numbers:** the retrieval score is in-distribution (not
   real accuracy), `leaky_roc_auc` is a diagnostic only, and PCA separation is not
   accuracy. Use `honest_roc_auc` + `permutation_p_value`.
6. **Data governance is strict:** train only on open data; never copy the closed
   competition dataset off its VM; keep `NMR_OFFLINE=1` there.
7. **Don't invent chemistry.** Solvent/pH provenance is "unverified" on purpose —
   surface the uncertainty, don't fabricate corrections.

To run it: `bash backend/nmr_api/run.sh` → open http://127.0.0.1:8100. To test it:
`python -m unittest discover -s backend/nmr_api/tests -p "test_*.py" -t .` from the
`ruuphenome/` root (expect 43 passing).

---

## Appendix A — Glossary

| Term | Meaning |
|---|---|
| **Track 1 / Domain 1** | Spectrum → metabolite profile (IDs + concentrations). |
| **Track 2 / Domain 2** | Concentration table + metadata → biomarkers. |
| **Table 1 / Table 2** | Organizer's inputs: (1) metabolites × samples concentrations, (2) per-sample metadata. |
| **Leakage-safe CV** | Feature selection/scaling/PCA fit **inside** each fold; the honest way to estimate performance. |
| **honest vs leaky AUC** | Honest = fit inside folds. Leaky = fit on all data first (over-optimistic; diagnostic only). |
| **SSL encoder** | Self-supervised masked-spectrum autoencoder; optional Track 1 ID enrichment; the only trainable model. |
| **Augmented / in-distribution retrieval** | Retrieval scored against the same references it trained on — not external accuracy. |
| **D2O** | Deuterium oxide, the NMR solvent for the competition (blood) matrix and the open reference corpus. |
| **`NMR_OFFLINE`** | Env var; when set, blocks all outbound network calls (closed-VM safety). |
| **NMRformer / NMRTransformer** | Optional third-party pretrained models; we use, not train, them. |

## Appendix B — Key files cheat-sheet

| I want to… | Look at |
|---|---|
| Add/'change an HTTP endpoint | `backend/nmr_api/main.py` |
| Change Track 2 discovery math | `backend/nmr_api/biomarker_engine.py`, `model_suite.py` |
| Change how tables are parsed | `backend/nmr_api/pipeline.py`, `spectral_cohort.py` |
| Change the UI | `backend/nmr_api/static/profiler.html` |
| Change SSL training | `backend/nmr_api/self_supervised.py`, `train_on_h100.py` |
| Rebuild the open corpus | `backend/nmr_api/build_open_corpus.py`, `open_data.py` |
| Run the H100 job | `docs/H100_TRAINING.md` |
