# RuuPhenome — What It Does & How It Works

*A plain-English + technical tour of the RuuPhenome ¹H-NMR metabolomics platform.*
*Scope: the served backend at `backend/nmr_api/` + the single-file UI `static/profiler.html`. Last verified against 111 passing tests.*

---

## 1. What RuuPhenome is, in one paragraph

RuuPhenome turns a **proton (¹H) NMR spectrum of a biofluid** (serum, plasma, or urine) into **(a) a list of identified metabolites with confidence levels and concentrations**, and then into **(b) a disease-relevant biomarker panel** with statistics and biological interpretation. It is built for the BDI-NPI "Phenome" hackathon and is organized around two tracks:

- **Track 1 — Identification & quantification:** *"Which molecules are in this spectrum, and how much of each?"*
- **Track 2 — Biomarker discovery:** *"Given those concentrations plus sample labels, which metabolites separate disease from control, and what do they mean biologically?"*

The whole system is deliberately **honest-by-construction**: every result carries a confidence tier, an identification-standard level, and a **Research-Use-Only (RUO)** disclaimer. It never claims clinical/diagnostic accuracy, and where a method fails it is kept as a *documented negative* rather than hidden. It is a single FastAPI backend (~48 HTTP endpoints) plus a Chenomx-style web canvas — no external services required at run time.

---

## 2. The big picture (data flow)

```
                        ┌─────────────────────────────────────────────────────┐
   Binned ¹H spectrum   │                    TRACK 1                           │
   (samples × ppm-bins) │  Annotate ──▶ Deconvolve (NNLS) ──▶ Target-decoy FDR │
   + optional:          │      │            │                      │          │
   • conditions         │  reference   overlap-resolved      keep only        │
     (solvent/pH/…)     │  matching    concentrations        FDR-confirmed     │
   • organizer pins     │  + D₂O guard  + fit R²             identifications    │
   • sample labels      │  + MSI level  + µM calibration          │            │
                        │      └── (optional) pSCNN learned channel ┘          │
                        └──────────────────────────┬──────────────────────────┘
                                                   │  FDR-confirmed concentration matrix
                        ┌──────────────────────────▼──────────────────────────┐
                        │                    TRACK 2                           │
                        │  Leakage-safe biomarker discovery (nested CV)        │
                        │  ├─ honest ROC-AUC ± CI, permutation p, stable panel │
                        │  ├─ differential abundance (BH-FDR)                   │
                        │  ├─ correlation / partial-correlation network        │
                        │  ├─ pathway enrichment (hypergeometric + BH-FDR)     │
                        │  └─ NCD relevance (Thai non-communicable diseases)   │
                        └─────────────────────────────────────────────────────┘
```

The two tracks share one engine, [`_run_cohort_pipeline()`](backend/nmr_api/main.py) in `main.py`, which normalizes → annotates → deconvolves → (if labels present) discovers biomarkers → interprets biology → scores NCD relevance.

---

## 3. Part I — What you can DO with it (capabilities)

### 3.1 Identify metabolites in a spectrum (Track 1)
Upload a **binned ¹H matrix** (CSV/TSV, samples × ppm-bins, either orientation — auto-detected). RuuPhenome returns, for every candidate metabolite:
- **Confidence tier:** ✓ FDR-confirmed · 📌 organizer pin · pSCNN (learned) · annotation-only (exploratory).
- **MSI identification level** (Metabolomics Standards Initiative): L2 "putatively annotated" is the ceiling — L1 is never claimed because it needs an authentic in-house standard.
- **D₂O reliability grade:** whether the molecule still has a usable non-exchangeable C–H signal in D₂O (OH/NH/SH protons vanish by H/D exchange).
- **Matched ppm positions** and coverage.

*Where:* UI "Upload Data" → results **Step 1 (annotation surface)**; API `POST /spectral/pipeline-file`, `POST /spectral/annotate`.

### 3.2 Quantify how much of each metabolite (Chenomx-style)
NNLS deconvolution un-mixes overlapping peaks and reports:
- **Concentration** — absolute **µM** if calibrated to an internal standard (DSS/TSP), otherwise relative.
- **Per-compound Fit R²** — how well the reconstruction explains *that molecule's own* peak region (a per-metabolite goodness-of-fit badge).
- A **compound-by-compound overlay** — the observed spectrum with each metabolite's modelled contribution drawn as a translucent colored area underneath (the Chenomx "watch each signature fit its peaks" view).
- An honest **caveats block** ("How to read these concentrations") spelling out calibration, peak model, baseline, and solvent assumptions.

*Where:* results **Step 3 (quantified metabolites)** + fit overlay; API `POST /spectral/export-concentrations` downloads the per-sample µM table as CSV with a provenance header.

### 3.3 Discover biomarkers from labels (Track 2)
Add a **Class/Group column** (or a metadata table) and RuuPhenome runs **leakage-safe** biomarker discovery:
- A **minimal stable panel** of metabolites that separates the groups.
- **Honest ROC-AUC** with a bootstrap 95% CI, a **permutation p-value**, and (binary) **Q²**.
- Per-metabolite **fold-change, direction, and single-marker AUC**.

*Where:* API `GET /biomarkers-safe`, `POST /track2/biomarkers`; UI biomarker panel.

### 3.4 Explain the biology
- **Differential abundance:** per-metabolite Mann-Whitney/Welch (2-group) or Kruskal-Wallis/ANOVA (multi-group) with **BH-FDR** q-values, effect sizes, and a volcano plot. (`POST /track2/differential`)
- **Correlation & networks:** pairwise Pearson/Spearman (FDR-filtered) plus a **Gaussian Graphical Model** partial-correlation network (direct associations conditioned on all other metabolites). (`POST /track2/correlation`)
- **Pathway enrichment:** hypergeometric over-representation test on KEGG/MetaCyc pathways, BH-FDR corrected. (`GET /biology`)
- **Metabolite cards:** curated role, disease associations, and pathway context. (`GET /enrich-names`, `GET /biology`)

*Where:* results **Step 5 (pathway enrichment)** + the "Explain biology" view.

### 3.5 Screen Thai non-communicable diseases (NCD)
`GET /ncd-screen` runs leakage-safe discovery on bundled public cohorts for five NCDs (diabetes, obesity/metabolic, cancer, cardiovascular, rheumatoid arthritis) and reports **internal cross-validated AUC + which of each disease's biomarkers are present in your sample** — explicitly labelled *internal CV only, not clinical validation*.

*Where:* results **Step 6 (NCD relevance)**.

### 3.6 Compare models & visualize
- **Model suite:** nested-CV comparison of seven models — elastic-net, linear SVM, PCA+logistic, PCA+linear SVM, random forest, HistGradientBoosting, and XGBoost (if installed) — with a complexity-penalized recommendation. (`GET /biomarkers-model-suite`)
- **Projections:** PCA + optional UMAP scores/loadings. (`GET /biomarkers-projection`)
- **Benchmark:** synthetic p≫n simulation showing honest-vs-leaky AUC inflation and biomarker recovery. (`GET /biomarkers-benchmark`)

### 3.7 Raw-data & lab workflow entry points
- **Bruker FID / processed spectrum:** upload a zipped Bruker folder or a two-column ppm/intensity CSV; RuuPhenome runs an `nmrglue`-based FFT → phase → baseline → peak-pick → assignment pipeline. (`POST /process-fid`, `POST /process-spectrum`)
- **Clinical-style profiler workflow:** QC gate → auto-profile → triage (accept/review/reject) → signed report. (`POST /profile/qc` → `/profile/auto` → `/profile/report`)
- **Reference ranges** from a healthy cohort. (`GET /reference-ranges`)

---

## 4. Part II — How it WORKS (the technical side)

### 4.1 Track 1 — identification & quantification

**Reference library.** [`spectral_cohort.REFERENCE_SHIFTS`](backend/nmr_api/spectral_cohort.py) is a curated dict of **578 metabolites** → characteristic ¹H shift centers (HMDB 5.0 + BMRB-merged). Honest limitation: HMDB does not record per-compound solvent/pH/temperature, so `solvent_confidence()` always returns `"unverified"` — shifts are treated as biofluid-conventional (aqueous/D₂O), never guaranteed.

**Targeted profiling (`annotate`).** For each metabolite it finds ppm-bins within tolerance of its reference shifts, but only counts a bin as "occupied" if the cohort-median intensity clears a **robust noise floor** — `baseline + 5 × 1.4826 × MAD` (median absolute deviation). This replaced an earlier broken 0.75-quantile rule that called ~all 578 metabolites on pure noise. A metabolite is called *present* if ≥50% of its shifts match **and** it retains a non-exchangeable resonance under the D₂O guard. Organizer pins bypass the occupancy gate and force presence.

**D₂O exchangeable-proton guard.** [`identification_quality.py`](backend/nmr_api/identification_quality.py) classifies each matched shift as water/HDO (4.70–4.90 ppm), downfield-exchangeable (>9.5 ppm), or non-exchangeable C–H. It also parses each metabolite's SMILES (offline PubChem cache, regex parser — no RDKit) into exchangeable (O/N/S–H) vs non-exchangeable (C–H) proton counts, then grades the molecule `reliable / caution / weak / invisible` for D₂O. The guard is **condition-aware**: aqueous/D₂O/unknown solvents apply it; the seven supported deuterated organic solvents (DMSO-d6, CDCl₃, CD₃OD, …) disable it.

**MSI level.** [`identification_quality.msi_level()`](backend/nmr_api/identification_quality.py) assigns L2 (putatively annotated, library-matched, ≥2 non-exchangeable resonances + ≥0.5 coverage), L3 (single resonance), or L4 (no evidence). **L1 is gated off** — no authentic-standard spike-in is run.

**NNLS deconvolution + target-decoy FDR (`deconvolve`).** Builds an (n_metabolites × n_bins) reference matrix of unit-area Gaussians (σ = 0.012 ppm), then per sample solves non-negative least squares `min ‖X − Rᵀc‖², c ≥ 0` with [`scipy.optimize.nnls`](backend/nmr_api/spectral_cohort.py). False positives are controlled by a **target-decoy** scheme (à la proteomics, Elias & Gygi 2007): every reference is duplicated at a ppm offset (decoy), NNLS is re-run, and a metabolite is accepted only where the decoy-derived FDR ≤ 0.05. Outputs per metabolite: concentration, `passes_fdr`, SNR-vs-decoy, and a **local Fit R²** over that compound's own peak support.

**Absolute quantification.** If an internal standard (DSS/TSP) is detected, a single linear calibration factor `standard_µM / fitted_intensity` scales all concentrations to µM. Assumes known spike-in, flat baseline, linear response — marked RUO, "not Chenomx-grade absolute."

**Honest peak model.** The reference peaks are **single Gaussians (singlets), not J-coupling multiplet lineshapes**. The UI states this everywhere; a low overall fit R² (<0.3) raises a warning that concentrations are *directional, not exact*.

**pSCNN learned channel (optional).** [`pscnn.py`](backend/nmr_api/pscnn.py) is a **pseudo-Siamese 1-D CNN**: two weight-independent 3-layer conv towers embed a (reference, sample) pair; a matching head over `[ref, sample, ref·sample, |ref−sample|]` outputs a present/absent logit. Trained on synthetic superposed mixtures of an open **30-compound panel** with ±0.012 ppm drift augmentation to absorb pH/temperature/referencing drift. At serve time it blends into the deterministic core as a **hybrid**: `present = FDR-confirmed ∪ pSCNN(prob ≥ 0.6)`. It activates only if `models/pscnn_identifier.pt` exists; otherwise `pscnn.status()` reports a non-silent degrade to deterministic-only.

**Physiological whitelist gating.** When the sample matrix is known (serum/urine), [`panel_reference_shifts()`](backend/nmr_api/spectral_cohort.py) restricts the 578-metabolite library to a curated biofluid panel — **37 serum** metabolites (Psychogios et al. 2011) or **32 urine** (Bouatra et al. 2013, resolved against the reference library) — which sharply cuts over-annotation and lets FDR confirm real biofluid compounds.

**Feature selection (`select_diagnostic_ppm`).** Ranks ppm positions by disease association (supervised, leakage-safe univariate screen) or signal content (unsupervised), excludes the water window, de-duplicates clusters, and annotates each to its nearest metabolite + NCD relevance.

### 4.2 Track 2 — biomarker discovery & analytics

The engine is [`biomarker_engine.discover()`](backend/nmr_api/biomarker_engine.py). Its whole design goal is **avoiding the p≫n self-deception** that plagues metabolomics ML:

- **Leakage-safe nested CV:** feature selection (variance filter → univariate screen → BH-FDR → top-k) happens **inside each training fold only**; test folds never see selection statistics (Ambroise & McLachlan 2002). Patient-grouped `StratifiedGroupKFold` prevents the same patient landing in train and test.
- **Univariate screen:** point-biserial correlation + t-test (binary) or ANOVA F-test (`f_classif`, multi-class), in O(n·p) without a p×p covariance matrix.
- **Honest metrics:** pooled out-of-fold ROC-AUC (macro one-vs-rest for multi-class), **Q²** (1 − PRESS/TSS), F1, and a **percentile bootstrap 95% CI** resampled at the patient level.
- **Permutation p-value:** shuffle labels ~100×, rerun the entire CV, report `(1 + #{perm AUC ≥ real}) / (n_perm + 1)`.
- **Stability:** mean pairwise Jaccard of per-fold selected features; the "stable panel" = features chosen in ≥50% of folds.
- **Leaky-AUC exhibit:** the same run with selection on the *full* data, to quantify the inflation leakage would cause.
- **PLS-DA VIP** ranking for the stable panel (binary).

**Differential abundance** ([`differential.py`](backend/nmr_api/differential.py)): whole-cohort, descriptive (not cross-validated) — group means, log₂ fold-change, Mann-Whitney/Welch or Kruskal-Wallis/ANOVA, Cohen's d / η², BH-FDR q-values, volcano array.

**Correlation** ([`correlation.py`](backend/nmr_api/correlation.py)): Pearson/Spearman matrix (Fisher-z p-values, BH-FDR, |r|≥0.3) **plus** a Gaussian Graphical Model — Ledoit-Wolf shrinkage covariance → precision matrix → partial correlations (Krumsiek et al. 2011), capped at 80 metabolites.

**Model suite** ([`model_suite.py`](backend/nmr_api/model_suite.py)): outer patient-grouped CV with an **inner tuning loop** per fold over seven models — elastic-net, linear SVM, PCA+logistic, PCA+linear SVM, random forest, HistGradientBoosting, and XGBoost (if installed). Reports AUC/F1/Brier/ECE and recommends the simplest model within 0.01 AUC of the best.

**Biology** ([`biology.py`](backend/nmr_api/biology.py)): hypergeometric pathway over-representation (BH-FDR) on curated KEGG/MetaCyc sets + per-metabolite biology cards.

### 4.3 The API & orchestration
[`main.py`](backend/nmr_api/main.py) is a FastAPI app with **~48 endpoints**, CORS-open, serving `profiler.html` at `/` with `Cache-Control: no-store`. Key shared machinery:
- **`DATASETS`** registry — six bundled public MetaboLights cohorts (MTBLS1, 242, 356, 424, 6213, 147) with kind/task/source.
- **`NCD_PANEL`** registry — five Thai NCDs → cohort + burden narrative, cached per-run in `_NCD_CACHE`.
- **`_run_cohort_pipeline()`** — the Track-1→Track-2 spine used by every `/spectral/*` endpoint; it also decides that **only FDR-confirmed concentrations feed Track 2** (falling back to raw abundance with a low-confidence flag if nothing passes).
- **`_identification_channels()`** — assembles the deterministic ∪ pSCNN hybrid and reports the pSCNN status.
- **`_PROFILE_STATE`** — stateful QC → auto-profile → report workflow.

### 4.4 The UI
`static/profiler.html` (~3,000 lines, single file) is a **Chenomx-style canvas**:
- A zoomable/pannable spectrum canvas rendering the measured trace, a selected compound's Lorentzian fit, and pinnable color-coded overlays.
- A **6-step results report**: (1) annotation surface with confirmed-vs-candidate tiers, (2) identification method + pSCNN status, (3) quantified metabolites + compound-by-compound fit overlay + caveats, (4) diagnostic ppm, (5) pathway enrichment (BH-FDR), (6) NCD relevance.
- A **reactive conditions form** (`_condReact`) where choosing a solvent/sample type auto-suggests the reference standard (DSS/TSP aqueous, TMS organic), excluded regions (water 4.70–4.90; urine urea 5.5–6.1; serum lipid bands), and the D₂O-guard state — grounded in IUPAC/Emwas 2018 conventions.
- A **searchable/filterable metabolite browser** (by name, FDR-confirmed only, D₂O-reliable only, organizer pins only).

---

## 5. Models & training

| Model | Purpose | Trained on | Status |
|---|---|---|---|
| **Deterministic core** (NNLS + target-decoy FDR) | Track-1 identification & quantification | — (algorithmic, no learning) | **Production, always on** |
| **pSCNN** pseudo-Siamese CNN | Drift-robust identification (hybrid recall booster) | Synthetic mixtures of a 30-compound open panel + ppm-drift augmentation | **Production if checkpoint present**, else silent degrade |
| **Masked-spectrum autoencoder** (SSL) | Self-supervised representation pretraining | Open BMRB ¹H corpus (masked-reconstruction) | Trainable via `train_on_h100.py` |
| **GISSMO quantifier** (F8) | Learned concentration regression | GISSMO-simulated mixtures | **Documented honest-negative — disabled, not wired in** |

Training runs on an **H100/LiCO** node via `python -m nmr_api.train_on_h100` (CUDA → MPS → CPU auto-detect, `bf16` autocast on CUDA). Everything is **seeded** (`torch.manual_seed` + `cuda.manual_seed_all` + `np.random.seed` + `random.seed`, default 2026) and checkpoints are **config-tagged** by device+hyperparameters so runs don't silently overwrite. `train_on_h100.py` was fixed to force CUDA and auto-label checkpoints.

### Honest performance (held-out, RUO)
Track-1 methods are benchmarked ([`track1_benchmark.py`](backend/nmr_api/track1_benchmark.py)) three ways, and the numbers are reported with their honesty context:

- **In-distribution synthetic** (library == matcher → optimistic): NNLS+FDR F1 ≈ **0.90**. *This is not real-world accuracy.*
- **BMRB experimental held-out** (real measured peak lists — real intensities & multiplet structure, DSS-referenced — the strongest available check without patient data): annotate F1 ≈ 0.64, deconvolve+FDR ≈ 0.24, pSCNN ≈ 0.42, **hybrid ≈ 0.53 at ~0.70 precision** — the best precision-respecting method.
- **GISSMO real-shift held-out:** deterministic collapses (~0.16), hybrid best (~0.40).

The story these tell: permissive annotation over-calls, strict FDR under-calls, and the **hybrid is the honest winner** on real data. The GISSMO quantifier (F8) scored F1 0.13 on the same real held-out and is therefore kept off as a "what not to ship" exhibit.

---

## 6. Data sources & governance

RuuPhenome draws on three tiers of data, kept strictly separate:

1. **Closed competition data (LiCO / H100 VM, train-only):** the organizer's `nmr-pattern/Human_Serum/…` and urine cohorts. Loaded by [`lico_loader.py`](backend/nmr_api/lico_loader.py) with a **positional Var↔row join** (loud on mismatch) and QC/pool/dilution exclusion. Governance rule: this data **never leaves the VM** — `run_track1()` returns only safe aggregate metrics (counts, fit R², confidence), never raw spectra or per-sample values. `NMR_OFFLINE=1` blocks all outbound calls.
2. **Public MetaboLights cohorts (external test):** MTBLS6213 (serum, rheumatoid arthritis), MTBLS1, MTBLS694, etc. [`mtbls_adapter.py`](backend/nmr_api/mtbls_adapter.py) parses ISA-Tab sample sheets + raw Bruker `pdata/1/1r` (via `nmrglue`) and buckets to a common 0.005-ppm grid, so the *same* loader/pipeline runs on real serum. Used as the genuine external sanity check.
3. **Open experimental reference (held-out validation):** real **BMRB** peak lists (`bmrb_experimental.py`, 18 compounds) and **GISSMO** spin-system shifts (`external_reference.py` / `gissmo_corpus.py`, ~94 compounds), bundled under `open_data/` so no network is needed at test time.

**Structural honesty caveat:** there is no open, real *binned* cohort wired directly into Track-1 — the bundled `demo_mtbls*.tsv` files are MetaboLights **MAF concentration tables** (used by Track 2), and demo Track-1 runs use synthetic `make_demo_binned()` cohorts. The strongest real evidence is the BMRB experimental held-out above.

---

## 7. Honesty & limitations (read this before trusting any number)

- **RUO everywhere.** All AUCs are *internal cross-validated discrimination* on single historical cohorts — an honest estimate of same-population performance, **not** clinical/diagnostic accuracy. External prospective validation is required for any clinical use.
- **Identification ≤ MSI L2.** Library-matched putative IDs, never authentic-standard-confirmed (L1), no orthogonal J-coupling confirmation.
- **Concentrations are directional.** Single-Gaussian peak model (no J-multiplet lineshapes), flat-baseline assumption, single-point internal-standard calibration. Not Chenomx-grade absolute quant.
- **Reference shifts are solvent/pH-unverified.** No curated pH/field-aware library; the D₂O guard is a generic rule, not per-metabolite validated.
- **Synthetic ≠ real.** In-distribution benchmarks are optimistic by construction; trust the held-out BMRB numbers.
- **Pathway enrichment is descriptive**, not causal. **NCD screening is a demo** on public cohorts.
- **Deliberately *not* built** (to avoid overclaiming): J-multiplet fitting, a pH/field-aware line-shape library, absolute-quant guarantees, and the F8 quantifier (kept as an honest negative).

---

## 8. Reproducibility & testing

- **111 tests pass** across 16 files (`backend/nmr_api/tests/`), covering signal processing, annotate/deconvolve, pSCNN train/inference, self-supervised corpus, Track-1 benchmarks, model-suite nested CV, provenance, and integration pipelines.
- **Deterministic training** via explicit seeding; **config-tagged checkpoints** prevent silent overwrites.
- **`requirements.lock.txt`** (Python 3.13) pins exact versions; `requirements.txt` gives flexible minimums.
- **Offline-capable:** all reference/experimental data is bundled under `open_data/`; `NMR_OFFLINE=1` guarantees no outbound network.

---

## 9. Quick file map

| Area | File | What it holds |
|---|---|---|
| API & orchestration | `backend/nmr_api/main.py` | ~48 endpoints, dataset/NCD registries, `_run_cohort_pipeline` |
| Track-1 core | `backend/nmr_api/spectral_cohort.py` | `REFERENCE_SHIFTS`, `annotate`, `deconvolve` (NNLS+FDR), panels, demo cohort |
| ID quality | `backend/nmr_api/identification_quality.py` | MSI levels, D₂O guard, SMILES exchangeable-proton parsing |
| Learned channel | `backend/nmr_api/pscnn.py` | pseudo-Siamese CNN, hybrid identification |
| Honest-negative | `backend/nmr_api/quantifier.py` | GISSMO transformer (disabled, F1 0.13) |
| Biomarkers | `backend/nmr_api/biomarker_engine.py` | leakage-safe nested-CV discovery |
| Analytics | `backend/nmr_api/differential.py`, `correlation.py`, `model_suite.py`, `biology.py` | differential, networks, model suite, pathways |
| Provenance | `backend/nmr_api/provenance.py` | condition-aware D₂O gating + warnings |
| Data ingestion | `backend/nmr_api/lico_loader.py`, `mtbls_adapter.py` | closed LiCO (train-only) + public MTBLS adapter |
| Held-out refs | `backend/nmr_api/bmrb_experimental.py`, `external_reference.py`, `gissmo_corpus.py` | real BMRB/GISSMO validation data |
| Training | `backend/nmr_api/train_on_h100.py`, `track1_benchmark.py` | H100 training + benchmarks |
| UI | `backend/nmr_api/static/profiler.html` | single-file Chenomx-style canvas + 6-step report |

---

*Generated from a full read of the served backend and UI. Figures (578 library metabolites, 37 serum / 32 urine panels, 30-compound pSCNN panel, ~48 endpoints, 111 tests) were verified against the source at documentation time.*
