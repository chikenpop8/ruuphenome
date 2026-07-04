# Track 1 — Required Functions (mapped to the spec, not to code modules)

> The BDI Phenome **Track 1** problem statement asks for five capabilities, not a
> list of files. This document is organised by those five required functions and,
> for each, says what it means, its data flow, how our code supports it, what is
> built vs missing, its rubric value, and the honest caveats.

**Track 1 data context (from the spec):** input is *cleaned* ¹H-NMR spectrum data —
**~20,000+ intensity × ppm features per sample**, delivered as **TSV/CSV** — with
**some compound annotations provided for training**. The goal is to identify
**compound type and position** from the spectrum; the core challenge is
**overlapping/superimposed signals**.

> **Standing honesty line (applies to every section):** our strong identification
> numbers are on *synthetic, in-distribution* data. On an **independent, real,
> held-out** benchmark — mixtures rendered from **BMRB metabolomics *experimental* ¹H
> peak lists** (real measured positions + intensities + multiplet structure; 18
> compounds incl. the full glucose/lactate/BCAA panel) — the honest picture is:
> permissive matching finds everything but over-calls (F1 ≈ 0.64 at precision ≈ 0.49),
> strict FDR matching is precise but misses most on real spectra (F1 ≈ 0.24), and the
> **learned + hybrid channel gives the best precision/recall balance among trustworthy
> methods (F1 ≈ 0.53 at precision ≈ 0.70) and is far more robust to a referencing
> offset** than strict deterministic. No method is clinical-grade; this is
> **research-use-only**. Reproduce: `python -m nmr_api.track1_benchmark --validate-bmrb`.
> See `docs/TRACK1_PLAN.md §9`.

> **About the "Correctness in bioinformatics & chemistry" blocks below:** each function
> carries citations to the peer-reviewed literature / reference databases that justify its
> chemistry and its bioinformatics/statistics. Every link was gathered by a researcher and then
> **independently re-fetched by a separate checker** to confirm it resolves and actually supports
> the claim (fabricated or mislabeled citations were dropped); the URLs were finally re-checked
> mechanically for a live HTTP response. Links prefer stable open hosts (PubMed / PMC / official
> database pages); a couple resolve to a publisher DOI that opens in a browser.

---

## 1. Compound Classification

**What it means here.** Decide *which* metabolites are present in a spectrum (and
*where* — which ppm positions) — e.g. "glucose is present, anomeric H1 at ~5.23 ppm."

**Input data.** A samples × ppm-bin matrix (`~20,000` bins) of intensities; optional
organizer-provided `{ppm: metabolite}` annotations.

**What the software does.**
- **Deterministic matching** — `spectral_cohort.annotate()`: a metabolite is called
  present when enough of its characteristic ¹H shifts fall in occupied bins
  (coverage ≥ 0.5). Transparent — every call lists which shifts matched. This is the
  explainable core (the open equivalent of Chenomx/ASICS).
- **pSCNN supervised classifier** — `pscnn.py`: a pseudo-Siamese 1D-CNN that decides
  present/absent per compound from a (reference, sample) pair; an **added evidence
  channel**, robust to ppm drift/noise where fixed-tolerance matching fails.
- **Hybrid evidence** — the deterministic call and the pSCNN probability are combined
  (the practical winner); the blending contract lives in
  `nmrformer_backend.hybridize()`.
- **MSI confidence level** — `identification_quality.msi_level()`: every call is tagged
  **Metabolomics Standards Initiative Level 2** ("putatively annotated" by library
  similarity) — **never Level 1**, which requires an authentic in-house standard.
- **Organizer-provided annotations** — treated as **authoritative pins**
  (`annotate()`): a provided `{ppm: metabolite}` bypasses the occupancy gate and is
  tagged `provenance:"organizer_pin"` (previously such labels were silently dropped).

**Output.** Per metabolite: `metabolite`, `matched_shifts` (positions), `coverage`,
`robust_coverage`, `msi_level`, a `d2o` reliability block, `provenance`, and
`mean_abundance`.

**Used next.** Feeds the sample × metabolite table for **Biomarker Discovery** (§4)
and the automated workflow (§5).

**Current code / modules.** `spectral_cohort.annotate` · `pscnn.py` ·
`identification_quality.py` · `nmrformer_backend.hybridize`.

**Implemented.** Deterministic matching, pSCNN classifier (optional), hybrid contract,
MSI tagging, D2O guard, authoritative pins. Benchmarked (`track1_benchmark.py`):
deterministic **F1 0.90** on clean synthetic, pSCNN better under drift, **hybrid best
on real GISSMO shifts (0.40)**.

**Missing / to improve.** The pSCNN is trained on a modest panel and on library-derived
patterns; the real-data ceiling is still low. Fine-tuning on the **provided competition
annotations** (on-VM only) is designed but not yet wired (F3). No authentic-standard
confirmation (so never MSI Level 1).

**Rubric value.** Technical (a learned classifier the spec asks for) · Bioinformatics
correctness (MSI levels, D2O guard, honest benchmark) · Innovation (hybrid deterministic
+ learned) · Reproducibility (`--compare`/`--validate-real` reproduce the numbers).

**Correctness in bioinformatics & chemistry** *(every link independently re-fetched and
verified to resolve and support the claim).*
- *Chemistry —* the reference ¹H shifts we match against are real, citable libraries:
  **HMDB 5.0** generated 312,980 predicted ¹H/¹³C NMR spectra for compound identification
  ([Wishart et al. 2022, *Nucleic Acids Res*](https://pmc.ncbi.nlm.nih.gov/articles/PMC8728138/));
  **BMRB** holds experimental reference spectra/shifts for 1,200+ metabolites
  ([Hoch et al. 2023, *Nucleic Acids Res*](https://pmc.ncbi.nlm.nih.gov/articles/PMC9825541/));
  **GISSMO** spin-system matrices simulate metabolite spectra across conditions
  ([Dashti et al. 2017, *Anal Chem*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5705194/)).
- *Chemistry —* our **0-ppm reference** convention (TMS for organics, **DSS for aqueous**)
  is the IUPAC recommendation
  ([Harris et al. 2008, IUPAC, *Magn Reson Chem*](https://pubmed.ncbi.nlm.nih.gov/18407566/)),
  and the **D2O/exchangeable-proton guard** rests on the well-known "D2O shake": labile OH/NH
  protons swap for deuterium and vanish
  ([U. Ottawa NMR Facility](http://u-of-o-nmr-facility.blogspot.com/2007/10/proton-nmr-assignment-tools-d2o-shake.html)).
- *Bioinformatics —* reporting **MSI Level 2 (putative), never Level 1 without an authentic
  standard**, is exactly the CAWG/MSI identification scheme
  ([Sumner et al. 2007, *Metabolomics* (MSI)](https://pmc.ncbi.nlm.nih.gov/articles/PMC3772505/)).

**Caveats.** Real held-out F1 is modest; synthetic numbers are optimistic; MSI Level 2,
never 1; do not present any single number as accuracy without the in-distribution caveat.

---

## 2. Pattern Recognition / Machine Learning

**What it means here.** Use ML to recognise compound patterns automatically —
especially to disentangle **overlapping signals** that defeat fixed-tolerance rules.

**Input data.** Open/simulated ¹H spectra + (on-VM only) the provided competition
annotations.

**What the software does.**
- **pSCNN classifier** (`pscnn.py`) — learns soft present/absent matching, trained on
  superposed reference-standard mixtures with **ppm-drift augmentation**.
- **pSCNN is now wired into the app** (`main._run_cohort_pipeline`): if
  `models/pscnn_identifier.pt` is present it loads once and blends with the deterministic
  calls → the **hybrid identification** (the honest Track-1 winner) is reachable from the
  product, not just the benchmark. Trained/persisted via `train_on_h100.py --supervised pscnn`.
- **GISSMO transformer quantifier** (`quantifier.py`) — an **honest-negative exhibit**, not a
  hero: trained on an H100 and evaluated on real BMRB held-out spectra, it scored **F1 0.13
  (worst method)** and did NOT close the sim-to-real gap (fixed-width GISSMO fingerprints, no
  J-multiplets). Kept documented + disabled; NOT wired into the app.
- **Open / simulated training data** (`gissmo_corpus.py`) — a bundled 94-compound GISSMO
  corpus + an on-the-fly mixture simulator with concentration labels; **unlimited
  training data from open sources**.
- **Offline-VM rule for provided annotations** — pretrain on open/simulated data
  **off-VM**; use the **provided competition annotations only for local fine-tuning
  inside the governed offline VM** (`NMR_OFFLINE=1`); **never export the closed data**.
- **H100 training role** — `train_on_h100.py --supervised gissmo-quant` runs the F8
  quantifier at scale (justified by ~250k-mixture data-generation + sweeps, not model
  size). Exact command + 7-point run spec in `docs/TRACK1_PLAN.md §9`.

**Output.** Trained checkpoints (`models/pscnn_identifier.pt`,
`models/gissmo_quantifier.pt`) + present/absent probabilities and relative concentrations.

**Used next.** The ML probabilities/concentrations become the AI evidence channel in
Compound Classification (§1) and the quantities in Biomarker Discovery (§4).

**Current code / modules.** `pscnn.py` · `quantifier.py` · `gissmo_corpus.py` ·
`train_on_h100.py` · optional `self_supervised.py` (SSL encoder — supporting evidence only).

**Implemented.** pSCNN (built, tested, **persisted checkpoint + wired into the app** as the
hybrid channel), the GISSMO corpus + simulator, the H100 training modes (`--supervised
pscnn` and `--supervised gissmo-quant`), and the governance split. The GISSMO quantifier was
**trained + evaluated on real held-out data and failed** (F1 0.13) — retained as an
honest-negative.

**Missing / to improve.** The GISSMO quantifier did NOT generalize (sim-to-real gap);
**not scaled** — realistic lineshapes (J-multiplets) would be the only thing worth trying and
only with reason to believe. F3 (on-VM fine-tune loader for the provided competition
annotations) is the highest-value remaining ML work. SSL retrieval must never be quoted as
accuracy.

**Rubric value.** Innovation (physics-simulated GISSMO quantifier; learned overlap
resolution) · Technical (transformer + CNN, H100-ready) · Reproducibility (config-tagged
checkpoints, bundled corpus, one command).

**Correctness in bioinformatics & chemistry** *(links independently verified).*
- *Chemistry —* the training data is **physically/quantum-mechanically accurate**: GISSMO
  optimizes spin-system matrices against experimental 1D-¹H spectra to simulate metabolite
  spectra ([Dashti et al. 2017, *Anal Chem*](https://pmc.ncbi.nlm.nih.gov/articles/PMC5705194/)),
  a library expanded to >1,100 compounds at many fields
  ([Dashti et al. 2018, *Anal Chem*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6201686/);
  [GISSMO library](https://gissmo.bmrb.io/)).
- *Chemistry —* the **ppm-drift augmentation** is grounded in measured physics: ¹H shifts move
  with pH, temperature, ionic strength and composition
  ([Bhinderwala et al. 2022, *J Magn Reson*](https://pmc.ncbi.nlm.nih.gov/articles/PMC9742302/);
  acid/base + metal-ion shift limits, [Tredwell et al. 2016, *Metabolomics*](https://pubmed.ncbi.nlm.nih.gov/27729829/)).
- *Bioinformatics —* deep networks are an **established, benchmarked** way to identify/quantify
  metabolites from ¹H NMR: a CNN+GRU does it on real human plasma
  ([Wang et al. 2023, NMRQNet](https://pmc.ncbi.nlm.nih.gov/articles/PMC10002723/)), and
  MLP/CNN/transformer quantifiers are actively benchmarked with data-augmentation shown to
  matter ([Johnson & Tipirneni-Sajja 2025, *Metabolites*](https://pmc.ncbi.nlm.nih.gov/articles/PMC12029129/)).

**Caveats.** ML numbers are in-distribution until proven on a real source the model did
**not** train on (GISSMO is in-distribution for the quantifier — its fair test is
BMRB-experimental / MTBLS1). Keep every learned model **optional** (no checkpoint → off).

---

## 3. Feature Selection

**What it means here.** From `~20,000` anonymous ppm features, select the **most
important positions** — dimensionality reduction that is also **biologically
interpretable**.

**Input data.** The binned intensity × ppm matrix; optional sample labels (for
supervised selection).

**What the software does.** `spectral_cohort.select_diagnostic_ppm()`:
- **Supervised mode** (labels given) — rank ppm positions by leakage-safe association
  with the outcome (point-biserial / ANOVA-F).
- **Unsupervised mode** — rank by signal content (intensity × variance).
- Each selected position is **annotated to its nearest reference metabolite** and its
  **NCD relevance** (via `NCD_RELEVANCE`): **glucose ~5.23** (hyperglycemia/diabetes),
  **lactate ~1.33** (glycolysis/insulin resistance), **BCAA methyls ~0.9–1.05**
  (valine/leucine/isoleucine → incident T2D), **aromatic amino acids** tyrosine/
  phenylalanine (T2D/CVD). The residual-water/HDO window is excluded.

**Output.** A ranked shortlist `{ppm, importance, nearest_metabolite, ppm_error,
ncd_relevance}` + an `ncd_relevant_positions` subset.

**Used next.** Turns 20k features into an interpretable, biomarker-ready set that anchors
the Track-1 → Track-2 story and reduces the p≫n burden.

**Current code / modules.** `spectral_cohort.select_diagnostic_ppm`, `NCD_RELEVANCE`,
`_nearest_reference`, `_ncd_relevance`.

**Implemented.** Both modes, NCD annotation, water-region exclusion, de-duplication.

**Missing / to improve.** Not yet embedded/wrapper (RFE) selection; not yet surfaced in
the UI; could feed the selected positions directly into the pSCNN as an attention prior.

**Why the selected ppm regions matter biologically.** They are the exact resonances the
diabetes/insulin-resistance literature uses — selecting them makes the model both smaller
*and* explainable to a clinician, and ties Track 1 directly to the Thai-NCD screening
goal (see `docs/IMPACT_AND_VALIDATION.md §2`).

**Rubric value.** Biological interpretation (metabolite/NCD-anchored positions) · Impact
(a small, cheap, deployable panel) · Innovation (learned saliency ↔ interpretable ppm) ·
Technical (dimensionality reduction for p≫n).

**Correctness in bioinformatics & chemistry** *(links independently verified).*
- *Chemistry (assignment) —* the ppm anchors we select on are the canonical biofluid
  assignments: **glucose ~5.23 ppm, lactate methyl ~1.33 ppm** and **BCAA methyls ~0.94–1.05
  ppm** are tabulated for serum/plasma
  ([Kaluarachchi et al. 2018, *Metabolomics*](https://pmc.ncbi.nlm.nih.gov/articles/PMC7122646/)),
  and the lactate doublet (~1.34) + glucose α-anomer H1′ (~5.23) are landmark IDs in the
  standard identification guide
  ([Dona et al. 2016, *Comput Struct Biotechnol J*](https://pmc.ncbi.nlm.nih.gov/articles/PMC4821453/)).
- *Biology —* those choices are **NCD-meaningful**: branched-chain + aromatic amino acids
  predict incident type-2 diabetes (>5-fold top-quartile risk)
  ([Wang et al. 2011, *Nat Med*](https://pubmed.ncbi.nlm.nih.gov/21423183/)) and a BCAA
  signature is mechanistically tied to insulin resistance
  ([Newgard et al. 2009, *Cell Metab*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3640280/)).
- *Bioinformatics —* doing selection **inside** the CV folds is required: whole-dataset
  selection gives optimistically biased error in p≫n omics
  ([Ambroise & McLachlan 2002, *PNAS*](https://pmc.ncbi.nlm.nih.gov/articles/PMC124442/)),
  and principled feature selection is standard for metabolomics biomarker discovery
  ([Grissa et al. 2016, *Front Mol Biosci*](https://www.frontiersin.org/journals/molecular-biosciences/articles/10.3389/fmolb.2016.00030/full)).

**Caveats.** Selection must stay leakage-safe (inside CV folds for supervised use);
"important" ≠ "causal"; naive PCA on sparse NMR bins is avoided on purpose.

---

## 4. Biomarker Discovery

**What it means here.** Track 1's job for biomarker discovery is to produce a **reliable,
quantified compound table** that Track 2 can mine — i.e. it is the *front half* of the
pipeline; Track 2 does the actual discovery statistics.

**Input data.** The identified compounds (§1) + their quantities.

**What the software does.**
- **Compound identity** — from Compound Classification (§1), with `msi_level` +
  `provenance`.
- **Relative abundance / concentration** — `spectral_cohort.deconvolve()` resolves
  overlapping peaks by NNLS against reference spectra with a **target-decoy FDR** null
  (and optional DSS/TSP µM calibration); the GISSMO quantifier (§2) gives a learned
  relative concentration.
- **Confidence fields** — `coverage`, `robust_coverage`, `msi_level`, `passes_fdr`,
  `snr_vs_decoy`, `d2o` reliability — so every quantity carries its trust level.

**Output.** A **sample × metabolite** table (`annotated_matrix` / `concentrations`) with
per-metabolite confidence.

**Used next — this is the Track-1 → Track-2 bridge** (`main._run_cohort_pipeline`): the
table feeds Track-2's already-strong engine —
- **Biomarker Discovery** (`biomarker_engine.discover`, leakage-safe nested CV + bootstrap
  95% CI + permutation p),
- **Predictive Model** (`model_suite.compare_models`),
- **Differential Analysis** (`differential.py`, MWU/Kruskal + BH-FDR + volcano),
- **Correlation Analysis** (`correlation.py`, pairwise + partial-correlation GGM network),
- **Biological interpretation** (`biology.interpret_panel`, pathway enrichment).

**Current code / modules.** `spectral_cohort.annotate`/`deconvolve` → `main._run_cohort_pipeline`
→ `biomarker_engine` / `model_suite` / `differential` / `correlation` / `biology`.

**Implemented.** The full bridge and the Track-2 analytics (see `docs/TRACK2_FUNCTIONS.md`,
`docs/IMPACT_AND_VALIDATION.md`).

**Missing / to improve.** Track-1 quantities are relative (µM only with an internal
standard); identification quality on real data (§1) bounds the reliability of the table.

**Rubric value.** Impact + Biological interpretation (an interpretable biomarker panel
→ NCD screening) · Bioinformatics correctness (confidence fields carried through;
leakage-safe downstream) · Reproducibility (one automated Track-1 → Track-2 path).

**Correctness in bioinformatics & chemistry** *(links independently verified).*
- *Chemistry —* resolving overlapping resonances by **linear-combination / NNLS fitting against
  a pure-compound library** is an established quantification strategy: **ASICS** auto-identifies
  and quantifies metabolites this way with improved precision
  ([Lefort et al. 2019, *Bioinformatics*](https://pubmed.ncbi.nlm.nih.gov/30977816/)), and
  joint library-based identification+quantification is statistically validated
  ([Zheng et al. 2011, BQuant, *Bioinformatics*](https://pmc.ncbi.nlm.nih.gov/articles/PMC3106181/)).
- *Bioinformatics —* the discovery statistics are textbook-correct: **Benjamini–Hochberg FDR**
  for multiple metabolites
  ([Benjamini & Hochberg 1995, *JRSS-B*](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x)),
  a **target-decoy null** to gate identifications empirically
  ([Elias & Gygi 2007, *Nat Methods*](https://pubmed.ncbi.nlm.nih.gov/17327847/)), and
  **leakage-safe nested cross-validation** to avoid selection-bias over-optimism
  ([Ambroise & McLachlan 2002, *PNAS*](https://pmc.ncbi.nlm.nih.gov/articles/PMC124442/), see §3).

**Caveats.** Garbage-in/garbage-out: a weak Track-1 identification propagates into Track-2;
the honest metrics (honest AUC + CI + permutation p) live in Track 2 — never quote a
Track-1 identification number as a disease-classification result.

---

## 5. Automated Workflow Development

**What it means here.** One automated pipeline from an uploaded cleaned spectrum to a
Track-2-ready table — no manual peak-picking, no closed software.

**The end-to-end flow (what the backend does).**
1. **Upload** a cleaned TSV/CSV spectrum (samples × ppm-bin) — `POST /spectral/pipeline-file`.
2. **Parse** the `~20,000`-feature intensity × ppm matrix — `spectral_cohort.load_binned_matrix`
   (auto-detects orientation; no fixed bin count).
3. **Normalise** — `pqn_normalize` (Probabilistic Quotient Normalization).
4. **Annotate compounds** — `annotate()` (deterministic matching) + optional
   organizer pins.
5. **Apply MSI + D2O guard** — `identification_quality` (Level 2 tagging; exclude
   residual-water/HDO + exchangeable protons; require a non-exchangeable C-H resonance).
6. **Resolve overlapping peaks** — `deconvolve()` (NNLS linear-combination fit + target-
   decoy FDR); **F7 scaling** keeps a 20k-bin matrix tractable (~11 s vs 188 s).
7. **Run deterministic + AI evidence** — optionally blend the pSCNN / GISSMO-quantifier
   channel (hybrid) when a checkpoint is present.
8. **Select diagnostic ppm** — `select_diagnostic_ppm` (interpretable, NCD-anchored).
9. **Quantify compounds** — relative concentrations (+ µM with an internal standard).
10. **Export the sample × metabolite table** — CSV / `annotated_matrix`.
11. **Send to Track 2** — the same call runs biomarker discovery + biology if labels exist.

**Input / Output.** Input: cleaned TSV/CSV spectrum. Output: identified-compound table +
concentrations + confidence + (optional) Track-2 results.

**Current code / modules.** `main._run_cohort_pipeline`, `/spectral/pipeline-file`,
`/spectral/annotate`, `/spectral/export-concentrations`, `/profile/*`
(confidence-gated single-spectrum path), `track1_benchmark.py` (validation harness).

**Implemented.** The full automated one-file pipeline, the confidence-gated profile
workflow, the 20k-bin scaling fix, MSI/D2O guard, pins, deconvolution + FDR, CSV export,
and the Track-1 → Track-2 hand-off — all deterministic + optional AI, no training required
at serve time.

**Missing / to improve.** UI surfacing of the new Track-1 fields (MSI/D2O/selected ppm);
a true raw-FID → binning batch path; validating on the actual organizer file when it lands.

**Rubric value.** Technical (a complete automated pipeline) · Impact (removes the manual
NMR-profiling bottleneck; on-premise, free) · Reproducibility (one command / one upload,
pinned deps) · Innovation (deterministic + learned evidence in one auditable flow).

**Correctness in bioinformatics & chemistry** *(links independently verified).*
- *Chemistry (processing) —* the **PQN normalization** step is the validated, dilution-robust
  method for ¹H-NMR (more robust than integral/vector-length norm)
  ([Dieterle et al. 2006, *Anal Chem*](https://pubmed.ncbi.nlm.nih.gov/16808434/)), and the
  overall sample→spectrum flow follows the recognized NMR-metabolomics protocol
  ([Beckonert et al. 2007, *Nat Protoc*](https://pubmed.ncbi.nlm.nih.gov/18007604/)).
- *Bioinformatics (reproducibility) —* the pipeline's reporting is built to the community
  **MSI minimum reporting standards** (sample prep, QC, identification, data pre-processing)
  ([Sumner et al. 2007, *Metabolomics*](https://pubmed.ncbi.nlm.nih.gov/24039616/);
  overarching framework, [MSI members 2007, *Nat Biotechnol*](https://pubmed.ncbi.nlm.nih.gov/17687353/)),
  which — with pinned dependencies — is what makes a run share-able and re-runnable.

**Caveats.** The pipeline is only as good as the identification step; every output carries
its confidence/MSI/D2O flags precisely so a reviewer can see where to distrust it. RUO.

---

## Summary table

| Required Track 1 Function | Input Needed | Software Method | Output Produced | Used For | Current Status | Caveat |
|---|---|---|---|---|---|---|
| **1. Compound Classification** | binned ¹H matrix (+ optional pins) | deterministic `annotate` + pSCNN + hybrid + MSI/D2O guard | metabolites w/ positions, coverage, MSI level, D2O reliability | Biomarker Discovery, Workflow | **Built** (deterministic F1 0.90 synth / hybrid 0.40 real) | real F1 modest; MSI L2 never L1; synthetic optimistic |
| **2. Pattern Recognition / ML** | open/simulated spectra (+ on-VM annotations) | `pscnn.py` classifier (wired into the app hybrid) + `quantifier.py` GISSMO transformer | present/absent probs; hybrid identification | AI evidence channel | pSCNN **trained + persisted + wired (hybrid)**; GISSMO quantifier **trained + evaluated → F1 0.13, failed, kept as honest-negative** | pSCNN is a recall booster (precision-respecting hybrid); GISSMO did not close the sim-to-real gap |
| **3. Feature Selection** | binned matrix (+ optional labels) | `select_diagnostic_ppm` (supervised/unsupervised, NCD-anchored) | ranked ppm positions + nearest metabolite + NCD relevance | interpretable panel; p≫n reduction | **Built** | leakage-safe use only; important ≠ causal |
| **4. Biomarker Discovery** | identified compounds + quantities | `deconvolve` (NNLS+FDR) + confidence fields → Track-2 engine | sample × metabolite table w/ confidence | Track-2 discovery/prediction/differential/correlation | **Bridge + Track-2 built** | garbage-in/garbage-out; relative quantities; honest metrics live in Track 2 |
| **5. Automated Workflow** | cleaned TSV/CSV spectrum | `_run_cohort_pipeline`: parse→normalize→annotate→MSI/D2O→deconvolve→AI→select→export | Track-2-ready table (+ optional Track-2 results) | end-to-end NCD screening | **Built** (F7-scaled to 20k bins) | only as good as identification; UI surfacing pending; RUO |

*Full detail: `docs/TRACK1_PLAN.md` (plan + H100 command §9), `docs/IMPACT_AND_VALIDATION.md`
(impact, biology, validation), `docs/TRACK2_FUNCTIONS.md` (downstream).*
