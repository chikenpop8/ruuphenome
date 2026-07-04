# RuuPhenome — Track-1 PLAN (Phenome: NMR Compound Identification)

*Goal: maximize the Phase-4 rubric (Innovation 15, Technical 20, Impact 20, Pitching 15; Bioinformatics correctness 10, Biological interpretation 10, Reproducibility 10) by closing the one real Track-1 gap — a supervised compound classifier trained on the provided annotations — while aggressively reusing the working deterministic pipeline and staying scientifically honest.*

> **Build status (2026-07-02):** Must-haves complete. **DONE:** F6 (MSI identification
> levels + D2O exchangeable-proton guard, `identification_quality.py` → `annotate()`), F4
> (authoritative organizer pins), **F7** (20k-bin NNLS scaling — 188s→~11s via coarse fit
> grid), **F5** (diagnostic-ppm selector `select_diagnostic_ppm`, NCD-anchored), **benchmark
> harness** (`track1_benchmark.py`). **66 tests pass.** Governance #1 RESOLVED (§4/§7).
> **Deterministic baseline (see §8):** annotate F1 0.11 (precision 0.058 — over-permissive);
> **NNLS + target-decoy FDR F1 0.896** (precision 0.839, recall 0.973) on in-distribution
> synthetic — the bar the supervised classifier must beat. **F2 pSCNN built** (`pscnn.py`,
> §9): on clean data deterministic wins (0.96) but under 0.04-ppm drift deterministic
> collapses (F1 0.24) while pSCNN holds (0.55) and hybrid is best (0.59) — the value case
> for a learned evidence channel. **Real held-out GISSMO validation DONE** (`--validate-real`,
> `external_reference.py`, 17 compounds): on real shifts deterministic **collapses to F1 0.16**
> (recall 0.12) while pSCNN **0.33** and hybrid **0.40** — the pSCNN value proven on real, not
> just synthetic, data; low absolutes honestly motivate F8. **F8 BUILT + runnable** (§9):
> `gissmo_corpus.py` (94-compound bundled GISSMO corpus + mixture simulator) + `quantifier.py`
> (Conv-patch Transformer) + `train_on_h100.py --supervised gissmo-quant` — verified end-to-end;
> the exact H100 command is in §9. **INDEPENDENT REAL held-out validation DONE**
> (`--validate-bmrb`): mixtures from **BMRB metabolomics *experimental* peak lists** (18
> compounds, real positions + intensities + multiplets; `bmrb_experimental.py` + bundled
> `open_data/bmrb_experimental_peaks.json`). Honest result on real spectra — permissive
> `annotate` F1 0.64 (precision 0.49), strict FDR F1 0.24 (precision 0.64), pSCNN F1 0.42,
> **hybrid F1 0.53 at precision 0.70 and most offset-robust**; not clinical.
> **F8 TRAINED on H100 + EVALUATED (2026-07-03): it does NOT help.** On the real BMRB held-out set
> the GISSMO quantifier scores F1 **0.13** (worst method); `hybrid+quant` only matches `hybrid`
> (~0.48) by trading precision for recall. **F8 did not close the sim-to-real gap** (it trains on
> fixed-width GISSMO fingerprints without J-multiplets / real intensities). Per the validate-first
> plan, **not scaling the run**. Practical winner stays **hybrid = deterministic + pSCNN**; F8 is
> an honest negative exhibit. **78 tests pass.** **NEXT (only if revisiting F8):** realistic
> lineshapes (task 4); else F3 (on-VM fine-tune loader).

---

## 1. Current state

Track-1 compound identification **already ships and works**, entirely as deterministic / inference-only code. **Do not rebuild any of this.**

| Capability | Where | What it does |
|---|---|---|
| Binned matrix ingestion | `spectral_cohort.load_binned_matrix` (:234) + `pqn_normalize` (:321) | Reads the ~20k-feature intensity×ppm TSV/CSV, auto-detects samples×bins vs transposed, coerces/sorts ppm labels. Generic bin count — nothing hardcodes 20,000. |
| Targeted reference-shift annotation | `spectral_cohort.annotate` (:336) | Calls a metabolite "present" when ≥50% of its ¹H shifts fall within `tol_ppm=0.03` of an occupied bin (median ≥ 75th-pct). Confidence = coverage×100. Reports `matched_shifts` (POSITION). |
| Reference library | `REFERENCE_SHIFTS` (:34) + `_merge_bmrb_library` (:159) | ~140 curated HMDB 5.0 entries extended to ~578 by open BMRB JSON. DSS/TSP internal standards. |
| **NNLS deconvolution (the overlap-resolver)** | `spectral_cohort.deconvolve` (:438) | Models each spectrum as a non-negative linear combo of unit-area Gaussian references (`scipy.optimize.nnls`), with a **proteomics-style target-decoy FDR null** (decoy = each ref shifted 0.37 ppm) and optional DSS/TSP µM calibration. This directly attacks the spec's "superimposed signals" challenge. |
| Single-spectrum path | `signal_processing.assign_metabolites` (:299) | Greedy nearest-peak matching, blended confidence, single-peak compounds capped at 68. |
| Confidence-gated workflow | `profile_workflow.py` | qc → auto_profile → triage → report; blended confidence `0.45·assign + 0.35·fit + 0.20·fdr`. |
| Optional SSL (off by default) | `self_supervised.py` | Masked-conv autoencoder pretrained **without labels** (`labels_used_during_pretraining=False`) on synthetic mixtures of **12** open BMRB standards; used only as a cosine-similarity retrieval side-signal, never overrides assignments. |
| Optional frozen NMRformer | `nmrformer_backend.py` / `_adapter.py` | Inference-only 72-class transformer; **weights not shipped**, requires a pre-picked peak list. |
| Track-1 → Track-2 bridge | `_run_cohort_pipeline` (main.py:996) | annotate → deconvolve → `biomarker_engine.discover` (leakage-safe nested CV + FDR) → `biology.interpret_panel`. **Honest and strong; leave unchanged.** |

**The key gap (the only thing worth building):**

> **No supervised spectrum → compound classifier is trained on the provided partial annotations.** Confirmed by grep: the only `.fit()` calls are Track-2 (metabolite-abundance → phenotype), never bin/spectrum → compound identity. Provided annotations enter only as `{ppm: metabolite}` **pins** merged into the reference dict (`annotate` :367, `parse_identified_peaks` :595) — and are then **silently dropped if the bin sits below the 75th-pct occupancy gate** (:385), never reach `deconvolve`, and train nothing. The spec's *"annotations provided FOR TRAINING"* clause is unaddressed.

Secondary gaps: fixed ~578-compound vocabulary (no open-vocabulary discovery); D2O/solvent is **disclaimed, not modeled** (`solvent_confidence()` always returns `'unverified'`); NNLS is a dense per-sample Python loop that will be the scaling bottleneck at 20k bins.

---

## 2. Core strategy

**The strategic call:** add a **supervised compound classifier trained on the provided annotations** as the Track-1 centerpiece — the exact thing the spec asks for and the one thing currently missing — **bolted onto the existing pipeline, not replacing it.**

Three tiers, in priority order:

1. **Centerpiece — supervised classifier (pSCNN-style pseudo-Siamese 1D-CNN).** Takes a raw (reference-standard spectrum, sample spectrum) **pair** into two weight-independent conv towers → present/absent per compound. Precedent: pSCNN 97.6% on real known mixtures, handles ±0.015 ppm drift, raw-spectrum-in (no fragile peak-picker). This satisfies "identify TYPE and POSITION" and "annotations for training" simultaneously and is **open-data trainable** by superposing reference standards.

   > **Honesty framing (verify-phase correction):** the claim that a supervised CNN is *better* than the shipped deterministic-NNLS + FDR pipeline is a **hypothesis to validate, not an established fact.** No head-to-head on 1D blood-in-D2O exists; the same literature shows CNN accuracy typically collapses from 95–99% to 66–90% under proper subject-wise/external validation; and the sim-to-real gap burdens the CNN *as much as* the current pipeline. We therefore ship the CNN **as an evidence channel blended through the existing `hybridize()` contract**, benchmarked against the deterministic baseline — never as a replacement, and never with a leaky in-distribution number as its headline.

2. **Optional — SSL warm-start.** Keep `self_supervised.py` exactly as-is (label-free pretraining) as an *optional* encoder warm-start for the CNN. Because superposing reference standards yields effectively unlimited labeled pairs, the label-scarcity premise that motivates SSL is weak here, so SSL stays **optional/supportive**, not load-bearing.

3. **Exactly ONE H100 "hero" innovation — a GISSMO-trained transformer quantifier** (compound identity + concentration from physically-exact simulated mixtures with known ground-truth). Precedent: transformer beat MLP/CNN decisively at 86 metabolites, ~3–10% MAPE @ 400 MHz on 250k simulated spectra. This is genuinely H100-scale (the *data-generation + sweep* is where the GPU earns its keep) and delivers ID + quantification + biomarker-ready output in one model.

   > **Rejected hero (verify-phase correction):** low-field→high-field **super-resolution is NOT the hero.** It is a crowded 2024–25 subfield (not novel), the most direct paper shows L→H conversion *degrades* quantification accuracy by 80–673%, and nothing is validated on 1D blood-in-D2O. If shown at all, it is an optional visualization side-panel explicitly off the quantification critical path.

---

## 3. Functions to build

| # | Function | H100? | Effort |
|---|---|---|---|
| F1 | `make_blood_mixture_corpus()` — synthetic labeled training data | optional | M |
| F2 | `train_pscnn_head()` — supervised pseudo-Siamese classifier (off-VM pretrain) | optional | L |
| F3 | `finetune_head_on_annotations()` — on-VM fine-tune on provided labels | no | M |
| F4 | `authoritative_pins` — make provided annotations ground truth in `annotate` | no | S |
| F5 | `select_diagnostic_ppm()` — sparse feature selection / important-position ranking | no | M |
| F6 | `msi_confidence_level()` + `d2o_exchangeable_guard()` — honesty layer | no | S |
| F7 | `crop_and_batch_nnls()` — 20k-bin deconvolution scaling fix | no | S |
| F8 | `train_gissmo_quantifier()` — the H100 hero transformer | **yes** | L |

### F1 — `make_blood_mixture_corpus()`
- **What:** Generates tens of thousands of labeled synthetic ¹H-NMR **blood-matrix** mixtures with known component identities/positions/concentrations.
- **Biology/bioinformatics of how:** Linearly superpose pure-compound reference spectra (from `REFERENCE_SHIFTS` / GISSMO spin matrices) at random lognormal concentration ratios, add the **broad protein/lipoprotein background** of blood (MetAssimulo blood mode reaches ~0.82 correlation with real blood), then augment: ppm roll (±0.015 shift), peak-width scaling, Gaussian noise, HDO/water-region (~4.7–4.8 ppm) masking. Because the matrix is **D2O**, exchangeable −COOH/−OH/−NH protons are removed/never simulated as diagnostic — only non-exchangeable C–H protons carry labels.
- **Reuses:** `_augment_batch` (self_supervised.py:106) already synthesizes labeled overlapping mixtures and *knows the component indices* (free multi-label target); `make_demo_binned` (spectral_cohort.py:772) already simulates binned cohorts from the fingerprint library; `GRID_PPM` (open_data.py:29).
- **Innovation:** Matrix-aware (blood + protein background + D2O exchangeable handling) synthetic corpus — most NMR-ML corpora are aqueous pure standards.
- **Thai-NCD impact:** Guarantees glucose / BCAAs / lactate / alanine are represented so the classifier learns the exact diabetes/insulin-resistance signals Thai cohorts validate.
- **H100:** optional (bulk generation + sweeps benefit from GPU). **Effort:** M.

### F2 — `train_pscnn_head()`
- **What:** Trains the pseudo-Siamese 1D-CNN present/absent classifier on F1's corpus **off-VM on open/simulated data only.**
- **Biology/bioinformatics of how:** Two weight-independent 6-conv-layer towers (32 kernels, 5×1, ReLU + maxpool) → concat → dense(100) → dropout 0.2 → sigmoid, on a (reference, sample) **pair**. CNN translation-invariance absorbs pH/temperature/referencing chemical-shift drift, which is *why* it resolves overlapping signals. One-vs-rest against each reference means new compounds need no retraining.
- **Reuses:** `MaskedSpectrumAutoencoder.embed()` as an optional frozen/warm-start backbone; `train_on_h100.py` config-derived checkpoint naming (`_checkpoint_suffix`) wraps a new `--supervised` mode unchanged.
- **Innovation:** First *learned* (not deterministic) compound identifier in RuuPhenome; scales to open vocabulary via pair-matching.
- **Thai-NCD impact:** Learns overlapping-signal disambiguation in crowded 0.9–1.05 (BCAA) and anomeric regions where deterministic matching struggles.
- **H100:** optional (trains on modest GPU; H100 helps sweeps). **Effort:** L.

### F3 — `finetune_head_on_annotations()`
- **What:** Fine-tunes only the classifier head on the competition-provided `{ppm: metabolite}` annotations **on the governed VM** (`NMR_OFFLINE=1`).
- **Biology/bioinformatics of how:** Freeze the encoder, train the final Linear/sigmoid head on the provided partial labels via cross-entropy/BCE. This is the literal implementation of the spec's "annotations for training." Add a leakage/exfiltration guard (subject-wise split inside the provided set; never writes network).
- **Reuses:** `parse_identified_peaks` (spectral_cohort.py:595) already parses the exact `{ppm: metabolite}` format a training loader needs; `NMR_OFFLINE` guard pattern (open_data.py:58).
- **Innovation:** Closes the "annotations provided for training" gap that no current code addresses.
- **Thai-NCD impact:** Adapts the open-data-pretrained model to the real competition blood-in-D2O distribution before inference.
- **H100:** no (small head, CPU/seconds). **Effort:** M.
  > **Governance caveat — see §4/§7:** fine-tuning on the *provided competition annotations* is **training on the closed dataset**, which the open-data-only rule as written forbids. This is a **policy question for the user**, not a solved point.

### F4 — `authoritative_pins`
- **What:** Make organizer-provided annotations **authoritative** in `annotate` instead of weak hints.
- **How:** An organizer label is ground truth, so bypass the occupancy/coverage gate (spectral_cohort.py:385/389) for pinned `{ppm: metabolite}` entries and force-add their metabolite columns; also thread pins into `deconvolve` (currently ignores them).
- **Reuses:** existing `identified_peaks` merge path.
- **Innovation:** Fixes a correctness bug (known-correct labels silently dropped).
- **Thai-NCD impact:** Ensures organizer-confirmed NCD metabolites always appear. **H100:** no. **Effort:** S.

### F5 — `select_diagnostic_ppm()`
- **What:** Ranks the most important ppm positions (spec approach 3 + biomarker discovery), reduces the p≫n (~20k feature) burden.
- **Biology/bioinformatics of how:** Sparse/regularized selection (LASSO/elastic-net or ST-CS compressed-sensing+clustering — **not naive PCA on sparse bins**, which underperforms) over CNN saliency / occupied bins. Report selected ppm windows as the biomarker deliverable, anchored on diagnostic handles: glucose α-anomeric H1 ~5.23 ppm (J≈3.7 Hz), lactate CH₃ ~1.33 ppm, BCAA methyls ~0.9–1.05 ppm, alanine ~1.48 ppm.
- **Reuses:** Track-2 `dimensionality.py` / `biomarkers.py` selection machinery.
- **Innovation:** Ties learned saliency to interpretable, sparse ppm positions. **Thai-NCD impact:** Directly outputs the glucose+BCAA insulin-resistance panel. **H100:** no. **Effort:** M.

### F6 — `msi_confidence_level()` + `d2o_exchangeable_guard()`
- **What:** Tags every call with an MSI level and forbids fabricated D2O chemistry.
- **Biology/bioinformatics of how:** Emit **Sumner 2007 MSI level** per call — a label-trained classifier with no authentic-standard spike-in and no orthogonal confirmation is **Level 2 (putatively annotated), never Level 1** (optionally Schymanski 1a/1b sublevels). The D2O guard rejects any diagnostic peak on −COOH/−OH/−NH protons or the HDO/water region (~4.7–4.8 ppm) — only non-exchangeable C–H protons are valid features. Require convergent evidence (shift + multiplicity + J-pattern; all expected signals present). Flag pH-labile metabolites (histidine, taurine, citrate). Assert **DSS** (preferred over pH-sensitive, protein-binding TSP) at 0.00 ppm.
- **Reuses:** `solvent_confidence()` (spectral_cohort.py:208) is a deliberately queryable stub designed as exactly this write-point.
- **Innovation:** Turns the disclaimed D2O gap into an enforced correctness guard (unit test fails if an exchangeable-proton peak is exposed as a match feature).
- **Thai-NCD impact:** Keeps quantification claims honest for protein-binding metabolites (tyrosine, histidine, lactate). **H100:** no. **Effort:** S.

### F7 — `crop_and_batch_nnls()`
- **What:** Fixes the 20k-bin deconvolution bottleneck.
- **How:** Crop to the informative ~0.5–9 ppm window and vectorize/batch the per-sample `nnls` loop (spectral_cohort.py:487); the ASICS pre-screen already prunes references, so only the bin dimension + Python loop need attention. **Reuses:** existing `deconvolve`. **Innovation:** reproducibility/efficiency at competition scale. **H100:** no. **Effort:** S.

### F8 — `train_gissmo_quantifier()` — **THE H100 HERO**
- **What:** A transformer that outputs compound identity + concentration from raw spectra, trained on physically-exact GISSMO-simulated mixtures with known ground truth.
- **Biology/bioinformatics of how:** GISSMO provides parameterized spin systems (chemical shifts + J-couplings) for 1286+ compounds → simulate physically exact overlapping ¹H mixtures at the competition field with **known concentrations AND ppm positions**. Train a 6-layer/8-head transformer encoder (self-attention over spectral regions captures long-range peak relationships within one metabolite). Optionally a CNN+GRU (NMRQNet-style) head for position/quantification.
- **Reuses:** `train_on_h100.py` LiCO/H100 launcher + auto-named checkpoints; `GRID_PPM`; the deterministic NNLS + target-decoy FDR as the **reproducible baseline to benchmark against**.
- **Innovation:** Physics-simulated, ground-truth-labeled quantifier — the hero. **Thai-NCD impact:** Delivers absolute-ish concentration estimates for glucose/BCAAs/lactate feeding Track-2. **H100:** **yes.** **Effort:** L.

---

## 4. H100 training plan

**Governance pattern (applies to every target):** *pretrain on open/simulated data OFF-VM → move checkpoint onto the governed VM → (optionally) fine-tune only the head on provided annotations ON the VM with `NMR_OFFLINE=1` → never exfiltrate.* Corpus download is best-effort and blocked by `NMR_OFFLINE=1`; on LiCO upload the bundled npz, do **not** pass `--rebuild-corpus`.

| Target | Open dataset(s) | Why an H100 is genuinely needed | Command sketch |
|---|---|---|---|
| **F8 GISSMO transformer (hero)** | GISSMO spin matrices (1286+), HMDB, MetAssimulo blood-mode sims | 250k-spectra simulated corpus + large transformer + augmentation/hyperparameter sweeps across 44–87 compounds — this is the quarter-million-spectrum regime the winning paper used | `python -m nmr_api.train_on_h100 --supervised gissmo-quant --epochs 200 --batch-size 256 --embedding-dim 128 --field-mhz <comp>` |
| **F2 pSCNN classifier** | Open BMRB standards + F1 superposed blood mixtures | Not H100-*demanding* (1D-CNN is modest) but H100 earns keep on one-vs-rest × many compounds + augmentation sweeps generating thousands of pairs | `python -m nmr_api.train_on_h100 --supervised pscnn --pairs 22000 --epochs 200 --batch-size 256` |
| **F1 corpus generation** | GISSMO/HMDB reference spectra | Bulk generation of tens of thousands of matrix-aware mixtures parallelizes well on GPU | (invoked by the above via `--rebuild-corpus` **off-VM only**) |
| **SSL warm-start (optional)** | `bmrb_1h_corpus.npz` (expand via `build_open_corpus.py` off-VM to ~5.5k BMRB, **remap entry-ids→names**) | Existing `train_on_h100` path (epochs 200, batch 256, dim 128) | `python -m nmr_api.train_on_h100 --epochs 200 --steps-per-epoch 128 --batch-size 256 --embedding-dim 128` |

> **Honest H100 note:** the models themselves (1D-CNN / CNN+GRU / small transformer on 10k–40k-length vectors) are not compute-heavy; the H100 is justified by **data generation scale + sweeps**, not model size. Say this plainly to judges.

**On-VM fine-tune (F3), the governance-sensitive step:**
`NMR_OFFLINE=1 python -m nmr_api.finetune_head --labels provided_annotations.json --encoder models/<checkpoint>.pt --freeze-encoder --subject-wise-split` — reads only local provided annotations, writes only a local head checkpoint, network-guarded.

> **Policy tension — RESOLVED (user decision 2026-07-02):** provided competition
> annotations MAY be used for supervised Track-1 fine-tuning, but **only inside the
> governed offline VM** (`NMR_OFFLINE=1`); the closed data is **never exported/uploaded**.
> Documented split: **open/simulated data = pretraining (off-VM); provided annotations =
> local competition fine-tuning only (on-VM)**. F3 is approved under that strict constraint.

---

## 5. Phased roadmap

**MUST-HAVE (do first):**
| Item | Moves |
|---|---|
| F6 MSI-level tagging + D2O exchangeable guard + DSS anchor | Bioinformatics correctness, Biological interpretation |
| F4 authoritative pins (+ pins into deconvolve) | Technical, Bioinformatics correctness |
| F7 20k-bin NNLS crop/batch | Technical, Reproducibility |
| F5 sparse diagnostic-ppm selection anchored on glucose/BCAA/lactate | Impact, Biological interpretation, Innovation |
| External/held-out benchmark harness (CNN & hero vs deterministic baseline) | Reproducibility, Technical |

**SHOULD-HAVE:**
| Item | Moves |
|---|---|
| F1 blood-matrix synthetic corpus | Technical, Innovation |
| F2 pSCNN supervised classifier (off-VM open-data) blended via `hybridize()` | **Innovation, Technical** (the spec's missing centerpiece) |
| F3 on-VM head fine-tune on provided annotations *(pending §7 policy)* | Technical, Impact |
| F8 GISSMO H100 hero quantifier | **Innovation, Impact, Pitching** |

**NICE-TO-HAVE:**
| Item | Moves |
|---|---|
| SSL corpus expansion (12 → ~5.5k, names remapped) | Technical |
| NMRQNet-style CNN+GRU quant head on the hero encoder | Innovation, Impact |
| L→H super-resolution as an **optional visualization side-panel only** (off critical path) | Pitching (with explicit caveat) |
| Open-vocabulary discovery beyond the 578-entry library | Innovation |

---

## 6. Honesty guardrails

- **MSI levels on every call.** Sumner 2007 four-level scheme (optionally Schymanski 1a/1b). A label-trained classifier with no authentic-standard spike-in and no orthogonal confirmation is **Level 2, never Level 1.** Make "reports a confidence level per identification" an asserted behavior.
- **Do not fabricate D2O/solvent chemistry.** Only non-exchangeable C–H protons are valid features; reject any −COOH/−OH/−NH or HDO-region (~4.7–4.8 ppm) diagnostic peak. Unit test fails if an exchangeable-proton peak is exposed. Use tolerance windows (±0.03 ppm, wider for ionizable groups), flag pH-labile metabolites, assert DSS (not TSP) at 0.00 ppm. Do not emit absolute concentrations for protein-binding metabolites (tyrosine, histidine, lactate) without a CPMG/deproteinization caveat.
- **External / held-out evaluation is mandatory.** Train on synthetic/open, hold out **real** spectra from a different instrument/day/site; **subject-wise (not spectrum-wise) splits**; report external/cross-acquisition accuracy separately from in-distribution. Cite the leakage evidence (95–99% leaky → 66–90% honest; one study 94%→66%) so judges trust the number. Benchmark every learned model against the deterministic NNLS + target-decoy FDR baseline (`mean_fit_r2`) and against an open method (NMRQNet template).
- **Keep SSL optional.** Because superposition yields unlimited labels, SSL is a supportive warm-start, not a necessity; label it as such.
- **No leaky / in-distribution numbers presented as accuracy.** The bundled SSL retrieval benchmark (top1=0.975) is explicitly *self-referential* (augmented queries vs the same 12 refs) — never present it as mixture-ID accuracy. Any CNN headline must be the external number.
- **"Better than current" is a hypothesis, not a claim.** State the supervised classifier as a candidate to validate against the leakage-safe deterministic baseline, not a proven improvement.

---

## 7. Open questions for the user

1. **Governance policy (blocking F3) — RESOLVED (2026-07-02):** provided annotations may be used for supervised fine-tuning **only inside the governed offline VM** (`NMR_OFFLINE=1`), closed data never exported; open/simulated = pretraining, provided annotations = local fine-tuning only. ~~open question~~
2. **Competition field strength (MHz)** for GISSMO/MetAssimulo simulation — needed to make synthetic lineshapes match the real spectra. What is it?
3. **Is a held-out REAL blood-in-D2O ¹H set available** (even unlabeled) for the mandatory external/cross-acquisition check? Without one, the sim-to-real gap stays unquantified and every learned number must be flagged as in-distribution-only.
4. **Referencing standard actually used** in the competition spectra — DSS, TSP, or none? Affects the 0.00 ppm anchor and cross-sample alignment validity.
5. **Compute budget on LiCO** — is the hero (F8, ~250k-spectra transformer regime) within the GPU_FOR_BDI 1-GPU / 1h wall-clock allotment, or should the hero be scoped to a smaller sweep?
6. **Scope confirmation:** is one H100 hero (F8) plus the supervised classifier (F2/F3) the right ambition for the timeline, or should we ship must-haves + F2 only and treat F8 as stretch?

---

**Files referenced (all absolute):**
- `/Applications/Vibing coding/Noom copy cat/ruuphenome/backend/nmr_api/spectral_cohort.py`
- `/Applications/Vibing coding/Noom copy cat/ruuphenome/backend/nmr_api/self_supervised.py`
- `/Applications/Vibing coding/Noom copy cat/ruuphenome/backend/nmr_api/signal_processing.py`
- `/Applications/Vibing coding/Noom copy cat/ruuphenome/backend/nmr_api/train_on_h100.py`
- `/Applications/Vibing coding/Noom copy cat/ruuphenome/backend/nmr_api/open_data.py`
- `/Applications/Vibing coding/Noom copy cat/ruuphenome/backend/nmr_api/build_open_corpus.py`
- `/Applications/Vibing coding/Noom copy cat/ruuphenome/backend/nmr_api/main.py`
- `/Applications/Vibing coding/Noom copy cat/ruuphenome/backend/nmr_api/profile_workflow.py`
- `/Applications/Vibing coding/Noom copy cat/ruuphenome/docs/H100_TRAINING.md`

---

## 8. How the must-have layer improves the score (built 2026-07-02)

### Bioinformatics correctness (10)
| Feature | What it enforces | Why it's *correct*, not cosmetic |
|---|---|---|
| **MSI levels** (`identification_quality.msi_level`) | Every identification is tagged **Level 2 (putative, library-similarity)** — **never Level 1** without an authentic standard (Sumner 2007). Weak matches → L3. | Stops the classic over-claim of "identified" when only a chemical shift matched. On the demo: 186×L2 + 73×L3, **0×L1** — auditable and honest. |
| **D2O exchangeable-proton guard** | Residual-water/HDO (4.70–4.90 ppm) and downfield-exchangeable (>9.5 ppm) matches are **excluded from evidence**; an ID **requires a non-exchangeable C-H resonance**; abundance is computed from robust bins only; DSS (not pH-sensitive TSP) is the 0.00 ppm anchor. | In D2O, –COOH/–OH/–NH protons vanish (Haslauer 2019). Matching on them fabricates chemistry. The guard *prefers non-exchangeable resonances* (the honest fix) instead of inventing solvent corrections. Demo: 268→**259** metabolites (water/exchangeable-only calls correctly dropped). |
| **Authoritative organizer pins** | Provided `{ppm:metabolite}` labels **bypass the occupancy gate** (ground truth), tagged `provenance:"organizer_pin"`. | Fixes a real bug — known-correct labels were previously *silently dropped* below the occupancy threshold. |
| **F7 NNLS scaling** | 20k-bin deconvolution runs on a coarse fit grid: **188 s → ~11 s**, identical small-grid results. | Reproducible at competition scale without a stalled demo. |
| **Benchmark harness** (`track1_benchmark.py`) | Scores identification precision/recall/F1 vs planted ground truth, with an explicit **in-distribution caveat**. | Gives a *measured* baseline instead of an unquantified claim. |

### Biological interpretation (10)
- **F5 diagnostic-ppm selector** (`select_diagnostic_ppm`) turns ~20k anonymous bins into a ranked shortlist where **each position is annotated to its nearest reference metabolite and its NCD relevance** — glucose 5.23 ppm → hyperglycemia/diabetes, lactate 1.33 → glycolysis/insulin resistance, BCAA methyls 0.9–1.05 → T2D risk (Newgard 2009, Guasch-Ferré 2016), etc. This is the bridge from a spectrum to the Thai-NCD story and straight into Track-2 discovery.
- Every identification now carries `msi_level`, `robust_coverage`, and a `d2o` block (non-exchangeable shifts, water/exchangeable counts, caveat) — so a reviewer can see *why* a metabolite was called and how trustworthy it is in D2O.

### Deterministic baseline (the bar to beat)
`python -m nmr_api.track1_benchmark` (10 cohorts, 15 planted compounds, in-distribution synthetic, D2O-observable ground truth):

| Method | Precision | Recall | F1 |
|---|--:|--:|--:|
| `annotate` (reference-shift matching) | 0.058 | 1.00 | 0.11 |
| **`deconvolve` + target-decoy FDR** | **0.839** | **0.973** | **0.896** |

The permissive matcher's 0.058 precision quantifies the "over-annotation" gap; the **FDR-controlled NNLS deconvolution (F1 ≈ 0.90)** is the real deterministic baseline. **Honesty:** in-distribution synthetic (library == matcher) → optimistic; a held-out REAL D2O-blood benchmark is still required before any accuracy claim. The supervised classifier (F2) will be added as an *extra evidence channel* and must beat this — especially on the (still-needed) real held-out set, where deterministic methods degrade most.

---

## 9. F2 — pSCNN classifier + deterministic-vs-pSCNN-vs-hybrid comparison (built 2026-07-02)

**F2 built** (`pscnn.py`): pseudo-Siamese 1D-CNN, two weight-independent BatchNorm-conv
towers + a head with explicit matching features (`r·s`, `|r−s|`); trained on open
synthetic superposed mixtures with **ppm-drift augmentation**; D2O-guard-consistent
(non-exchangeable fingerprints only); **OPTIONAL** (no checkpoint → silently off, like
SSL). It is an **added evidence channel**, blended — never a replacement.

**Comparison** (`python -m nmr_api.track1_benchmark --compare`; 12-compound panel, 20
mixtures/condition, synthetic):

| Condition | Deterministic (NNLS+FDR) | pSCNN | Hybrid |
|---|--:|--:|--:|
| **Easy** (clean, in-distribution) | **F1 0.96** | 0.71 | 0.90 |
| **Hard** (noise + 0.04 ppm drift) | F1 **0.24** (recall 0.16) | 0.55 | **0.59** |

**The honest story:** on clean/in-distribution data the **deterministic method wins**
(so pSCNN correctly stays a *complement*); under ppm drift *beyond the fixed
tolerance*, deterministic **collapses (0.24)** while the drift-augmented pSCNN **holds
(0.55)** and the **hybrid is best (0.59)**. That is the defensible value case for a
learned channel — robustness to real experimental variation.

### Real held-out validation on GISSMO (built 2026-07-02) — the decisive test
`python -m nmr_api.track1_benchmark --validate-real`. **Held-out by construction:** the
reference library + pSCNN training use our HMDB-derived shifts; the TEST spectra are built
from **independent GISSMO real ¹H shifts** (spin systems fit to experimental BMRB spectra,
via `external_reference.py`) — 17 compounds incl. all BCAAs + aromatic AAs.

| Condition | Deterministic | pSCNN | Hybrid |
|---|--:|--:|--:|
| **GISSMO real shifts (clean)** | F1 **0.16** (recall 0.12) | 0.33 | **0.40** |
| **GISSMO real + 0.03 ppm referencing offset + noise** | 0.24 | 0.33 | **0.37** |

**Findings (honest):** (1) On *real* shifts BOTH methods drop far below the ~0.90 synthetic
number — the real GISSMO peaks (dense multiplet structure, exact positions) differ from our
simplified library; this **sim-to-real gap is the reality**, and the simplified reference
library is the bottleneck. (2) **The deterministic fixed-tolerance matcher collapses (0.16,
recall 0.12)** on real peak variation. (3) **The pSCNN ~doubles it (0.33)** — the learned,
drift-augmented model generalises to real shift variation far better. (4) **The hybrid is
best in both conditions (0.40 / 0.37)** — the practical innovation. This proves the pSCNN's
value on **real held-out data**, not only synthetic drift.

> **The clean narrative:** deterministic is best on clean/in-distribution spectra; **pSCNN
> helps under real-world drift/noise; the hybrid method is the practical winner.** The low
> real-data absolutes are stated honestly and directly **motivate F8** — train on the real
> GISSMO peak patterns (not just the simplified library) to close the sim-to-real gap.
> Physically-exact simulated (from experimental BMRB fits), **not clinical validation.**

### Held-out REAL / realistic datasets for the decisive test (verified endpoints)
- **BMRB metabolomics** — ~1,100 pure-metabolite standards, ¹H peak lists in **D₂O / pH 7.4 / DSS** (biofluid-relevant), open (CC0). One true compound per entry = ground truth. `https://bmrb.io/ftp/pub/bmrb/metabolomics/entry_lists/experimental/bmseNNNNNN.str` (e.g. alanine `bmse000028`, glucose `bmse000015`); raw Bruker under `.../entry_directories/bmseNNNNNN/nmr/set01/1D_1H/`.
- **GISSMO** — 1,286 compounds, physically-exact simulated ¹H spectra + spin matrices (shifts + J). Build **realistic overlapping mixtures where deterministic fails**. `https://gissmo.bmrb.io/entry/list`; peaks at field: `https://gissmo.bmrb.io/entry/{ID}/simulation_1/peaks/{MHz}`; all sims: `https://gissmo.bmrb.io/static/all_simulations.zip`. (This is also the F8 hero's training data.)
- **MetaboLights MTBLS1** — real human-urine ¹H spectra (`FILES/*.nmrML` + Bruker zips) **plus** an identified-metabolite MAF (`m_MTBLS1_..._maf.tsv`) → real biofluid held-out with identity labels. `https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/MTBLS1/`.
- CASMDB (1,932 curated entries, `github.com/ccpnmr/CASMDB`) — answer-key library (note NC clause on HMDB-derived parts).

### Commands
```bash
# F2 — train the pSCNN at scale (OPEN data only; off-VM). Bundled reference library,
# downloads nothing. Runs on CPU (slow) or GPU. Saves models/pscnn_identifier.pt.
cd backend && python -m nmr_api.pscnn --mixtures 20000 --epochs 200 --grid 4096 --batch-size 256

# On-VM fine-tune on provided competition annotations (governance §4): pending F3 loader.

# Reproduce the deterministic baseline and the 3-way comparison:
python -m nmr_api.track1_benchmark                 # deterministic baseline
python -m nmr_api.track1_benchmark --compare       # deterministic vs pSCNN vs hybrid
```

### F8 — GISSMO H100 hero quantifier (BUILT + runnable, 2026-07-02)
A transformer that maps a ¹H spectrum → per-compound **relative concentration** (identity =
concentration above a threshold), trained on GISSMO-simulated mixtures. Built:
`gissmo_corpus.py` (fetch/bundle GISSMO shifts off-VM + on-the-fly mixture simulator with
concentration labels + ppm-drift augmentation), `quantifier.py` (Conv patch-embed → Transformer
encoder → softplus concentrations; OPTIONAL, no checkpoint → off), and the
`train_on_h100.py --supervised gissmo-quant` mode (config-tagged checkpoints). Verified
end-to-end on CPU/MPS; loss decreases; recovers planted compounds. **71 tests pass.**

**1. What you're training:** the GISSMO transformer quantifier (identity + relative
concentration) on open GISSMO-simulated ¹H mixtures.
**2. Why H100:** the ~250k-mixture-per-epoch generation + augmentation + the transformer over
long (up to 4096-bin) spectra + hyperparameter sweeps — data-generation scale, not model size.
**3. Exact command (from `backend/`):**
```bash
# Optional: expand the corpus OFF-VM first (needs internet), then commit the JSON:
python -c "from nmr_api import gissmo_corpus; print(gissmo_corpus.build_corpus(id_range=(1,400)))"
# Train on the H100/LiCO node (loads the bundled corpus; downloads nothing):
python -m nmr_api.train_on_h100 --supervised gissmo-quant \
    --field-mhz <COMPETITION_FIELD_MHZ> --mixtures 250000 --epochs 200 \
    --batch-size 256 --n-bins 4096 --patch 16
```
**4. Expected runtime:** ~a few hours on 1× H100 (dominated by data-gen). For a 1-GPU / 1-hour
LiCO slot, use `--mixtures 40000 --epochs 60`.
**5. Required input files:** `backend/nmr_api/open_data/gissmo_corpus.json` (bundled; built
off-VM) + the code. **No downloads on the node; no closed data.**
**6. Expected output files:** `models/gissmo_quantifier.pt` (+ config-tagged copy
`models/gissmo_quantifier_<epochs>ep_cuda_b<batch>.pt`) and `models/gissmo_quantifier_report.json`
(+ tagged copy).
**7. Send back:** copy **only** those two files (the `.pt` checkpoint + the `.json` report) off
the node — never any patient/closed data. We load the checkpoint and benchmark identification vs
deterministic + pSCNN + hybrid.
- **Governance:** GISSMO/simulated = open pretraining; provided annotations only for on-VM
  fine-tuning (`NMR_OFFLINE=1`), never exported.
- **Honest caveat:** real held-out F1 is still low; F8 is meant to **close the sim-to-real gap**
  by training on real GISSMO patterns. Its fair evaluation is on a real source it did NOT train
  on (BMRB experimental / MTBLS1) — the next validation step. Not clinical validation.
