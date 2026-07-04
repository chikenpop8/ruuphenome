# RuuPhenome — Current Project Handoff

Last verified: **July 4, 2026** · Pre-ship bug hunt + fixes, verified bibliography **2026-07-04** (111 tests passing)  
Active app root: `/Applications/Vibing coding/Noom copy cat/ruuphenome`

> **2026-07-02 — Track 2 analytics expansion (items 1–5).** The biomarker engine
> and model suite now report the **full classification metric set** (accuracy,
> sensitivity, specificity, precision, recall + **confusion matrix**; macro forms
> for multi-class), a **top-1/3/5/10 minimal-panel sweep**, and **multi-class
> (≥3 group) support** end-to-end (macro one-vs-rest AUC, per-class recall,
> multinomial LR; `derive_labels(multiclass=True)` preserves all groups instead of
> collapsing to two). Added `random_forest` to the model suite. New modules
> `differential.py` (`/track2/differential`: Mann-Whitney/Welch + Kruskal/ANOVA,
> BH q-values, volcano) and `correlation.py` (`/track2/correlation`: pairwise +
> **partial-correlation GGM network** per Krumsiek 2011, via Ledoit-Wolf
> shrinkage). UI (`profiler.html`) surfaces all of these. **No training required —
> all classical statistics/ML that fit fresh per upload.** The binary path is
> byte-for-byte unchanged, so prior numbers reproduce.

> **2026-07-02 — Judge-proofing (Impact + Phenome track).** Added **bootstrap 95%
> CIs** on every AUC (`discover()` → `honest_roc_auc_ci95`, group-level percentile
> bootstrap; flagged approximate per Bengio & Grandvalet 2004) — shown in the UI.
> Added a one-command **external cross-cohort validation** tool
> (`python -m nmr_api.external_validation`, RUO, open-data) that runs the
> leakage-safe engine on unseen MetaboLights cohorts; validated on **MTBLS161
> (ME/CFS)** — serum AUC 0.742 [0.62–0.87], urine 0.720 [0.56–0.85], both perm
> p=0.015, no leakage inflation. Recorded honest bundled-cohort numbers **with
> CIs**: MTBLS1 diabetes **0.932 [0.90–0.98]** (panel recovered isoleucine +
> 2-oxoisovalerate = the BCAA signature), MTBLS356 0.792, MTBLS424 0.572 (honest
> weak-signal exhibit). Pinned deps (`requirements.lock.txt` + `.python-version`).
> New fully-cited **`docs/IMPACT_AND_VALIDATION.md`** (Thai NCD burden [IDF/NHES7],
> metabolite→pathway→disease biology, validation methods, datasets/commands,
> limitations, rubric map). 55 tests pass. No unseen open diabetes ¹H-NMR
> disease/control cohort was found for external testing (most are MS-based,
> access-gated, or empty MAFs) — see docs §7.
>
> The impact/validation doc now also carries: **§9 per-function workflow / impact /
> innovation** (with an SVG diagram `docs/track2_workflow.svg`), **§10 completed
> score-focused upgrades**, and **§11 pitch talking points + demo script**. Dataset
> search extended across MetaboLights FTP (118 studies MTBLS2–119) **and
> Metabolomics Workbench REST**: **no** open *binary* T2D-vs-control ¹H-NMR cohort
> with a populated per-sample table exists (diabetes metabolomics is overwhelmingly
> MS-based). **Added ST004325** (Metabolomics Workbench; 1D ¹H-NMR in **D₂O** @ 600
> MHz; T1D urine, 3 duration groups, 247 samples) as a second external cohort — it
> exercises the **multi-class** path on real NMR data: macro-AUC 0.598 [0.56–0.67],
> **leakage inflation 0.0003**, perm p 0.005. `external_validation.py` now loads both
> MetaboLights (FTP MAF) and Metabolomics Workbench (REST) cohorts (`_build_tables` /
> `_build_tables_mw`). Both are pipeline-validation demos, not clinical (docs §4b,
> §7). Also fixed: `screen_features` is now NaN-safe (multi-class `f_classif` needs
> finite input).

> **2026-07-02 — Track-1 plan + honesty layer (started).** Full plan in
> **`docs/TRACK1_PLAN.md`** (grounded, scored; strategy = supervised pSCNN classifier as an
> *added evidence channel* + one H100 hero = **GISSMO-simulated-mixture quantifier**; L→H
> super-resolution rejected as hero). **Governance decided:** provided competition
> annotations may be used for supervised Track-1 fine-tuning **only inside the offline VM**
> (`NMR_OFFLINE=1`), never exported; open/simulated = pretraining. **Built the honesty layer:**
> new `identification_quality.py` (MSI levels — library match = **Level 2, never Level 1**
> without an authentic standard; + **D2O exchangeable-proton guard** — residual-water/HDO
> 4.70–4.90 ppm and >9.5 ppm matches excluded from evidence, identification requires a
> non-exchangeable C-H resonance, DSS anchor), wired into `spectral_cohort.annotate()`, plus
> **authoritative organizer pins** (provided `{ppm:metabolite}` now bypass the occupancy gate
> instead of being silently dropped). Demo: 268→259 metabolites, MSI 186×L2 + 73×L3, 0×L1.
> Also built **F7** (20k-bin NNLS scaling — coarse fit grid, `_NNLS_BIN_CAP`; 188s→~11s,
> demo path unchanged), **F5** (`spectral_cohort.select_diagnostic_ppm` — ranks important
> ppm positions, annotated to nearest metabolite + NCD relevance), and the **benchmark
> harness** (`track1_benchmark.py`). **Deterministic baseline** (in-distribution synthetic):
> `annotate` F1 **0.11** (precision 0.058 — over-permissive), **`deconvolve`+target-decoy
> FDR F1 0.896** (precision 0.839, recall 0.973) — the bar the supervised classifier must
> beat; held-out REAL benchmark still needed (honesty caveat in docs §8). How the layer
> improves Bioinformatics-correctness + Biological-interpretation is documented in
> `docs/TRACK1_PLAN.md §8`. **66 tests pass.** NEXT: F2 pSCNN (added evidence channel via
> `hybridize()`), then F8 GISSMO H100 hero.

> **2026-07-02 — Track-1 F2 pSCNN + comparison (built).** New `pscnn.py` — pseudo-Siamese
> 1D-CNN compound identifier (BatchNorm towers + `r·s`/`|r−s|` matching head; open synthetic
> mixtures with ppm-drift augmentation; D2O-guarded; **OPTIONAL** — no checkpoint → off). CLI
> `python -m nmr_api.pscnn --mixtures 20000 --epochs 200 --grid 4096` trains + saves. **3-way
> comparison** (`track1_benchmark --compare`): EASY (clean) deterministic F1 **0.96** > pSCNN
> 0.71; HARD (0.04-ppm drift) deterministic **collapses to 0.24** while pSCNN holds **0.55**
> and **hybrid 0.59** — the honest value case for a learned channel (robustness to drift), NOT
> a replacement. **Held-out real datasets sourced + verified** (docs §9): **BMRB** (~1,100 D₂O
> standards, peak-list endpoints), **GISSMO** (1,286 sim compounds — also F8 data), **MTBLS1**
> (real spectra + identified-metabolite MAF). **F8 GISSMO H100 command spec** in docs §9
> (trainer still to build). Caveat everywhere: F1 numbers are **synthetic/in-distribution, not
> clinical**; real BMRB/GISSMO held-out is the decisive test. **68 tests pass.** NEXT: F8
> trainer, F3 on-VM fine-tune loader, real held-out validation.

> **2026-07-02 — Real held-out GISSMO validation (DONE).** New `external_reference.py` fetches
> physically-exact ¹H shifts from **GISSMO** (spin systems fit to experimental BMRB; verified
> `GISSMO_IDS` map, cached). `track1_benchmark.run_real_validation` (`--validate-real`) is
> HELD-OUT by construction: library + pSCNN training use our HMDB shifts; TEST spectra use the
> INDEPENDENT GISSMO real shifts (17 compounds incl. all BCAAs + aromatic AAs). **Result: on
> real shifts the deterministic matcher COLLAPSES (F1 0.16, recall 0.12), pSCNN ~doubles it
> (0.33), hybrid is best (0.40)** — the pSCNN's value proven on REAL, not just synthetic drift;
> clean narrative = deterministic best on clean, pSCNN helps under real drift/noise, hybrid is
> the practical winner (docs §9). Low absolutes = honest sim-to-real gap (simplified library is
> the bottleneck) → motivates F8 (train on real GISSMO patterns). **70 tests pass.** NEXT: F8
> GISSMO H100 trainer, F3 on-VM fine-tune loader.

> **2026-07-02 — F8 GISSMO transformer quantifier (BUILT + runnable).** `gissmo_corpus.py`
> (`build_corpus` fetches GISSMO ¹H shifts off-VM → bundled `open_data/gissmo_corpus.json`,
> **94 compounds**; `load_corpus`; on-the-fly `simulate_batch` with concentration labels +
> ppm-drift aug), `quantifier.py` (Conv patch-embed → Transformer encoder → softplus per-compound
> relative concentration; identity = conc>threshold; OPTIONAL, no checkpoint → off), and
> `train_on_h100.py --supervised gissmo-quant` (config-tagged checkpoints, no download on node).
> Verified end-to-end (loss ↓, recovers planted compounds). **The exact H100 command + 7-point
> run spec is in `docs/TRACK1_PLAN.md §9`** (`python -m nmr_api.train_on_h100 --supervised
> gissmo-quant --field-mhz <F> --mixtures 250000 --epochs 200 --batch-size 256 --n-bins 4096`;
> outputs `models/gissmo_quantifier*.pt` + report JSON; send those 2 files back). **Honest
> caveat kept:** real held-out F1 still low; F8 closes the sim-to-real gap; its fair eval is on a
> real source it did NOT train on (BMRB/MTBLS1). **71 tests pass.** NEXT: run F8 on H100 → eval on
> BMRB/MTBLS1; F3 on-VM fine-tune loader.

> **2026-07-03 — Track 2 UI ship-ready polish (profiler.html only).** Rebuilt the biomarker /
> differential / correlation result views into a premium, demo-ready layer while keeping the data
> flow and all four endpoints (`/track2/preview|biomarkers|differential|correlation`) unchanged.
> Added: **hero metric cards** (accent ROC-AUC card with 95% CI + permutation-p + a coloured
> confidence dot [strong/moderate/weak/exploratory], cohort, stable-panel, Q²/classes), a
> **colour-coded confusion matrix** (green diagonal = correct, red off-diagonal = confused, alpha
> scaled by row fraction), **direction pills**, animated staggered table rows, cleaner spacing
> (new `.t2-*` CSS layer), larger/smoother modal, and **back-navigation** between results ↔
> differential ↔ correlation. **Bugs fixed:** `downloadT2Panel` stuffed the entire result JSON
> into an `onclick` attribute (fragile on quotes/size) → now reads a stashed `LAST_T2_RESULT`;
> added **Escape-to-close** (backdrop-click already existed); differential group-mean labels now
> use `class_labels` names. Verified end-to-end on a generated 44-sample × 25-metabolite Table 1
> + Table 2 (metadata): discovery recovered the planted BCAA/aromatic-AA + glucose signature
> (AUC 1.0, perm p 0.0099), all four endpoints 200, all views render with **no console errors**,
> back-nav + Escape + non-significant-row dimming all work. Ready to test on the real data table
> + metadata. No backend/`.py` changes in this task.

> **2026-07-02 — Track-1 required-functions map.** New **`docs/TRACK1_REQUIRED_FUNCTIONS.md`**
> re-frames Track 1 by the **five capabilities the spec actually asks for** (not by code
> modules): **1. Compound Classification · 2. Pattern Recognition / ML · 3. Feature Selection ·
> 4. Biomarker Discovery · 5. Automated Workflow Development.** Each section gives meaning,
> input→software→output→next-step data flow, which modules/functions support it, what's built
> vs missing, the Phenome-rubric value (Technical/Innovation/Impact/Bioinformatics/Biology/
> Reproducibility), and honesty caveats — grounded in the Track-1 data context (cleaned ¹H
> spectra, ~20k intensity×ppm features, TSV/CSV, some training annotations, overlapping-signal
> challenge). Ends with a one-row-per-function summary table. **Read this first for "what does
> Track 1 do";** `docs/TRACK1_PLAN.md` remains the strategy/H100 doc.
> **Update:** each function now has a **"Correctness in bioinformatics & chemistry"** block with
> **29 verified literature/database citations** (HMDB, BMRB, GISSMO, Sumner MSI, IUPAC DSS,
> D2O-shake; GISSMO QM sims + chemical-shift-drift physics + NMRQNet/DL-NMR benchmarks; glucose/
> lactate/BCAA assignments + Wang 2011/Newgard 2009 BCAA→T2D + Ambroise 2002 leakage; ASICS/
> BQuant deconvolution + Benjamini-Hochberg + target-decoy FDR; PQN/Beckonert protocol + MSI
> reporting). Citations were gathered then **independently re-fetched by a separate adversarial
> checker** (2 fabricated/mislabeled dropped) and every URL mechanically HTTP-checked live
> (25/26 → 200; the B-H 1995 DOI resolves in-browser only). Links prefer PubMed/PMC/official DB.

> **2026-07-03 — Independent REAL held-out validation (BMRB experimental).** New
> `bmrb_experimental.py` fetches **BMRB metabolomics *experimental* 1D ¹H peak lists** (real
> measured positions + intensities + multiplet structure, DSS-referenced) — a source
> independent of both the HMDB-derived library and GISSMO sims. Three fetch routes (raw
> `peaklist.xml`; set-level `transitions/1H.list`; peak-pick of the reconstructed Bruker
> `1r` spectrum for entries that ship only the binary, e.g. valine/isoleucine). Bundled
> off-VM to `open_data/bmrb_experimental_peaks.json` (**18 compounds** incl. full
> glucose/lactate/alanine + **BCAA triad**). New `track1_benchmark.run_bmrb_validation()` +
> `--validate-bmrb` renders test mixtures from the real peaks and scores 4 methods. **Honest
> result (real data):** permissive `annotate` recall 0.96 / precision 0.49 / **F1 0.64**;
> FDR `deconvolve` precision 0.64 / recall 0.15 / **F1 0.24**; pSCNN **F1 0.42**; **hybrid F1
> 0.53 at precision 0.70** and most robust to a 0.03 ppm referencing offset. Framing:
> permissive over-calls, strict FDR misses most on real spectra, **hybrid is the best
> precision-respecting method** — not clinical. 2 network-free tests added. **73 tests pass.**
> Source data source discovery was a 3-way parallel probe (BMRB/HMDB/MetaboLights); BMRB won.

> **2026-07-03 — Track-1 UI surfacing + end-to-end integration hardening.** **(UI)**
> `profiler.html renderSpectralPipeline` now surfaces the Track-1 identification quality the
> backend already computed but hid: per-metabolite **MSI level badge** + **D₂O-reliability
> badge** (reliable/caution/weak, with the exchangeable-proton caveat on hover) + **📌
> organizer-pin** provenance, an **Identification** header row (MSI standard · D₂O-guarded ·
> pin count), and a new **Feature selection — diagnostic ppm** section (Track-1 fn 3, NCD-
> anchored; e.g. leucine→BCAA/T2D). `_run_cohort_pipeline` now calls `select_diagnostic_ppm`
> (supervised when a full label vector is present, else unsupervised). **(Integration)** New
> `tests/test_integration_pipeline.py` exercises the real `/spectral/pipeline-file` spine
> (load_binned_matrix → extract_embedded_labels → _run_cohort_pipeline) on organizer-style
> inputs: both orientations, `ppm`-suffixed headers, tab/comma, inline label column, and a
> **20k-bin** matrix (must stay <90 s via the F7 downsample). Asserts Track-1 quality fields
> carry through and the result is `jsonable_encoder`-safe. **Bug found + fixed:**
> `extract_embedded_labels` could mistake a continuous ppm-bin column for a label in small
> cohorts (low cardinality) → now requires a label to cover a majority of samples. **78 tests
> pass.** NEXT: run F8 on the H100 (export/train spec below), then wire the trained quantifier
> into `--validate-bmrb`; task 4 (realistic GISSMO lineshapes) still open.

> **2026-07-03 — F8 actually uses the GPU (was a CPU-only bug) + 24× faster data-gen.**
> `quantifier.train` and `pscnn.train` never moved the model/tensors off CPU — the H100 would
> have sat idle while `_device_summary()` merely *reported* CUDA. Fixed: `_pick_device()`
> (cuda→mps→cpu), `model.to(device)`, batches via `torch.as_tensor(..., device=device)`,
> on-device indexing, and TF32 + **bf16 autocast** on CUDA (Hopper speedup); `predict`/`identify`
> run on the model's device. `train_on_h100` now prints the training device and records
> `trained_on_device` in the report. **Data-gen bottleneck fixed:** drift augmentation used to
> recompute each fingerprint's Gaussian sum per peak per step (~640 ms/batch → **34.6 h** data-gen
> for the full run, starving the GPU). A ppm drift is just a translation on a uniform grid, so
> `gissmo_corpus._translate` (one interp) + a per-epoch `build_drift_bank` (O(1) indexed sampling)
> cut it to **~1.45 h** (full run) / **~4 min** (lico1h). Verified on MPS: models train on-device,
> loss ↓, cross-device inference OK. **78 tests pass.** Pack `ruuphenome_f8_lico_pack.tar.gz`
> rebuilt with the fixes. Also: reviewed the user's LiCO Run Script (3-lens workflow) — real risks
> = compute-node internet + cu118-vs-Hopper (use cu124); gave a hardened script that prefers the
> node's preinstalled CUDA torch.

> **2026-07-03 — F8 TRAINED on H100 + quantifier wired into `--validate-bmrb`.** First real run
> completed on an **NVIDIA H100 80GB (sm_90), CUDA confirmed** (`trained_on_device: cuda`):
> lico1h profile (40k mixtures × 60 epochs, 94 compounds, n_bins 4096, batch 256), **528 s**,
> loss **0.038 → 0.0072**, checkpoint `gissmo_quantifier_60ep_cuda_b256.pt` (599 KB). **Honest
> note:** GPU util sat at **8–11%** (3.5/80 GB) the whole run — CPU mixture-generation is the
> bottleneck, so the H100 is under-used; fine for lico1h, matters if scaling to the full run
> (moving data-gen onto the GPU is the fix — offered, not yet built). **Wiring:** `run_bmrb_validation`
> now scores the trained GISSMO quantifier + a `hybrid+quant` channel on the real BMRB held-out set
> it never trained on — loads ONLY if `models/gissmo_quantifier.pt` is present (else skipped, so the
> benchmark still runs). Added `_canonical()` to map the quantifier's 94 GISSMO names to the panel's
> library keys (verified **17/18 panel compounds map**; avoids the 'alanine' ⊂ 'phenylalanine'
> collision). CLI prints the quantifier rows + loaded status; report now records
> `steps_per_epoch_actual`. **To evaluate the real checkpoint:** drop `gissmo_quantifier.pt` into
> `backend/nmr_api/models/` and run `python -m nmr_api.track1_benchmark --validate-bmrb`.
> **78 tests pass** (wiring guarded by checkpoint presence; throwaway test checkpoint removed so
> `status().trained` stays honest).

> **2026-07-03 — F8 REAL held-out result: it does NOT help (honest negative).** Ran the trained
> H100 checkpoint through `--validate-bmrb` on real BMRB experimental spectra (17/18 vocab mapped).
> Clean condition F1: deterministic-permissive 0.64, deterministic-FDR 0.23, pSCNN 0.43, **hybrid
> 0.48**, **quantifier (F8) 0.13** (P 0.21 / R 0.10), **hybrid+quant 0.49** (P 0.52 / R 0.48).
> **Verdict:** the GISSMO-trained quantifier is the WORST standalone method and does not generalize
> to real spectra; `hybrid+quant` only matches `hybrid` and does so by trading precision for recall
> (wrong direction for an ID table). **F8 did NOT close the sim-to-real gap.** Root cause: it trains
> on fixed-width single-Gaussian GISSMO fingerprints (no J-multiplets / real intensities) that real
> BMRB spectra have. **Decision (per user's "validate-first" plan): do NOT scale the run or build
> GPU data-gen — it isn't warranted.** The practical winner remains **hybrid = deterministic +
> pSCNN**. F8 is now an honest negative exhibit (rigorous held-out test → clear negative → reported
> plainly), which strengthens the scientific-honesty / bioinformatics-correctness story rather than
> a headline claim. If F8 is ever revisited, realistic lineshapes (task 4: J-multiplets + field-
> dependent width) is the only thing that could plausibly close the gap — but only worth it with a
> reason to believe it will pay off.

> **2026-07-03 — Track-1 prototype-ready sprint (fixed the audit's blockers).** A 7-stream code
> audit found real gaps; fixed the P0/P1 + key P2:
> • **P0 `annotate()` was broken on real data** — the occupancy gate was a relative 0.75-quantile
>   that called ~25% of bins occupied on ANY input (verified: **103/578 metabolites on pure noise,
>   575/578 on a sparse spectrum**). Replaced with an **absolute noise floor** (`_occupancy_floor`:
>   median + 5·1.4826·MAD, frac-of-max fallback for degenerate/cleaned baselines). Now **0/578 on
>   noise**, recovers only planted compounds on sparse. Added `AnnotateNoiseFloorTests`.
> • **P0 the hybrid winner was benchmark-only** — pSCNN was imported nowhere in `main.py`, no
>   checkpoint existed, and the app's "hybrid" was a different (off) nmrformer thing. Now:
>   `pscnn.default_panel()` (30 common metabolites), `pscnn.identify_cohort()`, **wired into
>   `main._run_cohort_pipeline`** as `out["identification"]` (deterministic FDR core ∪ pSCNN@0.6 =
>   hybrid), a **persisted checkpoint** `models/pscnn_identifier.pt` via new `train_on_h100.py
>   --supervised pscnn`, surfaced in `/plugins` (`track1_identification`) and in the profiler UI
>   (Identification-method section + pSCNN✓ badge). The honest hybrid is now reachable from the app.
> • **P1 reproducibility** — `pscnn.train` now calls `torch.manual_seed` → `--validate-bmrb` returns
>   identical F1 across runs (was 0.41 vs 0.44).
> • **P1 organizer annotations in the UI** — `/spectral/pipeline-file` takes an optional
>   `identified_peaks` file; a "＋ Attach organizer annotations" affordance threads it to the
>   authoritative-pin path (verified: a non-library pinned compound is forced present with
>   `provenance:organizer_pin`). The competition's headline feature is now demoable.
> • **P2 concentration bridge** — pipeline now feeds the NNLS/FDR **concentration** matrix (not the
>   raw bin-mean abundance) into discovery (`discovery_matrix_source`). Diagnostic-ppm goes
>   supervised on the **labeled subset** (was all-or-nothing). Low-fit-R² UI caveat added.
> • **F8 honesty** — demoted from "hero" to honest-negative across `quantifier.py` docstring +
>   `status()`, `TRACK1_REQUIRED_FUNCTIONS.md`; `select_diagnostic_ppm` "leakage-safe" wording
>   corrected to "display-only, whole-cohort".
> **80 tests pass.** DEFERRED (low-risk P2, documented not done): decoy-wraparound in the FDR null,
> confidence propagation into discovery, `annotate` dead `bin_ppm` arg, pH-aware tol, selector unit
> tests. NEXT real ML: F3 on-VM fine-tune on the competition's provided annotations.

> **2026-07-03 — Track-2 input safety + F3 loader prepped.** **(Safety)** `deconvolve`'s
> concentration matrix contains ALL NNLS-fitted targets, not just FDR-passing — so
> `_run_cohort_pipeline` now feeds Track 2 ONLY the **FDR-confirmed** concentration columns
> (`discovery_input=nnls_fdr_confirmed`), falling back to annotate abundance only when nothing
> passes FDR. Added a `track1_quality` block (`confidence high/low` from n_fdr_confirmed<3 or
> fit_r2<0.3), a UI **low-confidence banner** ("results are EXPLORATORY"), and a
> `low_confidence_warning` on the biomarkers object — so biomarker/pathway outputs are never
> presented as high-confidence when Track-1 quality is poor (e.g. coarse/transformed inputs).
> **(F3)** New `finetune_loader.py` — on-VM fine-tune path for organizer annotations:
> `load_annotation_panel` (parses {ppm,metabolite} OR {metabolite,shifts}), `build_finetune_panel`,
> `finetune_pscnn` (**gated on NMR_OFFLINE=1**, local read/write, **0 network imports**, saves to a
> chosen path so the serve checkpoint isn't clobbered). Ready-to-use, **not trained**, no closed
> data used. `pscnn.save_checkpoint` gained an optional `path`. **84 tests pass** (+4 F3). Final
> readiness verification all green (annotate 0 on noise / 2 on sparse; hybrid wired + clear
> fallback when checkpoint absent; Track-2 gets FDR-confirmed only + low-confidence warnings;
> `--validate-bmrb` hybrid F1 0.50–0.535).

> **2026-07-03 — Searchable/filterable metabolite browser (profiler.html).** When the pipeline
> modal's annotation (up to 578) or quantification list is long, a "🔍 Search & filter all N …→"
> button opens an in-modal browser: live name search + filter checkboxes (annotation: D₂O-reliable
> only / organizer-pins only; quantification: passes-FDR only), a live "showing X of N" count, and
> a "← Back to results" button that re-renders the summary (back-and-forth, no re-fetch). Result
> stashed in `PIPELINE_VIEW`; reusable row builders `_annRowHtml`/`_quantRowHtml`; `openMetaboliteBrowser`
> + `filterBrowser`. Verified on the demo (225 annotated → "glut" filters to 8, D₂O/pin filters +
> back-nav work, no console errors). Track-2 tables can get the same pattern on request.

> **2026-07-03 — Interactive metabolite correlation network (GGM UI).** Replaced the
> static `ggmNetworkSvg(edges)` render with `initGgmNetwork(allEdges)`: a force-directed
> layout (repulsion + spring + centering + damping, `requestAnimationFrame`-driven),
> **draggable nodes**, **hover-to-trace** (hovering a node dims non-neighbor nodes/edges
> so its own links stay legible), a **live "Min |partial r|" slider** that re-filters
> edges in place, and an r/q **tooltip** per edge. `correlationNetwork` now calls
> `GET /biomarkers-correlation?dataset=...&r_threshold=0.05` (new endpoint in `main.py`:
> resolves the dataset TSV → `biomarkers.build_matrix` → `correlation.analyze`) and
> renders the shell (`#ggm-thr` slider, `#ggm-host`/`#ggm-svg`/`#ggm-tip`) before wiring
> the sim. Verified in-browser on `mtbls1` (12 direct edges; strongest creatine–
> creatinine r=0.97): hover dims non-neighbors correctly, slider changes edge count live
> (10 edges @ r≥0.15 → 3 @ r≥0.9), drag works, force loop is armed (only visually static
> in a backgrounded preview tab due to rAF throttling — confirmed via `_ggmRAF`), no
> console errors.

> **2026-07-04 — Full verified bibliography (`docs/REFERENCES.md`, new).** Harvested
> every academic/method citation across `backend/nmr_api/*.py`, `docs/*.md` and
> `static/profiler.html` via a multi-agent workflow, then **web-verified each one**
> (real DOI/PMID + does the paper's actual content support how RuuPhenome uses it) —
> 111 of 115 raw citations confirmed real (the other 4 are textbook statistical
> primitives, not literature citations). Result: **~60 distinct references**, organized
> by what they ground — §A signal processing (Dieterle PQN, Eilers AsLS, ACME phase,
> MAD noise), §B identification/quantification (NNLS, Elias & Gygi target-decoy FDR,
> Sumner MSI levels, Wei 2022 pSCNN), §C reference data (HMDB 5.0, BMRB, GISSMO,
> MetaboLights/Workbench), §D the Track-2 **leakage-safe methodology** (Ambroise &
> McLachlan 2002, Vabalas 2019, Diaz-Uriarte 2022, Benjamini-Hochberg, the
> bootstrap/permutation-CI literature, the full classifier-suite citations), §E the
> **GGM** (Krumsiek 2011, Ledoit-Wolf, Fisher), §F PCA/UMAP, a new **§G "Biological
> interpretation" section** explicitly grounding the NMR→biology step
> (`biology.interpret_panel()`: HMDB cards + MetaboAnalyst/MetPA-style hypergeometric
> pathway enrichment + KEGG + BH-FDR — this layer existed in code but had no citation
> section before), §H the clinical/epidemiology grounding (Newgard 2009, Wang 2011,
> Würtz 2015, Cheng 2012, Thai NCD-burden papers), and §I reproducibility standards
> (Sandve 2013, FAIR, Model Cards). **3 citation corrections were flagged but are NOT
> yet applied to code/docs/UI** — see "Pending citation corrections" below.

> **2026-07-04 — Pre-ship bug hunt (adversarially verified) + all fixes applied.** A
> workflow fanned 11 finders across every backend module cluster + the UI, then ran an
> independent high-effort skeptic against every finding whose job was to REFUTE it
> before it counted (default-to-refuted unless a concrete real trigger is confirmed
> against the actual code). 19 raw findings → **9 survived** (7 confirmed, 2 plausible;
> 10 refuted as unreachable-behind-an-existing-guard or a misread — see below). Plus one
> found by direct smoke-test: `pytest` failed to **collect at all** when run from
> `backend/` (all 16 test files import `backend.nmr_api.*`, and there was no
> `conftest.py`/rootdir pin anywhere in the repo). **All 10 fixed and verified:**
>
> 1. **New `ruuphenome/pytest.ini`** (`pythonpath = .`, `testpaths`) — `pytest` now
>    collects and passes from either `ruuphenome/` or `backend/nmr_api/`.
> 2. `spectral_cohort.deconvolve()` — guarded the decoy-shift modulus (`decoy_span`)
>    against `ppm_max <= 0` (was an unhandled `ZeroDivisionError` on an all-upfield or
>    lone-0-ppm-column upload).
> 3. `spectral_cohort.deconvolve()` — **target-decoy FDR selection is now monotone**:
>    accepts the deepest-passing-threshold prefix instead of accepting/rejecting each
>    concentration index independently, which could previously reject a strong
>    metabolite while accepting a weaker one below it ("holes" in the accepted set) when
>    a decoy happened to land on a real occupied peak.
> 4. `biomarkers.py` minimal-panel selection now carries the **integer column index**
>    through ranking instead of re-resolving the column by name afterward — two blank/
>    `unknown` metabolite names no longer collapse the panel onto one duplicated column
>    (was inflating the reported panel AUC and silently dropping a real second marker).
> 5. `dimensionality.project()` — `explained_variance_ratio_` is now `nan_to_num`'d
>    before rounding; a zero-variance (constant) cohort used to return `NaN` percentages,
>    which Starlette's JSON renderer rejects outright → unhandled HTTP 500.
> 6. `main.py` **`/profile/report` re-triage bug**: the report endpoint used to re-bucket
>    the stored auto-profile with **default** thresholds (hi=0.85/lo=0.5) instead of the
>    thresholds the user actually profiled with, silently flipping e.g. an accepted call
>    to "review" — and `profile_workflow.triage()` mutated the shared stored result
>    objects in place while doing it. Fixed both: `_PROFILE_STATE["auto_thresholds"]`
>    now threads the real hi/lo into the report, and `triage()` no longer writes
>    `result.status` (it buckets into accept/review/reject locally instead).
> 7. `library.build_library()` / `pipeline.analyze()` — guarded the `df_meta["smiles"]`
>    column access; a compound-upload table without a `smiles` column used to raise an
>    uncaught `KeyError` (surfaced to the user as a cryptic HTTP 422).
> 8. `static/profiler.html` `stepCompound()` — early-returns on an empty compound list;
>    arrow-key compound stepping used to compute `i % 0` → `NaN` → an `undefined` deref
>    on any dataset that yields zero compounds.
> 9. `spectral_cohort.total_area_normalize()` — guards a non-finite scale factor; an
>    all-zero/all-NaN cohort used to normalize to all-`NaN` (→ HTTP 500 on serialize).
> 10. `signal_processing.estimate_noise()` — floors σ at `1e-4 × peak amplitude`; a
>     clean/synthetic near-noise-free spectrum used to report **SNR in the millions**
>     (σ→0 hit only a `1e-12` floor); now caps at a believable "excellent" ~10,000 and
>     never binds on a real experimental spectrum (real ¹H noise is always ≳0.01% of
>     peak height).
>
> **Verification:** app imports cleanly (53 routes); **111/111 tests pass** from both
> `ruuphenome/` and `backend/nmr_api/`; each fix was re-checked against its exact
> failure scenario (ppm_max=0 no longer throws; all-zero cohort normalizes finite;
> duplicate-name panel keeps 2 distinct columns; constant-cohort PCA returns `0.0%` not
> NaN; report status no longer mutates on re-triage; empty-compound arrow-stepping
> verified in-browser with no console errors; clean-spectrum SNR reads 10,000 not 7.2M).
> **3 latent-but-currently-unreachable issues were found and intentionally left AS-IS**
> (each is already prevented by an upstream guard today, so fixing them is optional
> hardening, not urgent — flag if a new endpoint bypasses the guard):
> `differential_analysis` raises `IndexError` on <2 classes, but every real caller
> enforces ≥2 classes first; `biomarker_engine.discover` crashes on a single-patient-
> group class, but every user-upload endpoint wraps `discover()` in try/except → HTTP
> 422 (and all 5 bundled datasets have ≥7 patient-groups per class); the GGM force-sim's
> `requestAnimationFrame` isn't cancelled on modal-close (self-terminates in ~4s writing
> to already-detached SVG nodes — no crash, no user-visible artifact, no lasting leak).
>
> **Pending citation corrections** (flagged in `docs/REFERENCES.md`, not yet applied to
> code/UI — small, low-risk text edits): (a) the conditions-form help text in
> `profiler.html` (~line 3532) mislabels a citation as "Everett 2016 (IUPAC)" — the
> paper is actually **Dona et al. 2016** (Everett is a co-author, not first; it's a
> metabolite-ID guide, not the IUPAC shift-convention paper — that's Harris 2008); (b)
> "Ahola-Olli 2019" is sourced from **four Finnish cohorts**, not UK Biobank — fix any
> "UK Biobank" phrasing tied to that citation; (c) Hand & Till 2001 defines one-vs-
> **one** multi-class AUC but `model_suite.py` uses one-vs-**rest**
> (`roc_auc_score(multi_class='ovr')`) — either reword the citation ("in the spirit of")
> or switch to `'ovo'` to match the paper exactly.

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
- `POST /track2/biomarkers` — two-table discovery (full metrics + confusion matrix + top-1/3/5/10 panel sweep; `multiclass` form flag preserves ≥3 groups)
- `POST /track2/preview`
- `POST /track2/metadata-columns`
- `POST /track2/discover-with-metadata`
- `POST /track2/differential` — whole-matrix differential (Mann-Whitney+Welch for 2 groups; Kruskal-Wallis+ANOVA for >2; BH q-values; volcano)
- `POST /track2/correlation` — pairwise (Pearson/Spearman, FDR) + partial-correlation Gaussian Graphical Model network (Krumsiek 2011); optional metabolite-vs-covariate correlation
- `GET /biology`
- `GET /enrich-names`

### Laboratory workflow

- `GET /laboratory-workflow`
- `POST /laboratory-workflow/evaluate-qc`

## Known gaps and bugs — do not overclaim

Split by the kind of judgment needed to fix it: **bioinformatics/domain-science**
gaps need NMR/metabolomics chemistry knowledge (solvent behavior, referencing,
biological validity); **tech/engineering** gaps are software correctness or
missing infrastructure that don't require chemistry judgment to fix.

### Bioinformatics / domain-science gaps

1. **Solvent mismatch risk: reference shifts vs. real D2O blood/serum samples.**  
   The competition matrix is blood, run in D2O. `REFERENCE_SHIFTS`
   (`spectral_cohort.py`, 578 entries) is sourced from **HMDB 5.0**, which is
   biofluid-appropriate (HMDB's own reference spectra are conventionally
   aqueous/D2O near physiological pH) — but **neither this dict nor
   `open_data/bmrb_reference_shifts.json` records solvent/pH/temperature per
   entry**, so no single shift value can currently be audited or trusted with
   certainty. The bigger risk is **NMRformer**: public NMR shift-*prediction*
   training corpora skew heavily DMSO-d6/CDCl3 (the default in general organic
   chemistry characterization, not biofluid metabolomics), so an NMRformer
   activated without D2O-specific validation would silently carry that bias.
   NMRformer is confirmed **inactive by default** right now — keep it that way
   for real blood data until it is fine-tuned/validated specifically on D2O
   serum, per gap 5 below.  
   **Fix:** flag/down-weight metabolites whose diagnostic peaks are
   exchangeable protons (COOH/OH/NH/NH₂) — these shift by 0.1–1+ ppm or vanish
   entirely via D/H exchange in D2O regardless of source database. Prefer each
   metabolite's non-exchangeable shifts for D2O matching where available.

2. **pH is a static display label, not an active correction.**  
   The UI shows "pH 7.00" but nothing in the pipeline shifts reference peaks
   for the sample's actual pH. This matters independently of the D2O/DMSO
   question — e.g. histidine's imidazole protons alone move ~0.5 ppm across
   physiological pH swings, and blood pH varies sample to sample.  
   **Fix:** either wire real pH-dependent shift correction for the handful of
   metabolites with known strong pH sensitivity, or stop displaying a pH value
   as if it's already accounted for.

3. **Matching tolerance is fixed and chemistry-blind.**  
   `tol_ppm=0.03` in `spectral_cohort.annotate()` is one flat number for every
   metabolite and every proton type. Aromatic/aliphatic CH shifts are
   genuinely solvent/pH-insensitive at that scale; exchangeable protons are
   not, and nothing currently distinguishes them.  
   **Fix:** per-functional-group tolerance/confidence instead of one constant
   (ties directly into gap 1's exchangeable-proton flagging).

4. **Two metabolite-naming systems that don't reconcile.**  
   Confirmed concretely this session on the same demo dataset: the dashboard
   (Reference Card / compound table) shows names taken verbatim from the
   original MetaboLights file's `metabolite_identification` column
   (`L-alanine`, `L-Lactic acid`, `D-phenylalanine`); Track-1 annotation
   (`REFERENCE_SHIFTS`) uses simplified generic names for the *same molecules*
   (`alanine`, `lactate`, `phenylalanine`). Same compound, different string, in
   two different parts of the same UI.  
   **Fix:** a canonical-name normalization layer (strip `L-`/`D-`/`(R)-`/`(S)-`
   stereo-prefixes, reconcile `"X acid"` ↔ `"X"`, small synonym table) applied
   everywhere metabolite names are displayed or compared.

5. **NCD-relevance match/no-match inherits the naming gap.**  
   `_ncd_relevance()` in `main.py` matches by case-insensitive exact string
   only. A real biomarker overlap (e.g. `lactate` vs `L-Lactic acid`) can be
   silently reported as "✗ not found" purely from naming-scheme mismatch, not
   because the biomarker is actually absent — this undermines the honesty the
   rest of the NCD panel works to maintain.  
   **Fix:** route through the same canonical-name layer once gap 4 is built.

6. **Annotation is currently over-permissive.**  
   The demo annotates 268 metabolites from the 578-entry reference library.
   Useful for showing coverage, but real lab claims need stricter scoring,
   duplicate/synonym collapsing (overlaps with gap 4), ambiguity handling,
   mixture validation and comparison against manual/Chenomx-reviewed truth.

7. **Concentration export is not yet laboratory-validated.**  
   NNLS deconvolution and CSV export work, and the UI labels the table in µM
   when internal-standard calibration is present — but internal-standard
   (DSS/TSP) referencing only corrects field/frequency calibration drift, it
   does **not** correct for solvent-induced shift differences between
   molecules (a separate problem from gap 1). Treat outputs as Chenomx-style
   estimates until validated with standards and manual review on the target
   instrument/matrix.

8. **NMRformer is bundled but not a free pass.**  
   The files and adapter exist and ship pre-trained (~97.8% on 72 classes),
   but direct startup does not activate it by default. Even when active, use
   it as supporting evidence only, until target-matrix (D2O blood) validation
   proves it improves assignments safely — see gap 1.

9. **PCA/UMAP are exploratory, not predictive evidence.**  
   Full-cohort PCA/UMAP separation is not classifier performance. Predictive
   metrics must keep imputation, scaling, feature selection and PCA inside
   training folds (the codebase already does this correctly for AUC/discovery
   — just don't let a PCA plot substitute for it in a pitch).

10. **Clinical/disease claims are limited by labels.**  
   MTBLS242 is longitudinal surgery time points, not disease versus control.
   MTBLS1, MTBLS424 and MTBLS356 provide labeled demos, but final clinical
   claims need independent validation. Name each cohort's disease as it
   actually is — e.g. MTBLS356 is antiphospholipid syndrome (a thrombotic
   vascular disease), used as the cardiovascular panel slot, not a general
   ischemic-heart-disease cohort. A true AMI cohort (MTBLS395) exists but its
   outcome labels are ethically withheld and cannot be used. See "How to add a
   new NCD cohort".

### Tech / engineering gaps

1. **No metadata schema for reference-library provenance.**  
   Neither `REFERENCE_SHIFTS` (hardcoded dict in `spectral_cohort.py`) nor
   `open_data/bmrb_reference_shifts.json` records solvent/pH/temperature per
   shift value — flat `{name: [ppm, ppm, ...]}` only. This is the missing data
   model blocking bioinformatics gap 1 above.  
   **Fix:** extend both to `{name: [{"shift": ppm, "solvent": ..., "pH": ...,
   "temp_k": ..., "source": ...}, ...]}` (or a parallel metadata file keyed the
   same way), backfilling from BMRB's real experimental metadata where the
   entry originated there.

2. **In-memory-only caching, lost on every restart.**  
   `_NCD_CACHE` (and any future persisted-model cache) lives only in the
   running process. A server restart silently re-triggers the ~1 minute
   `/ncd-screen` recompute on next call — the progress bar covers this UX-wise,
   but it's worth knowing. Not urgent for a hackathon demo; note if it becomes
   real friction.

3. **Inline numeric labels are missed.**  
   `spectral_cohort.extract_embedded_labels()` currently skips numeric
   columns, so `Class = 0/1` is ignored even when the column name is clearly a
   label. String labels such as `control/case` work. Fix by allowing numeric
   label columns when the column name matches a label synonym or when
   cardinality is small and the values are not ppm bins.

4. **The strongest cohort pipeline assumes preprocessed/binned data.**  
   It matches the workshop request for binned NMR peak/spectral files. It is
   not yet a full raw multi-sample FID → alignment → binning production
   pipeline. Single-spectrum raw processing exists, but batch raw cohort
   alignment/binning is still a separate future feature.

5. **NMRformer startup/activation isn't clarified in docs/tests.**  
   Separate from the D2O validation question (bioinformatics gap 1) — this is
   purely "does `NMRFORMER_ADAPTER_MODULE` + the bundled adapter actually
   wire up correctly." Add a small integration smoke test for when the adapter
   is active, and document the exact env/config needed to activate it.

6. **The laboratory workflow is RUO design only.**  
   There is no full LIMS, authentication, electronic signature or durable
   append-only audit store.

7. **Frontend is a large single-file app.**  
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

Priority order — the top 3 are the highest-leverage fixes surfaced by the D2O
solvent-mismatch review (see "Known gaps and bugs" above for full detail on
each):

1. **[Bioinformatics] Flag exchangeable-proton-dependent metabolites** for D2O
   matching (gap 1) — the single most consequential fix for real blood-sample
   correctness. Requires adding solvent/exchangeability metadata (tech gap 1)
   to `REFERENCE_SHIFTS` first.
2. **[Bioinformatics] Build the canonical-name normalization layer** (gap 4) —
   fixes both the dashboard/Track-1 display mismatch and the NCD-relevance
   matching honesty gap (gap 5) in one change.
3. **[Tech] Extend the reference-shift data model** to carry per-entry
   solvent/pH/temperature/exchangeability (tech gap 1) — this unblocks #1 and
   makes gap 1 auditable instead of just documented as a risk.
4. Fix numeric inline label detection in `spectral_cohort.extract_embedded_labels`
   (tech gap 3).
5. Tighten Track 1 annotation: synonym collapsing (overlaps with #2 above),
   minimum unique resonances, ambiguity scoring, matrix-specific exclusions and
   external validation (bioinformatics gap 6).
6. Test `/spectral/pipeline` and `/spectral/pipeline-file` on the actual
   organizer binned files and metadata, then record the expected file format.
7. Validate quantification against internal standards/manual Chenomx-style
   review before making strong concentration claims (bioinformatics gap 7).
8. Add a true raw cohort path if needed: batch import → common ppm grid →
   alignment → water/artifact masking → adaptive/fixed binning → matrix export
   (tech gap 4).
9. Clarify NMRformer startup in docs/tests (tech gap 5) — keep it inactive for
   real blood data until D2O-specific validation (bioinformatics gap 1) passes.
10. Update this handoff after any major code or validation change.

## Git / repository state — read before committing or pushing

- **Remote:** `origin` → `https://github.com/chikenpop8/ruuphenome_bdikku2026.git`
  (same URL for fetch and push).
- **Current branch:** `feat/open-data-corpus-builder`. Last commit on it:
  `9041556 feat(models): commit pSCNN checkpoint so a clean clone reproduces the hybrid`.
- **Nothing described in this handoff since that commit has been committed or pushed.**
  Every changelog entry above from "F8 GISSMO transformer quantifier" onward through
  today's bug-hunt fixes is a **single large uncommitted working-tree diff** — check
  `git status --short` and `git diff --stat` before doing anything destructive. Expect
  ~15 modified tracked files and ~40+ new untracked files (new modules under
  `backend/nmr_api/`, new `docs/*.md` + SVGs, `docs/REFERENCES.md`, `pytest.ini`,
  new `tests/test_*.py`, `requirements.lock.txt`, `.python-version`).
- **Before committing, run the full suite once more** (now works from either directory
  thanks to the new `pytest.ini`):
  ```bash
  cd "/Applications/Vibing coding/Noom copy cat/ruuphenome"
  KMP_DUPLICATE_LIB_OK=TRUE backend/nmr_api/.venv/bin/python -m pytest -q
  # expect: 111 passed
  ```
- **Check before staging — likely scratch, not meant to ship as-is:** four untracked
  items look like redundant exports of the same H100/LiCO training-pack bundle:
  `newruuf8/`, `ruuphenome_f8_lico_pack/`, `ruuphenome_f8_lico_pack 2/` (note the space
  in the name), and `ruuphenome_f8_lico_pack.tar.gz` (each ~150–620 KB; `newruuf8/` and
  `ruuphenome_f8_lico_pack/` are near-identical file listings). Don't blanket
  `git add -A` without checking — confirm with the user which (if any) of these should
  be committed, gitignored, or deleted, rather than shipping 3–4 copies of the same pack.
- **`chulaemail.md`** (untracked, repo root) — a second, teammate-facing onboarding doc
  (zero-context version of this file, apparently meant to be emailed to a collaborator
  at Chulalongkorn University). Not sensitive; fine to commit, just note it duplicates
  parts of this file and may drift out of sync with it over time.
- Never force-push; never rewrite history on `main`. This repo backs a hackathon
  submission — prefer opening/updating a PR over pushing straight to a shared branch
  if the grading process expects a specific PR.

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
