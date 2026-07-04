# RuuPhenome — The WOW Feature: "The Honest AUC"

> **The one thing to talk about:** RuuPhenome is the rare metabolomics tool whose
> headline accuracy number is one you can *trust* — because it is measured the way
> that makes cheating impossible, and because it **re-discovered biomarkers that
> landmark clinical studies already found**, without ever inflating its own score.
>
> *Companion evidence file: [IMPACT_AND_VALIDATION.md](IMPACT_AND_VALIDATION.md) — every figure below is cited there. RUO (research use only); not a diagnostic device.*

---

## 1. What the WOW feature is

Almost every metabolomics "AI" demo reports a big AUC (0.95, 0.99…). Almost all of
them are **inflated by data leakage** — the model quietly peeks at the test data
during feature selection, so the number looks amazing and means nothing. In a
country with **~1.6–2.1 million *undiagnosed* diabetics** (NHES 7 / IDF 2024, see
IMPACT doc §1), a screening tool that *lies about its own accuracy* is worse than
useless — it's dangerous.

RuuPhenome's WOW feature is the opposite: a **leakage-safe, false-discovery-
controlled discovery engine** where the reported number is the honest one.

Two guarantees run end-to-end:

1. **Identification is false-discovery-controlled** — every metabolite call must
   beat a *decoy* (a shuffled/shifted fake) at a chosen FDR, so noise cannot
   masquerade as signal. This is the **target–decoy** principle that made
   proteomics trustworthy, brought to ¹H-NMR (`spectral_cohort.deconvolve`).
2. **Discovery is leakage-safe** — feature selection, PCA, and scaling are all
   fitted **inside each cross-validation fold only**, patients are never split
   across train/test, and we additionally report a **permutation p-value**, a
   **bootstrap 95% CI**, and — the honest flex — the **leaky AUC we would have
   gotten if we cheated**, so the inflation gap is visible
   (`biomarker_engine.discover`, `model_suite.compare_models`).

**The demonstration that makes judges lean forward:** on a real type-2-diabetes
urine cohort (MTBLS1, n=132), the engine — with *no hints* — independently selected
**isoleucine and 2-oxoisovalerate** as top biomarkers, at **honest AUC 0.932
(95% CI 0.90–0.98), permutation p = 0.005.** Those are *exactly* the branched-chain
amino-acid markers that large prospective human studies tie to future diabetes
(Würtz 2015, *Circulation*; Cheng 2012, *Circulation*). **We re-derived known
science from scratch — and our accuracy number survived every anti-cheating check.**

---

## 2. Why the impact is huge

- **It fixes the field's most common lie.** Selection-bias leakage is the single
  largest source of over-optimistic biomarker results (Vabalas 2019; Diaz-Uriarte
  2022). Most hackathon/industry demos have it; RuuPhenome quantifies and removes
  it. A judge who knows metabolomics will recognize this immediately.
- **It makes NCD screening *deployable*.** Thailand's diabetes/hypertension burden
  is enormous and largely undiagnosed. A trustworthy, **on-premise, open-source**
  screen (no cloud, no per-sample license vs Chenomx) is exactly what a low-resource
  public-health setting can actually adopt.
- **It is auditable.** Because the number is honest, an expert can *check* it — and
  checking is the whole point of science. The tool is defensible, not just flashy.

---

## 3. Why it is ถูกต้อง · แม่นยำ · น่าเชื่อถือ (correct · precise · reliable)

### ✅ ความสมเหตุสมผลทางวิทยาศาสตร์ — Scientifically sound method
Every step is the *textbook-correct* choice, not an ad-hoc one, and each is
literature-backed:
- **In-fold feature selection** (never on the full dataset) → prevents selection-bias
  leakage — Diaz-Uriarte 2022 (*PLOS Comput Biol*, Tip 6); Vabalas 2019 (*PLOS ONE*).
- **Patient-grouped `StratifiedGroupKFold`** → the same patient never appears in
  train *and* test (critical for repeated-measures cohorts).
- **Target–decoy FDR** for identification → a real statistical false-discovery
  guarantee, not an arbitrary intensity cutoff — Elias & Gygi 2007 (*Nat Methods*).
- **PQN normalization** (Dieterle 2006), **NNLS/ASICS-style deconvolution** to
  un-mix overlapping peaks, **partial-correlation (GGM) networks** (Krumsiek 2011),
  **BH-FDR** on pathway/differential tests (Benjamini–Hochberg 1995).
- **MSI Level-2 honesty** — every ID is labelled "putatively annotated" (library
  match, no in-house standard); we never claim Level 1 (Sumner 2007).

### 💪 ผลลัพธ์ต้องหนักแน่น — Results are robust, not fragile
- **Honest AUC + 95% CI + permutation p** on every discovery (5 repeats,
  200 permutations, group-level bootstrap CI): T2D **0.932 (0.90–0.98), p=0.005**;
  ME/CFS serum **0.742**, urine **0.720**, both p=0.015.
- **Leaky ≤ honest** — in the ME/CFS and T1D-duration cohorts the leakage inflation
  is **≈ 0.000–0.0003**, i.e. *zero* optimism. A model that can't inflate its own
  score is a model you can trust.
- **Honesty exhibits kept on purpose.** MTBLS424 (breast-cancer relapse, n=590)
  reports **AUC 0.57** — a weak, un-inflated floor. A tool that reports 0.57 when the
  signal is weak is *far* more credible than one that always reports 0.95.

### 🔁 ทำซ้ำแล้วผลต้องเสถียร — Reproducible & stable on re-run
- **Fixed random seeds throughout**, **pinned `requirements.lock.txt` + `.python-version`**,
  one-command external-validation & benchmark scripts (Sandve 2013 reproducibility rules).
- **Config-tagged model checkpoints** so a re-train never silently overwrites, and
  `--validate-bmrb` runs deterministically.
- **Automated test suite (111 tests)** guarding the statistics (FDR, nested CV,
  BH-correction, provenance) — re-run and the numbers land the same.

### 📚 อธิบายที่มาที่ไปเทียบกับงานวิจัยเดิมได้ชัดเจน — Traceable & convergent with prior research
- **Convergent validity:** the biomarkers we *discover* match what independent human
  cohort studies *already published* — isoleucine / 2-oxoisovalerate → diabetes
  (Würtz 2015; Cheng 2012); phenylalanine/hypoxanthine → ME/CFS. Independent methods
  landing on the same molecules is the strongest kind of external check.
- **Every number is traceable** to a public MetaboLights/BMRB accession and re-runs
  end-to-end (IMPACT doc §5). Provenance is recorded per run.
- **Held-out validation on *real measured* spectra**, not just synthetic self-tests:
  Track-1 identification is benchmarked on **real BMRB experimental ¹H peak lists**
  (`track1_benchmark.py --validate-bmrb`), and we explicitly flag that in-distribution
  synthetic scores are optimistic — the decisive number is the real held-out one.

---

## 4. The pitch (say this)

**One line:**
> *"Anyone can show you a 0.99 AUC. We show you an AUC that survives every
> anti-cheating test — and to prove the method is real, it re-discovered the exact
> diabetes biomarkers that landmark clinical studies found, from scratch."*

**30-second version:**
> "The dirty secret of metabolomics AI is data leakage — the model peeks at the
> answer, so the accuracy is fake. RuuPhenome is built the opposite way: feature
> selection happens *inside* each validation fold, patients are never split, and we
> even print the inflated number we'd get if we cheated, so you can see the gap. The
> proof it works? On a real diabetes cohort it independently picked out isoleucine
> and 2-oxoisovalerate — the branched-chain amino acids that *Circulation* papers
> link to future diabetes — at an honest AUC of 0.93. Correct, precise, reproducible,
> and grounded in the literature. That's a screen you can actually trust."

---

## 5. Where to see it / how to reproduce
- **In the app:** run **Discover → biomarker model suite** (shows honest AUC, leaky
  AUC, calibration, per-model, all leakage-safe) and **PCA/UMAP** (exploratory only,
  clearly labelled). Any report's **Step 2** shows FDR-confirmed IDs; **Step 4** the
  diagnostic positions.
- **Reproduce the numbers:** `python -m nmr_api.external_validation` (open cohorts),
  `python -m nmr_api.track1_benchmark --validate-bmrb` (real held-out identification),
  `pytest backend/nmr_api/tests` (111 tests). Fixed seeds → same results every time.

## 6. Honest limitations (keeps it defensible)
Small n on most disease cohorts (54–132) → wide CIs; single-cohort internal CV, not
external prospective validation; identification is MSI Level 2 (no authentic
standard); concentrations are directional (single-Gaussian model), not Chenomx-grade
absolute. **RUO — a screening/triage aid, not a diagnostic device.** Stating these
is *part* of why the tool is น่าเชื่อถือ.

---

## Key references (the science this stands on)
*Full list + per-cohort evidence in [IMPACT_AND_VALIDATION.md](IMPACT_AND_VALIDATION.md) and [IMPACT_PERSONALIZED_MEDICINE.md](IMPACT_PERSONALIZED_MEDICINE.md).*

**The "honest AUC" methodology**
- Diaz-Uriarte et al. (2022), *PLOS Comput Biol* — biomarker-discovery leakage (Tip 6). DOI [10.1371/journal.pcbi.1010357](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010357)
- Vabalas et al. (2019), *PLOS ONE* — ML validation at small n. DOI [10.1371/journal.pone.0224365](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0224365)
- Westerhuis et al. (2008), *Metabolomics* — permutation testing. [link](https://link.springer.com/article/10.1007/s11306-007-0099-6)
- Benjamini & Hochberg (1995), *JRSS-B* — FDR control. DOI [10.1111/j.2517-6161.1995.tb02031.x](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x)

**Identification & deconvolution rigor**
- Elias & Gygi (2007), *Nat Methods* — target–decoy FDR (brought here to NMR). DOI [10.1038/nmeth1019](https://doi.org/10.1038/nmeth1019)
- Tardivel et al. (2017), *Metabolomics* — ASICS NNLS ¹H-NMR deconvolution. DOI [10.1007/s11306-017-1244-5](https://doi.org/10.1007/s11306-017-1244-5)
- Sumner et al. (2007), *Metabolomics* — MSI identification levels. DOI [10.1007/s11306-007-0082-2](https://pmc.ncbi.nlm.nih.gov/articles/PMC3772505/)

**Convergent validity — the biomarkers we re-discovered**
- Würtz et al. (2015), *Circulation* — BCAA/aromatic AAs & cardiometabolic risk. DOI [10.1161/CIRCULATIONAHA.114.013116](https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.114.013116)
- Cheng et al. (2012), *Circulation* — metabolite profiles & cardiometabolic risk. DOI [10.1161/CIRCULATIONAHA.111.067827](https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.111.067827)

**Reproducibility**
- Sandve et al. (2013), *PLOS Comput Biol* — reproducible computational research. DOI [10.1371/journal.pcbi.1003285](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003285)
