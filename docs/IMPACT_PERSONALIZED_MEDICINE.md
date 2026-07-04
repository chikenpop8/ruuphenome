# RuuPhenome — Features That Change Personalized Medicine (Pitch Impact Sheet)

> Mapped to the judging rubric: **Impact (20)**, **Technical (20)**, **Problem & Data (15)**,
> **Pitching (15)**, and the **Phenome track (30): "ถูกต้อง แม่นยำ น่าเชื่อถือ."**
> Evidence: [IMPACT_AND_VALIDATION.md](IMPACT_AND_VALIDATION.md) · [WOW_FEATURE.md](WOW_FEATURE.md).
> RUO — a screening/triage aid, not a diagnostic device.

---

## The thesis (say this first)

> **"Your genome is the blueprint. Your metabolome is *you, right now.*"**

Genomics tells you what *might* happen. The **metabolome** — the small molecules your
body is making, burning, and shedding this minute — is the layer where genes, diet,
drugs, and disease all finally show up. It is the **most personal, most actionable,
most real-time** molecular readout of a human being. That is precisely what
personalized medicine needs — and RuuPhenome reads it **from a single ¹H-NMR scan,
open-source and on-premise.**

Personalized medicine has been stuck behind three walls: **cost** (closed tools like
Chenomx), **trust** (AI metabolomics that inflates its own accuracy), and **access**
(cloud-only pipelines that can't run where the patients are). RuuPhenome knocks down
all three.

---

## The five features that make the impact

### 1. One scan → a personal metabolic fingerprint
**What it does:** from a single, cheap, non-destructive ¹H-NMR spectrum, it identifies
and quantifies *dozens* of metabolites for that one individual — a personal molecular
profile, not a population average.
**Why it changes personalized medicine:** this *is* the substrate of precision
medicine — an individual, dynamic phenotype you can measure again next month and watch
change with treatment. No separate assay per compound; one measurement, a whole profile.
> *Impact / Phenome.*

### 2. Open-source & on-premise → democratized precision metabolomics
**What it does:** runs entirely on a local machine (or the hospital's own server), no
cloud, no per-sample license, no data leaving the building — a free, auditable
alternative to closed commercial profilers.
**Why it changes the world:** it puts precision metabolomics in reach of Thai hospitals,
universities, and every low-and-middle-income setting that could never afford the
closed tools. **Access is the single biggest impact lever** — and keeping data
on-premise is privacy-by-design (PDPA-friendly).
> *Impact (biggest score lever) / Problem & Data.*

### 3. Trustworthy discovery + NCD screening → catching the undiagnosed
**What it does:** matches a person's metabolites to disease signatures and discovers
per-cohort biomarker panels with a **leakage-safe, false-discovery-controlled** engine
— the "honest AUC" that survives every anti-cheating test, and that independently
**re-discovered the known diabetes biomarkers** (isoleucine, 2-oxoisovalerate; honest
AUC 0.93).
**Why it changes the world:** Thailand has **~1.6–2.1 million *undiagnosed* diabetics**
and **~8 million unaware hypertensives** (NHES 7 / IDF 2024). A trustworthy metabolic
screen means early, low-cost triage → personalized follow-up before disease is
symptomatic.
> *Impact / Phenome (แม่นยำ, น่าเชื่อถือ).*

### 4. Confidence & provenance on every result → research you can act on
**What it does:** every call carries its evidence — target–decoy FDR, MSI identification
level, D₂O-reliability, condition-aware chemistry, and full run provenance — plus honest
uncertainty (95% CI, permutation p, and even the *inflated* number you'd get if you
cheated).
**Why it matters:** this is the bridge from "a cool demo" to "a result a clinician or
regulator can audit and believe." Honesty is a feature, not a disclaimer.
> *Technical / Phenome (ถูกต้อง) — reproducible: fixed seeds, pinned lockfile, 111 tests.*

### 5. Biology, not just numbers → mechanism per individual
**What it does:** turns a metabolite list into biology — pathway enrichment, curated
roles, and disease relevance — so a result reads as *"this points to BCAA metabolism /
insulin resistance,"* not just *"compound X is high."*
**Why it changes personalized medicine:** actionable interpretation is what makes a
molecular profile a *decision*, not a data dump.
> *Impact / Phenome (เทียบงานวิจัยเดิม — findings converge with published human cohorts).*

---

## One-slide rubric map (drop straight into the deck)

| Rubric criterion | RuuPhenome's strongest evidence |
|---|---|
| **Problem & Data (15)** | Real, quantified undiagnosed NCD burden in Thailand + open real MetaboLights/BMRB data; the metabolome framed as the *actionable* layer of personalized medicine. |
| **Technical (20)** | Target–decoy FDR identification, leakage-safe nested patient-grouped CV, NNLS deconvolution, pSCNN hybrid, provenance, 111 tests — all reproducible with fixed seeds. |
| **Impact (20)** | Democratized, on-premise precision metabolomics for LMIC NCD screening + individual metabolic profiling — access, trust, and privacy in one open tool. |
| **Pitching (15)** | The "honest AUC" moment: *it re-discovered the known diabetes biomarkers from scratch, and the number survived every anti-cheating test.* |
| **Phenome track (30)** | "ถูกต้อง แม่นยำ น่าเชื่อถือ" — the full honesty stack (see WOW_FEATURE.md): sound methods, robust CIs, reproducible seeds, convergence with prior research. |

---

## Pitch lines (personalized-medicine framing)

- **Hook:** *"Your genome is your blueprint; your metabolome is you, right now — and personalized medicine has never been able to read it cheaply, openly, and honestly. RuuPhenome does."*
- **World impact:** *"We take a tool that costs a fortune and lives in the cloud, and make it free, open, and on-premise — so precision metabolomics can finally run where the patients actually are."*
- **Trust close:** *"And the proof it works: from one real diabetes cohort, it re-discovered the exact biomarkers that Circulation papers found — at an honest accuracy that can't be inflated. Correct, precise, reproducible. That's a phenome tool a country can trust."*

---

## Keep it honest (this *earns* Phenome points, doesn't lose them)
Small cohorts → wide CIs; internal cross-validation, not external prospective trials;
MSI Level-2 identification; concentrations directional, not absolute; **RUO, a
screening aid, not a diagnostic device.** Stating the limits is exactly what
"น่าเชื่อถือ" looks like to expert judges.

---

## What is actually *innovative* here (say this to the Technical judges)
None of the individual methods are magic — that's the point. **The innovation is the
combination**: to our knowledge RuuPhenome is the first **open-source** ¹H-NMR tool to
put a **target–decoy false-discovery guarantee on identification** (a rigor borrowed
from proteomics, Elias & Gygi 2007) *and* a **leakage-safe nested-CV guarantee on
biomarker discovery** (Diaz-Uriarte 2022; Vabalas 2019) in one auditable, on-premise
pipeline — then prove it on **real held-out spectra** with **convergent validity**
against published human cohorts. Established science, assembled in a way no free tool
had before. That is defensible innovation, not hype.

---

## References (all real; every claim is traceable)

**Problem & data — Thailand's NCD burden**
- IDF Diabetes Atlas, 11th ed. (2024), Thailand — https://diabetesatlas.org/data-by-location/country/thailand/
- Aekplakorn et al. (2024), *BMC Public Health* — NHES VI hypertension. DOI [10.1186/s12889-024-20643-1](https://pmc.ncbi.nlm.nih.gov/articles/PMC11562084/)
- Thailand 7th National Health Examination Survey (NHES 7, 2024–25) — undiagnosed diabetes/hypertension estimates.
- WHO SEARO (2023), Thailand hypertension care cascade — https://www.who.int/southeastasia/news/detail/31-05-2023-thailand-improving-hypertension-care-cascade

**Statistical rigor — why the accuracy is trustworthy (ถูกต้อง · น่าเชื่อถือ)**
- Diaz-Uriarte et al. (2022), *PLOS Comput Biol* — ten quick tips for biomarker discovery (leakage). DOI [10.1371/journal.pcbi.1010357](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010357)
- Vabalas et al. (2019), *PLOS ONE* — ML validation with limited data. DOI [10.1371/journal.pone.0224365](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0224365)
- Westerhuis et al. (2008), *Metabolomics* — permutation testing in metabolomic classification. [link](https://link.springer.com/article/10.1007/s11306-007-0099-6)
- Benjamini & Hochberg (1995), *JRSS-B* — FDR control. DOI [10.1111/j.2517-6161.1995.tb02031.x](https://rss.onlinelibrary.wiley.com/doi/10.1111/j.2517-6161.1995.tb02031.x)
- Tsamardinos et al. (2018), *Mach Learn*; LeDell et al. (2015), `cvAUC` — CV-AUC confidence intervals. [Tsamardinos](https://link.springer.com/article/10.1007/s10994-018-5714-4) · [LeDell](https://pubmed.ncbi.nlm.nih.gov/26279737/)
- Bengio & Grandvalet (2004), *JMLR* — no unbiased estimator of K-fold CV variance (honesty caveat). [pdf](https://www.jmlr.org/papers/volume5/grandvalet04a/grandvalet04a.pdf)

**Method foundations — the technical building blocks (innovation)**
- Elias & Gygi (2007), *Nat Methods* 4:207 — target–decoy FDR. DOI [10.1038/nmeth1019](https://doi.org/10.1038/nmeth1019)
- Tardivel et al. (2017), *Metabolomics* 13:109 — **ASICS** NNLS ¹H-NMR deconvolution. DOI [10.1007/s11306-017-1244-5](https://doi.org/10.1007/s11306-017-1244-5)
- Dieterle et al. (2006), *Anal Chem* 78:4281 — **PQN** normalization. DOI [10.1021/ac051632c](https://doi.org/10.1021/ac051632c)
- Krumsiek et al. (2011), *BMC Syst Biol* 5:21 — Gaussian graphical (partial-correlation) metabolite networks. DOI [10.1186/1752-0509-5-21](https://doi.org/10.1186/1752-0509-5-21)
- Koch, Zemel & Salakhutdinov (2015), *ICML Deep Learning Workshop* — Siamese networks (architectural lineage of the pSCNN learned channel).

**Biomarker biology — convergent validity vs prior research (เทียบงานวิจัยเดิม)**
- Würtz et al. (2015), *Circulation* — BCAA/aromatic AAs & incident cardiometabolic risk. DOI [10.1161/CIRCULATIONAHA.114.013116](https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.114.013116)
- Cheng et al. (2012), *Circulation* — metabolite profiles & cardiometabolic risk (Gln/Glu). DOI [10.1161/CIRCULATIONAHA.111.067827](https://www.ahajournals.org/doi/full/10.1161/CIRCULATIONAHA.111.067827)
- Wang et al. (2011), *Nat Med* — 5-amino-acid incident-T2D signature. [link](https://www.nature.com/articles/nm.2307)
- Guasch-Ferré et al. (2016), *Diabetes Care* — BCAA & T2D meta-analysis. [link](https://diabetesjournals.org/care/article/39/5/833/30646)
- Ahola-Olli et al. (2019), *Diabetologia* — circulating metabolites & incident T2D. [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC6861432/)
- Louca et al. (2022), *Metabolites* — BCAAs & hypertension (CoMETS). [link](https://pmc.ncbi.nlm.nih.gov/articles/PMC9324896/)

**Reporting & reproducibility standards (Phenome + trust)**
- Sumner et al. (2007), *Metabolomics* — MSI identification levels. DOI [10.1007/s11306-007-0082-2](https://pmc.ncbi.nlm.nih.gov/articles/PMC3772505/)
- Wilkinson et al. (2016), *Sci Data* — FAIR principles. DOI [10.1038/sdata.2016.18](https://www.nature.com/articles/sdata201618)
- Sandve et al. (2013), *PLOS Comput Biol* — ten rules for reproducible computational research. DOI [10.1371/journal.pcbi.1003285](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1003285)
- Mitchell et al. (2019), *FAT\** — Model Cards for model reporting. DOI [10.1145/3287560.3287596](https://dl.acm.org/doi/10.1145/3287560.3287596)

*The complete, in-line-cited evidence table (per-cohort AUCs, CIs, permutation p) lives in [IMPACT_AND_VALIDATION.md](IMPACT_AND_VALIDATION.md).*
