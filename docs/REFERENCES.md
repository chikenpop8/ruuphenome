# RuuPhenome — References & Scientific Grounding

> Every method, database, and clinical claim in RuuPhenome traces to a published source.
> This file lists each reference, its verified citation (DOI/PMID), and **exactly which part
> of the app it grounds** — code file where relevant. Compiled by harvesting citations across
> the source, docs, and UI, then web-verifying each one is a real publication whose actual
> content matches how we use it. RUO — research use only; not a diagnostic device.
>
> **Legend:** ✅ *implemented* = grounds shipped, running code · 📐 *design/precedent* = a
> method we reimplement or a precedent we cite · 📋 *planning* = cited in planning docs, not
> wired into shipped code.

---

## A · NMR signal processing & spectral preprocessing

| Ref | What it grounds |
|---|---|
| **Dieterle et al. 2006** — Probabilistic Quotient Normalization… *Anal. Chem.* 78(13):4281–4290. [10.1021/ac051632c](https://doi.org/10.1021/ac051632c) | ✅ **PQN normalization** — `pqn_normalize()` (spectral_cohort.py): total-area → median reference spectrum → divide by median per-bin quotient. The default dilution-robust normalize step. |
| **Eilers & Boelens 2005** — Baseline Correction with Asymmetric Least Squares Smoothing. *LUMC tech. report.* | ✅ **AsLS baseline** — `baseline_correct()` (signal_processing.py): asymmetric-weighted Whittaker smoother. |
| **Chen et al. 2002** — Automatic phase correction… entropy minimization. *J. Magn. Reson.* 158:164–168. [10.1016/S1090-7807(02)00069-1](https://doi.org/10.1016/S1090-7807(02)00069-1) | ✅ **ACME auto-phasing** — `auto_phase()` estimates p0/p1 via nmrglue's ACME. |
| **Rousseeuw & Croux 1993** — Alternatives to the Median Absolute Deviation. *JASA* 88(424):1273–1283. [10.1080/01621459.1993.10476408](https://doi.org/10.1080/01621459.1993.10476408) | ✅ **Robust noise σ = 1.4826·MAD** — `estimate_noise()` sets SNR/prominence thresholds and the occupancy floor. |
| **Helmus & Jaroniec 2013** — nmrglue: open-source Python NMR analysis. *J. Biomol. NMR* 55:355–367. [10.1007/s10858-013-9718-x](https://doi.org/10.1007/s10858-013-9718-x) | ✅ **NMR I/O library** — Bruker read, digital-filter removal, phase application (`read_bruker_zip()`). |
| **Harris et al. 2008** — Further conventions for NMR shielding and chemical shifts (IUPAC 2008). *Magn. Reson. Chem.* 46:582–598. [10.1002/mrc.2225](https://doi.org/10.1002/mrc.2225) | ✅ **0-ppm reference convention** — DSS/TSP/TMS anchor logic behind the D₂O guard and conditions form. |
| **IUPAC Recommendations 2001** — NMR Nomenclature: spin properties & chemical-shift conventions. *Pure Appl. Chem.* 73(11):1795–1818. [10.1351/pac200173111795](https://doi.org/10.1351/pac200173111795) | ✅ **DSS as pH-independent aqueous standard** at 0.00 ppm — `REFERENCE_STANDARD` (identification_quality.py). |
| **Emwas et al. 2018** — Recommended strategies for spectral processing of 1D ¹H-NMR biofluids. *Metabolomics* 14:31. [10.1007/s11306-018-1321-4](https://doi.org/10.1007/s11306-018-1321-4) | ✅ **Excluded-region defaults** — water 4.70–4.90, urine urea, serum lipid bands in the conditions form. |

---

## B · Metabolite identification & quantification (Track 1)

| Ref | What it grounds |
|---|---|
| **Lawson & Hanson 1974/1995** — Solving Least Squares Problems (NNLS, Ch. 23). SIAM. [10.1137/1.9781611971217](https://doi.org/10.1137/1.9781611971217) | ✅ **NNLS deconvolution** — `deconvolve()` un-mixes overlapping peaks as a non-negative linear combination of unit-area Gaussian references (`scipy.optimize.nnls`). |
| **Elias & Gygi 2007** — Target-decoy search strategy… *Nat. Methods* 4(3):207–214. [10.1038/nmeth1019](https://doi.org/10.1038/nmeth1019) | ✅ **Target–decoy FDR** — decoys built by shifting each reference +0.37 ppm into empty space; FDR(t)=#decoys≥t/#targets≥t, accept ≤0.05. Adapted from proteomics. |
| **Tardivel et al. 2017** — ASICS: automatic identification & quantification in ¹H-NMR. *Metabolomics* 13:109. [10.1007/s11306-017-1244-5](https://doi.org/10.1007/s11306-017-1244-5) | 📐 **NNLS linear-combination profiling precedent** — the "ASICS-style" label on Step 3. |
| **Lefort et al. 2019** — ASICS R package: full ¹H-NMR workflow. *Bioinformatics* 35(21):4356–4363. [10.1093/bioinformatics/btz248](https://doi.org/10.1093/bioinformatics/btz248) | 📋 ASICS workflow precedent (planning). |
| **Zheng et al. 2011** — BQuant: Bayesian identification+quantification in ¹H-NMR. *Bioinformatics* 27(12):1637–1644. [10.1093/bioinformatics/btr118](https://doi.org/10.1093/bioinformatics/btr118) | 📋 Library-based ID+quant precedent for the NNLS strategy (planning). |
| **Sumner et al. 2007** — Minimum reporting standards (CAWG/MSI). *Metabolomics* 3(3):211–221. [10.1007/s11306-007-0082-2](https://doi.org/10.1007/s11306-007-0082-2) | ✅ **MSI identification levels** — `msi_level()` caps every call at Level 2 (library match, no authentic standard), never Level 1. |
| **MSI members 2007** — The Metabolomics Standards Initiative. *Nat. Biotechnol.* 25:846–848. [10.1038/nbt0807-846b](https://doi.org/10.1038/nbt0807-846b) | ✅ Companion standard for the four-level scheme. |
| **Schymanski et al. 2014** — Identifying small molecules via HRMS: communicating confidence. *Environ. Sci. Technol.* 48(4):2097–2098. [10.1021/es5002105](https://doi.org/10.1021/es5002105) | 📋 ID-confidence sub-levels (planning refinement; not implemented). |
| **Dona et al. 2016** — A guide to the identification of metabolites in NMR-based metabolomics. *Comput. Struct. Biotechnol. J.* 14:135–153. [10.1016/j.csbj.2016.02.005](https://doi.org/10.1016/j.csbj.2016.02.005) | ✅ **Landmark ¹H assignments** (lactate doublet ~1.34, glucose α-anomer ~5.23) behind the diagnostic-ppm handles. *(Previously mis-labeled "Everett 2016 / IUPAC" — see corrections.)* |
| **Kaluarachchi et al. 2018** — Serum vs plasma metabolites by ¹H-NMR & UPLC-MS. *Metabolomics* 14:32. [10.1007/s11306-018-1332-1](https://doi.org/10.1007/s11306-018-1332-1) | ✅ Serum/plasma ppm anchors reported by `select_diagnostic_ppm()`. |
| **Bhinderwala et al. 2022** — Chemical shift variations in common metabolites. *J. Magn. Reson.* 345:107335. [10.1016/j.jmr.2022.107335](https://doi.org/10.1016/j.jmr.2022.107335) | ✅ Physics justification for the ppm-drift augmentation in training-mixture simulation. |
| **Tredwell et al. 2016** — Acid/base ¹H shift limits in human urine. *Metabolomics* 12:152. [10.1007/s11306-016-1101-y](https://doi.org/10.1007/s11306-016-1101-y) | ✅ Bounds the ppm-drift augmentation magnitude. |
| **Haslauer et al. 2019** — Guidelines for D₂O use in ¹H-NMR metabolomics. *Anal. Chem.* 91(17):11063–11069. [10.1021/acs.analchem.9b01580](https://doi.org/10.1021/acs.analchem.9b01580) | ✅ **D₂O exchangeable-proton guard** — OH/NH/SH protons vanish by H/D exchange, so only non-exchangeable C–H count toward coverage. |
| **Chenomx (Weljie et al. 2006)** — Targeted profiling: quantitative ¹H-NMR. *Anal. Chem.* 78(13):4430–4442. [10.1021/ac060209g](https://doi.org/10.1021/ac060209g) | 📐 Design analogue for the profiler UX (Reference Card + compound table, identified-first). |
| **rDolphin (Cañueto et al. 2018)** — automatic profiling of 1D ¹H-NMR. *Metabolomics* 14:24. [10.1007/s11306-018-1319-y](https://doi.org/10.1007/s11306-018-1319-y) | 📐 Open-source targeted-profiling precedent. |

### Learned identification / quantification channels

| Ref | What it grounds |
|---|---|
| **Wei et al. 2022** — Deep-learning compound ID in NMR mixtures (pSCNN). *Molecules* 27(12):3653. [10.3390/molecules27123653](https://doi.org/10.3390/molecules27123653) | ✅ **pSCNN hybrid channel** — the entire `pscnn.py`: two weight-independent conv towers over (reference, sample) pairs; union with the FDR set at present-prob ≥ 0.6. |
| **Bromley et al. 1993** — Signature Verification using a "Siamese" Time-Delay Neural Network. *NIPS 6.* | 📐 Architectural lineage of the pseudo-Siamese design. |
| **Koch, Zemel & Salakhutdinov 2015** — Siamese Networks for One-shot Image Recognition. *ICML DL Workshop.* | 📐 One-shot Siamese precedent for the pSCNN twin towers. |
| **NMRformer (2025)** — Transformer peak assignment in 1D ¹H-NMR. *Anal. Chem.* 97(1):904–911. [10.1021/acs.analchem.4c05632](https://doi.org/10.1021/acs.analchem.4c05632) | ✅ Optional neural assignment backend blended in `analyze_spectrum()`. |
| **Dosovitskiy et al. 2021** — An Image is Worth 16×16 Words (ViT). *ICLR.* [arXiv:2010.11929](https://arxiv.org/abs/2010.11929) | 📐 ViT-hybrid design of the GISSMO transformer `Quantifier` (Conv1d patch-embed + TransformerEncoder + softplus). |
| **Johnson & Tipirneni-Sajja 2025** — Optimizing NN-based NMR quantification. *Metabolites* 15(4):249. [10.3390/metabo15040249](https://doi.org/10.3390/metabo15040249) | 📋 Transformer-quantifier precedent (quantifier ships as a documented honest-negative). |
| **Wang et al. 2023** — NMRQNet: DL identification/quantification in plasma. *bioRxiv.* [10.1101/2023.03.01.530642](https://doi.org/10.1101/2023.03.01.530642) | 📋 CNN+GRU quantifier precedent (planning). |

---

## C · Reference chemical-shift libraries, databases & data sources

| Ref | What it grounds |
|---|---|
| **Wishart et al. 2022** — HMDB 5.0. *Nucleic Acids Res.* 50(D1):D622–D631. [10.1093/nar/gkab1062](https://doi.org/10.1093/nar/gkab1062) | ✅ **Reference ¹H shift library** (`REFERENCE_SHIFTS`) + curated per-metabolite biology cards. |
| **Hoch et al. 2023** — Biological Magnetic Resonance Data Bank (BMRB). *Nucleic Acids Res.* 51(D1):D368–D376. [10.1093/nar/gkac1050](https://doi.org/10.1093/nar/gkac1050) | ✅ **Real assigned shifts** merged into the library **and the independent real held-out test set** for identification benchmarking. |
| **Dashti et al. 2017** — GISSMO: spin-system modeling of NMR spectra. *Anal. Chem.* 89(22):12201–12208. [10.1021/acs.analchem.7b02884](https://doi.org/10.1021/acs.analchem.7b02884) | ✅ **Physically-exact ¹H shifts** — GISSMO panel cached (`external_reference.py`) + the simulated-mixture training corpus (`gissmo_corpus.py`). |
| **Dashti et al. 2018** — Applications of Parametrized NMR Spin Systems. *Anal. Chem.* 90(18):10646–10649. [10.1021/acs.analchem.8b02660](https://doi.org/10.1021/acs.analchem.8b02660) | 📋 GISSMO scale (>1,100 compounds) supporting evidence. |
| **Haug et al. 2013** — MetaboLights repository. *Nucleic Acids Res.* 41(D1):D781–D786. [10.1093/nar/gks1004](https://doi.org/10.1093/nar/gks1004) | ✅ **External validation cohorts** (MTBLS161, MTBLS242…) + MAF reference format. |
| **Sud et al. 2016** — Metabolomics Workbench. *Nucleic Acids Res.* 44(D1):D463–D470. [10.1093/nar/gkv1042](https://doi.org/10.1093/nar/gkv1042) | ✅ External validation data source (REST API; e.g. study ST004325). |
| **Kim et al. 2018** — PUG-REST update (PubChem). *Nucleic Acids Res.* 46(W1):W563–W570. [10.1093/nar/gky294](https://doi.org/10.1093/nar/gky294) | ✅ Reference-card enrichment (IUPAC name, CAS, CID, InChIKey, SMILES) + cross-DB links. |
| **Degtyarenko et al. 2008** — ChEBI: database & ontology. *Nucleic Acids Res.* 36(D1):D344–D350. [10.1093/nar/gkm791](https://doi.org/10.1093/nar/gkm791) | ✅ ChEBI cross-reference links. |
| **Kanehisa & Goto 2000** — KEGG: Kyoto Encyclopedia of Genes and Genomes. *Nucleic Acids Res.* 28(1):27–30. [10.1093/nar/28.1.27](https://doi.org/10.1093/nar/28.1.27) | ✅ KEGG links + pathway names underlying enrichment. |
| **CASMDB (2024/2026)** — Open 1D ¹H-NMR metabolite annotation DB. *Anal. Chem.* [10.1021/acs.analchem.5c04525](https://doi.org/10.1021/acs.analchem.5c04525) | 📋 Candidate answer-key library for held-out validation (planning). |
| **MTBLS242 (Palau-Rodriguez et al. 2015)** — Serum fingerprint of severe obesity & bariatric surgery. *Am. J. Clin. Nutr.* 102(6):1313–1322. [10.3945/ajcn.115.110536](https://doi.org/10.3945/ajcn.115.110536) | ✅ Serum cohort whose MAF format `library.py` parses; 21-metabolite serum fallback panel. |
| **MTBLS1 (Salek et al. 2007)** — Urinary metabolomics of T2D across species. *Physiol. Genomics* 29(2):99–108. [10.1152/physiolgenomics.00194.2006](https://doi.org/10.1152/physiolgenomics.00194.2006) | ✅ Worked-example urine cohort (Track-2 + GGM demo). |

---

## D · Biomarker discovery — leakage-safe validation methodology (Track 2)

**Study design & anti-leakage**

| Ref | What it grounds |
|---|---|
| **Ambroise & McLachlan 2002** — Selection bias in gene extraction from microarrays. *PNAS* 99(10):6562–6566. [10.1073/pnas.102102699](https://doi.org/10.1073/pnas.102102699) | ✅ **Feature-selection-inside-each-fold** design of `discover()`; the "leaky AUC" exhibit quantifies the bias this paper describes. |
| **Varma & Simon 2006** — Bias in error estimation with CV for model selection. *BMC Bioinformatics* 7:91. [10.1186/1471-2105-7-91](https://doi.org/10.1186/1471-2105-7-91) | ✅ Nested-CV rationale. |
| **Vabalas et al. 2019** — ML validation with a limited sample size. *PLoS ONE* 14(11):e0224365. [10.1371/journal.pone.0224365](https://doi.org/10.1371/journal.pone.0224365) | ✅ Justifies nested CV: flat k-fold is optimistic at small n; pooled-data selection is the largest bias source. |
| **Diaz-Uriarte et al. 2022** — Ten quick tips for biomarker discovery with ML. *PLoS Comput. Biol.* 18(8):e1010357. [10.1371/journal.pcbi.1010357](https://doi.org/10.1371/journal.pcbi.1010357) | ✅ All label-using steps (screen, impute, scale, PCA) fit inside training folds only. |
| **Saeb et al. 2017** — Cross-validation strategies (subject-wise splitting). *GigaScience* 6(5):1–6. [10.1093/gigascience/gix020](https://doi.org/10.1093/gigascience/gix020) | ✅ **Patient-grouped `StratifiedGroupKFold`** — whole subjects held out together, no leakage across folds. |

**Significance, confidence & stability**

| Ref | What it grounds |
|---|---|
| **Benjamini & Hochberg 1995** — Controlling the False Discovery Rate. *JRSS-B* 57(1):289–300. [10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) | ✅ **BH-FDR** used throughout: in-fold feature selection, differential analysis, pathway enrichment, GGM edges. |
| **Westerhuis et al. 2008** — Assessment of PLS-DA cross validation. *Metabolomics* 4(1):81–89. [10.1007/s11306-007-0099-6](https://doi.org/10.1007/s11306-007-0099-6) | ✅ **Permutation testing** + Q² (PRESS-based) robustness metric. |
| **Ojala & Garriga 2010** — Permutation Tests for Studying Classifier Performance. *JMLR* 11:1833–1863. | ✅ **Label-permutation null** p = (1+#{permAUC≥real})/(n_perm+1). |
| **Efron 1979** — Bootstrap Methods: Another Look at the Jackknife. *Ann. Stat.* 7(1):1–26. [10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552) | ✅ **Patient-clustered percentile bootstrap** 95% CI on the CV-AUC. |
| **Tsamardinos et al. 2018** — Bootstrapping out-of-sample predictions for CV. *Mach. Learn.* 107(12):1895–1922. [10.1007/s10994-018-5714-4](https://doi.org/10.1007/s10994-018-5714-4) | ✅ Pooled-OOF bootstrap CI method. |
| **LeDell et al. 2015** — Efficient CIs for cross-validated AUC (cvAUC). *Electron. J. Stat.* 9(1):1583–1607. [10.1214/15-EJS1035](https://doi.org/10.1214/15-EJS1035) | ✅ Co-grounds the CV-AUC confidence interval. |
| **Bengio & Grandvalet 2004** — No Unbiased Estimator of the Variance of K-Fold CV. *JMLR* 5:1089–1105. | ✅ Honesty caveat: the CV-AUC CI is **approximate** (no unbiased K-fold variance estimator exists). |
| **Kalousis et al. 2007** — Stability of feature selection algorithms. *Knowl. Inf. Syst.* 12(1):95–116. [10.1007/s10115-006-0040-8](https://doi.org/10.1007/s10115-006-0040-8) | ✅ **Jaccard stability** of the per-fold panel (`_stable_from_folds()`). |

**Statistical tests & effect sizes (differential analysis)**

| Ref | What it grounds |
|---|---|
| **Mann & Whitney 1947** — rank-sum test. *Ann. Math. Stat.* 18(1):50–60. [10.1214/aoms/1177730491](https://doi.org/10.1214/aoms/1177730491) | ✅ Default two-group non-parametric headline p. |
| **Welch 1947** — t-test with unequal variances. *Biometrika* 34:28–35. [10.1093/biomet/34.1-2.28](https://doi.org/10.1093/biomet/34.1-2.28) | ✅ Two-group parametric fallback. |
| **Kruskal & Wallis 1952** — Ranks in one-criterion variance analysis. *JASA* 47(260):583–621. [10.1080/01621459.1952.10483441](https://doi.org/10.1080/01621459.1952.10483441) | ✅ Default >2-group non-parametric test. |
| **Fisher 1925 / Snedecor 1934** — one-way ANOVA (F-test). | ✅ >2-group parametric fallback + ANOVA-F multi-class feature screen. |
| **Tate 1954** — Point-Biserial Correlation. *Ann. Math. Stat.* 25(3):603–607. [10.1214/aoms/1177728730](https://doi.org/10.1214/aoms/1177728730) | ✅ Binary univariate association screen (`screen_features()`). |
| **Cohen 1988** — Statistical Power Analysis (Cohen's d). Routledge. | ✅ Two-group standardized effect size. |
| **Pearson 1905** — correlation ratio / η² (eta-squared). | ✅ Multi-group effect size. |

**Classifier suite & metrics (honest model comparison)**

| Ref | What it grounds |
|---|---|
| **Cortes & Vapnik 1995** — Support-Vector Networks. *Mach. Learn.* 20:273–297. [10.1007/BF00994018](https://doi.org/10.1007/BF00994018) | ✅ Linear-SVM challenger. |
| **Breiman 2001** — Random Forests. *Mach. Learn.* 45:5–32. [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324) | ✅ Random-forest challenger + permutation feature importance. |
| **Friedman 2001** — Greedy Function Approximation (gradient boosting). *Ann. Stat.* 29(5):1189–1232. [10.1214/aos/1013203451](https://doi.org/10.1214/aos/1013203451) | ✅ Histogram gradient-boosting challenger. |
| **Chen & Guestrin 2016** — XGBoost. *KDD '16* 785–794. [10.1145/2939672.2939785](https://doi.org/10.1145/2939672.2939785) | ✅ Optional XGBoost challenger. |
| **Zou & Hastie 2005** — Regularization & Variable Selection via the Elastic Net. *JRSS-B* 67(2):301–320. [10.1111/j.1467-9868.2005.00503.x](https://doi.org/10.1111/j.1467-9868.2005.00503.x) | ✅ Elastic-net logistic baseline. |
| **Tibshirani 1996** — Regression Shrinkage and Selection via the Lasso. *JRSS-B* 58(1):267–288. | ✅ L1-logistic sparse panel selection. |
| **Wold et al. 2001 / Chong & Jun 2005** — PLS-regression & VIP. *Chemom. Intell. Lab. Syst.* [10.1016/S0169-7439(01)00155-1](https://doi.org/10.1016/S0169-7439(01)00155-1) | ✅ PLS-DA **VIP** biomarker-ranking statistic. |
| **Hand & Till 2001** — Generalisation of AUC for multi-class. *Mach. Learn.* 45(2):171–186. [10.1023/A:1010920819831](https://doi.org/10.1023/A:1010920819831) | ✅ Multi-class ROC-AUC. *(We use one-vs-rest; Hand-Till defines one-vs-one — see corrections.)* |
| **Brier 1950** — Verification of Forecasts. *Mon. Weather Rev.* 78(1):1–3. | ✅ Probabilistic-accuracy/calibration metric + tiebreaker. |
| **Naeini et al. 2015 / Guo et al. 2017** — Expected Calibration Error. *AAAI / ICML.* | ✅ 8-bin ECE calibration metric. |

---

## E · Metabolite correlation network (GGM)

| Ref | What it grounds |
|---|---|
| **Krumsiek et al. 2011** — Gaussian graphical modeling reconstructs pathway reactions. *BMC Syst. Biol.* 5:21. [10.1186/1752-0509-5-21](https://doi.org/10.1186/1752-0509-5-21) | ✅ **GGM partial-correlation network** — each edge conditioned on all other metabolites, so indirect (shared-driver) links vanish; the interactive Discover network. |
| **Ledoit & Wolf 2004** — Well-conditioned estimator for large covariance matrices. *J. Multivar. Anal.* 88(2):365–411. [10.1016/S0047-259X(03)00096-4](https://doi.org/10.1016/S0047-259X(03)00096-4) | ✅ **Shrinkage covariance** so the precision matrix inverts in the p≥n regime. |
| **Fisher 1921** — "Probable error" of a correlation coefficient. *Metron* 1:3–32. | ✅ **Fisher z-transform** → per-edge p-values (dof = n−p) before BH-FDR edge selection. |

---

## F · Dimensionality reduction (exploratory — hypotheses, not proof)

| Ref | What it grounds |
|---|---|
| **Pearson 1901** — On lines and planes of closest fit. *Phil. Mag.* 2(11):559–572. [10.1080/14786440109462720](https://doi.org/10.1080/14786440109462720) · **Hotelling 1933** — *J. Educ. Psychol.* 24:417–441. [10.1037/h0071325](https://doi.org/10.1037/h0071325) | ✅ **PCA** — exploratory variance/loadings, UMAP pre-reduction. |
| **McInnes, Healy & Melville 2018** — UMAP. [arXiv:1802.03426](https://arxiv.org/abs/1802.03426) | ✅ **UMAP** nonlinear embedding for cohort visualization (with `interpretation_warning`). |

---

## G · Biological interpretation (NMR → biology: metabolite → pathways & meaning)

*This is the layer that turns an identified metabolite list into biological meaning —
`biology.py` / `biology.interpret_panel()`, wired into every discovery endpoint in `main.py`.*

| Ref | What it grounds |
|---|---|
| **Wishart et al. 2022** — HMDB 5.0. *Nucleic Acids Res.* 50(D1):D622–D631. [10.1093/nar/gkab1062](https://doi.org/10.1093/nar/gkab1062) | ✅ **Per-metabolite biology cards** (`METABOLITE_BIOLOGY`) — role, disease associations, pathways, direction; each tagged source "HMDB 5.0 (curated)". |
| **Xia & Wishart (MetaboAnalyst 4.0) 2018** — *Nucleic Acids Res.* 46(W1):W486–W494. [10.1093/nar/gky310](https://doi.org/10.1093/nar/gky310) · **MetPA (Xia & Wishart 2010)** — *Bioinformatics* 26(18):2342–2344. [10.1093/bioinformatics/btq418](https://doi.org/10.1093/bioinformatics/btq418) | ✅ **Pathway over-representation** — `pathway_enrichment()` uses the hypergeometric (Fisher) over-representation test, the same method MetaboAnalyst/MetPA applies. |
| **Kanehisa & Goto 2000** — KEGG. *Nucleic Acids Res.* 28(1):27–30. [10.1093/nar/28.1.27](https://doi.org/10.1093/nar/28.1.27) | ✅ **Pathway definitions** — the KEGG pathway names (Glycolysis, TCA cycle, BCAA degradation…) the enrichment counts against. |
| **Benjamini & Hochberg 1995** — FDR. *JRSS-B* 57(1):289–300. [10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x) | ✅ **BH-FDR** correction over the per-pathway p-values (multiplicity control). |

*The specific metabolite→disease links these cards assert (BCAA→T2D, TMAO→CVD, etc.) are
evidenced by the clinical literature in the next section.*

---

## H · Clinical & biological grounding (what the biomarkers mean)

| Ref | What it grounds |
|---|---|
| **Newgard et al. 2009** — BCAA metabolic signature & insulin resistance. *Cell Metab.* 9(4):311–326. [10.1016/j.cmet.2009.02.002](https://doi.org/10.1016/j.cmet.2009.02.002) | ✅ **BCAA ↔ insulin resistance / T2D** — the `NCD_RELEVANCE` map behind diagnostic-ppm annotation + convergent-validity story. |
| **Wang et al. 2011** — Metabolite profiles & risk of developing diabetes. *Nat. Med.* 17(4):448–453. [10.1038/nm.2307](https://doi.org/10.1038/nm.2307) | ✅ BCAA/aromatic-AA predict incident T2D (>5× top-quartile risk). |
| **Cheng et al. 2012** — Metabolite profiling & metabolic risk. *Circulation* 125(18):2222–2231. [10.1161/CIRCULATIONAHA.111.067827](https://doi.org/10.1161/CIRCULATIONAHA.111.067827) | ✅ Gln/Glu ratio ↔ cardiometabolic risk. |
| **Würtz et al. 2015** — Metabolite profiling & CV event risk (3 cohorts). *Circulation* 131(9):774–785. [10.1161/CIRCULATIONAHA.114.013116](https://doi.org/10.1161/CIRCULATIONAHA.114.013116) | ✅ Phenylalanine ↔ incident CV events (convergent validity). |
| **Guasch-Ferré et al. 2016** — Metabolomics in (pre)diabetes: meta-analysis. *Diabetes Care* 39(5):833–846. [10.2337/dc15-2251](https://doi.org/10.2337/dc15-2251) | ✅ BCAA/T2D-risk corroboration. |
| **Ahola-Olli et al. 2019** — Circulating metabolites & T2D risk (Finnish cohorts). *Diabetologia* 62(12):2298–2309. [10.1007/s00125-019-05001-w](https://doi.org/10.1007/s00125-019-05001-w) | ✅ Glucose/lactate/alanine in incident-T2D sets. *(Correct source is four Finnish cohorts, not UK Biobank — see corrections.)* |
| **Louca et al. 2022** — Blood metabolite markers of hypertension (CoMETS, n=44,306). *Metabolites* 12(7):601. [10.3390/metabo12070601](https://doi.org/10.3390/metabo12070601) | ✅ BCAA ↔ hypertension (extends signature to 2nd NCD). |
| **Wang et al. 2011 (Nature) / Tang et al. 2013 (NEJM)** — TMAO & cardiovascular disease. [10.1038/nature09922](https://doi.org/10.1038/nature09922) · [10.1056/NEJMoa1109400](https://doi.org/10.1056/NEJMoa1109400) | ✅ TMAO biology card. |
| **Cori & Cori 1929** — Glycogen formation from lactic acid (Cori cycle). *J. Biol. Chem.* 81(2):389–403. | ✅ Lactate energy-shuttle biology card. |
| **Anthony et al. 2000** — Leucine & mTOR-dependent translation. *J. Nutr.* 130(10):2413–2419. | ✅ Leucine/mTOR biology card. |
| **Beckonert et al. 2007** — NMR metabolic profiling protocols. *Nat. Protoc.* 2(11):2692–2703. [10.1038/nprot.2007.376](https://doi.org/10.1038/nprot.2007.376) | 📋 Sample→spectrum protocol framework (doc reference). |

**Impact / epidemiology framing**

| Ref | What it grounds |
|---|---|
| **Julkunen et al. 2023** — Atlas of plasma NMR biomarkers (UK Biobank, n=118,461). *Nat. Commun.* 14:604. [10.1038/s41467-023-36231-7](https://doi.org/10.1038/s41467-023-36231-7) | 📐 Impact anchor: one NMR run → 249 measures mapped to >700 diseases. |
| **Buergel et al. 2024** — Metabolomic vs genomic prediction (n=700,217). *Nat. Commun.* 15:10092. [10.1038/s41467-024-54357-0](https://doi.org/10.1038/s41467-024-54357-0) | 📐 Impact anchor: NMR scores beat polygenic scores for most diseases. |
| **Rattanavipapong et al. 2022** — Economic burden of T2D in North Thailand. *Front. Endocrinol.* 13:824545. [10.3389/fendo.2022.824545](https://doi.org/10.3389/fendo.2022.824545) | 📐 Thailand cost-of-complications figures. |
| **Aekplakorn et al. 2024** — Hypertension trends in Thailand (NHES, 2004–2020). *BMC Public Health* 24:3149. [10.1186/s12889-024-20643-1](https://doi.org/10.1186/s12889-024-20643-1) | 📐 Thailand hypertension epidemiology (~48% undiagnosed). |

---

## I · Reproducibility, data & reporting standards

| Ref | What it grounds |
|---|---|
| **Sandve et al. 2013** — Ten Simple Rules for Reproducible Computational Research. *PLoS Comput. Biol.* 9(10):e1003285. [10.1371/journal.pcbi.1003285](https://doi.org/10.1371/journal.pcbi.1003285) | ✅ Fixed seeds, pinned lockfile, one-command scripts, VC'd code. |
| **Wilkinson et al. 2016** — FAIR Guiding Principles. *Sci. Data* 3:160018. [10.1038/sdata.2016.18](https://doi.org/10.1038/sdata.2016.18) | ✅ Open-licensed corpora, every study referenced by accession. |
| **Mitchell et al. 2019** — Model Cards for Model Reporting. *FAT\* '19* 220–229. [10.1145/3287560.3287596](https://doi.org/10.1145/3287560.3287596) | ✅ Model-documentation standard (intended use, provenance, metrics-with-CIs, limitations). |

---

## Citation corrections to make before judging

The web verification flagged three attribution slips (all real papers — the *labels* are off):

1. **"Everett 2016 (IUPAC)" → Dona et al. 2016.** The paper cited (Comput. Struct. Biotechnol. J. 14:135–153) is *A guide to the identification of metabolites…* by **Dona et al.** (Jeremy Everett is a co-author, not first). It's a metabolite-ID guide, **not** the IUPAC shift-convention paper — IUPAC conventions are **Harris et al. 2008**. Fix the label in the conditions-form help text (profiler.html) and TRACK1 plan.
2. **Ahola-Olli et al. 2019 — source cohorts.** The study is **four Finnish cohorts (n≈11,896)**, not UK Biobank. Correct any "UK Biobank" phrasing tied to this citation.
3. **Hand & Till 2001 vs one-vs-rest AUC.** Hand-Till defines a **one-vs-one** multi-class AUC; the code uses `roc_auc_score(multi_class='ovr')` (one-vs-rest). Either cite it as "in the spirit of Hand-Till" or switch to `multi_class='ovo'` to match the paper exactly.

Everything else (≈57 papers) verified as correctly attributed and correctly used.

---

*Compiled from a full citation harvest across `backend/nmr_api/*.py`, `docs/*.md`, and
`static/profiler.html`, with each reference web-verified against its DOI/PMID. Cross-check
numeric claims in [IMPACT_AND_VALIDATION.md](IMPACT_AND_VALIDATION.md); method derivations in
[WORKFLOWS_DATA_ALGORITHM.md](WORKFLOWS_DATA_ALGORITHM.md).*
