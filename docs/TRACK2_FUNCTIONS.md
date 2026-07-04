# Track 2 — How Each Function Works

A technical reference for the five Track-2 analytics functions: what each one
does, the algorithm behind it, the code path, and the inputs/outputs. For the
*why-it-matters* (impact, sourced biology, validation, limitations) see
[`IMPACT_AND_VALIDATION.md`](IMPACT_AND_VALIDATION.md).

**Golden rule:** everything here runs **fresh on each upload** (classical
statistics / classical ML — no training, no GPU), and every label-using step is
fit **inside cross-validation folds** so reported performance is leakage-free.

---

## 0. The data model — two tables in

Track 2 takes the organizer's two tables:

| Input | Shape | Example columns |
|---|---|---|
| **Table 1 — concentrations** | metabolites × samples (a MetaboLights MAF) | `metabolite_identification`, `S1`, `S2`, … |
| **Table 2 — metadata** | one row per sample | `Sample Name`, `Factor Value[Disease]`, covariates |

**Shared assembly (every function starts here):**
1. `biomarkers.build_matrix(table1)` → a **samples × metabolites** matrix `X`
   (transposes the MAF, tolerant of delimiter, EU decimals `12,34`, and missing
   tokens `n.d.`/`<LOD`). Sample columns are everything not in the MAF annotation
   block (`smiles`, `inchi`, `chemical_shift`, …).
2. `spectral_cohort.parse_metadata(table2)` → the metadata frame.
3. `spectral_cohort.derive_labels(meta, sample_ids, …)` → `{sample: class}` from the
   authoritative ISA-Tab `Factor Value[…]` column (never guessed from sample names).
   With `multiclass=True` it keeps **all** classes (0..K-1); binary is the default.
4. Rows are restricted to labeled samples; `groups = sample IDs` (one sample = one
   patient) so cross-validation holds whole patients out.

The **pre-flight preview** (`POST /track2/preview`) runs steps 1–3 only and reports
detected shape, the chosen condition column, class balance, sample-ID match rate,
and warnings — so a mis-read file is caught *before* any result.

---

## 1. The shared leakage-safe engine (read this once)

Functions 1, 2 and 5 sit on one cross-validation core. Inside **every training
fold**, in this order, using **only that fold's training rows**:

```
_impute_median   → median-fill missing cells (medians learned on train only)
variance_filter  → drop near-constant features
screen_features  → univariate score + p-value (binary: point-biserial t-test;
                   ≥3 groups: ANOVA-F via sklearn f_classif)
benjamini_hochberg → keep features passing BH-FDR (else fall back to top-k by score)
top-k            → the k highest-scoring survivors
StandardScaler   → fit on train, applied to the held-out fold
classifier       → LogisticRegression (multinomial for ≥3 classes)
```

Nothing that touches the labels is ever fit on the held-out data → **no selection
or scaling leakage**. Out-of-fold predictions are pooled to score the model. The
engine *also* computes a **leaky AUC** (features chosen on all data first) purely to
*show* the inflation gap — it is a diagnostic, never quoted as performance.

Code: `biomarker_engine._select_in_fold`, `_fit_eval` / `_fit_eval_multiclass`,
`_repeated_cv`. CV splitter: `StratifiedGroupKFold` (grouped) or `StratifiedKFold`.

---

## 2. Function 1 — Biomarker Discovery

**Goal:** find the *smallest stable panel* of metabolites that separates the groups,
with an honest performance estimate.

**Code:** `biomarker_engine.discover()`  ·  **Endpoints:** `POST /track2/biomarkers`,
`POST /track2/discover-with-metadata`, `GET /biomarkers-safe` (bundled cohorts),
`GET /ncd-screen`.

**How it works, step by step:**
1. **Signal guard** — fail fast with a clear message if every concentration cell is
   missing/constant.
2. **Repeated grouped CV** — for `repeats` rounds, run the §1 fold pipeline; collect
   out-of-fold predictions each round.
3. **Stable panel** — a feature is "stable" if it was selected in **≥ half of all
   folds**; `stable_panel_counts` gives its selection frequency, and
   `topk_stability_jaccard` is the mean pairwise Jaccard of the per-fold selections
   (how reproducible the choice is).
4. **VIP ranking** — PLS-DA Variable Importance in Projection for the panel
   (VIP > 1 = influential); binary tasks only.
5. **Permutation test** — shuffle the labels `n` times, re-run the whole leakage-safe
   CV each time, and report `p = (1 + #{permuted AUC ≥ observed}) / (n + 1)`. This is
   the real "is it better than chance?" test.
6. **Panel-size sweep** — re-run the CV at `k ∈ {1, 3, 5, 10}` to show *how few*
   metabolites still separate the groups (`panel_sweep`).
7. **Bootstrap 95% CI** — group-level percentile bootstrap of the pooled out-of-fold
   scores → `honest_roc_auc_ci95` (approximate; see IMPACT doc §3).
8. **Effect sizes** — per stable marker: direction, fold-change, single-marker AUC
   (`panel_stats`).

**Key outputs (JSON):**
```json
{
  "task_type": "binary",
  "honest_roc_auc": 0.742, "honest_roc_auc_ci95": [0.618, 0.873],
  "honest_f1": 0.60, "honest_q2": 0.12,
  "permutation_p_value": 0.0149, "n_permutations": 200,
  "leaky_roc_auc": 0.721, "leakage_inflation": -0.0211,
  "classification_metrics": { "accuracy":0.69, "sensitivity":0.54, "specificity":0.80,
                              "precision":0.67, "recall":0.54, "f1":0.60, "confusion_matrix":[[..]] },
  "stable_panel": ["L-Phenylalanine", "Hypoxanthine"],
  "panel_sweep": [ {"k":1,"honest_roc_auc":0.744}, {"k":3,"honest_roc_auc":0.756}, ... ],
  "topk_stability_jaccard": 0.7, "vip_scores": {...}, "panel_stats": [...]
}
```

---

## 3. Function 2 — Predictive Model

**Goal:** compare several classifiers *honestly* and recommend one for risk
stratification / triage.

**Code:** `model_suite.compare_models()`  ·  **Endpoints:** `GET /biomarkers-model-suite`,
`POST /biomarkers-model-suite-upload`.

**How it works:**
- **Nested, patient-grouped CV.** An **outer** `StratifiedGroupKFold` estimates
  generalization; an **inner** CV loop (`_inner_auc`) tunes each model's
  hyperparameters on the outer-training data only — the correct scheme when you both
  tune and evaluate.
- **Models compared:** `elastic_net_logistic`, `linear_svm`, `pca_logistic`,
  `pca_linear_svm`, `random_forest`, `hist_gradient_boosting`, `xgboost` (if
  installed). Each is a scikit-learn `Pipeline` with median imputation + scaling
  (+ PCA for the `pca_*` models) fit **inside** every fold.
- **Feature handling:** sparse/tree models use the §1 fold-internal selection;
  `pca_*` models use all features then reduce with fold-internal PCA (95% variance).
- **Metrics per model:** ROC-AUC (mean + std), F1, Brier score, calibration error
  (ECE, binary), and a **confusion matrix** + full metric set via
  `classification_metrics`.
- **Recommendation rule:** the **simplest** model within 0.01 ROC-AUC of the best
  (Occam — resists small-n overfitting), tie-broken by Brier.

**Multi-class:** fully supported — macro one-vs-rest AUC, macro-F1, per-class recall;
XGBoost switches to `multi:softprob`; the two-class guard is relaxed to `≥2`.

**Output:** a ranked `models` list (each with metrics + `stable_panel` +
`confusion_matrix`), plus `recommended_model` and the validation description.

---

## 4. Function 3 — Differential Analysis

**Goal:** for **every** metabolite, show which way it moves between groups and whether
that difference is significant after multiple-testing correction.

**Code:** `differential.differential_analysis()`  ·  **Endpoint:** `POST /track2/differential`.

**How it works** (this is *descriptive* full-cohort analysis — no train/test split
applies because no accuracy is claimed, so there is no leakage concern):
- **Per-metabolite** compute group means, then:
  - **Two groups:** log2 fold-change (positive class ÷ reference), **Welch t-test**
    (`scipy.stats.ttest_ind(equal_var=False)`) **and Mann-Whitney U**
    (`mannwhitneyu`, the rank-based, non-normal-robust headline for NMR data), plus
    **Cohen's d** effect size.
  - **>2 groups:** one-way **ANOVA** (`f_oneway`) **and Kruskal-Wallis** (`kruskal`,
    headline), plus **η²** effect size and the class in which the metabolite is
    highest.
- **FDR correction:** Benjamini-Hochberg **q-values across all metabolites**
  (`bh_qvalues`); significance is `q < 0.05`.
- **Volcano** (two-group): `{metabolite, log2_fold_change, neg_log10_q, significant}`
  for a volcano plot; the ranked table is sorted by q-value.

**Output:**
```json
{ "task_type":"binary", "test":"mann_whitney_u + welch_t", "n_significant":3,
  "table":[ {"metabolite":"Hypoxanthine","log2_fold_change":1.39,"p_value":..,
             "q_value":0.0023,"direction":"higher_in_control","effect_size":..} , ...],
  "volcano":[ {"metabolite":"Hypoxanthine","log2_fold_change":1.39,"neg_log10_q":2.6,
               "significant":true}, ...] }
```

---

## 5. Function 4 — Correlation Analysis

**Goal:** show how metabolites co-vary (network) and relate to clinical variables —
distinguishing **direct** links from indirect ones.

**Code:** `correlation.analyze()` + `correlation.covariate_correlation()`  ·
**Endpoint:** `POST /track2/correlation`.

**Two products on the metabolite matrix:**

**(a) Pairwise correlation heatmap.** Pearson or Spearman (Spearman = Pearson on
column ranks, robust to outliers/non-normality). Per-pair p-values via the Fisher
z-transform, then Benjamini-Hochberg FDR over the upper triangle. Output: a rounded
`correlation_matrix` + FDR-filtered `pairwise.edges`.

**(b) Partial-correlation Gaussian Graphical Model (GGM) network** — the innovation.
Raw correlation networks are dense and dominated by *indirect* links (if A→B→C, then
A and C correlate even without a direct relationship). A GGM removes that:
```
Σ̂  = Ledoit-Wolf shrunk covariance          # invertible even when features ≈/> samples (p≫n)
P   = Σ̂⁻¹  (precision / inverse covariance)
ζ_ij = −P_ij / √(P_ii · P_jj)                # partial correlation: A–B conditioned on all others
```
Each edge is a **direct** association (conditioned on every other metabolite),
following [Krumsiek et al. 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC3224437/).
p-values via Fisher-z (dof `n − p`), BH-FDR, and an `|r|` threshold → `network.edges`
(a node/edge list ready for graph drawing). *Verified:* on a chain X0→X1→X2 it keeps
X0–X1 and X1–X2 but drops the spurious X0–X2 edge.

**p≫n guard:** for raw ~20k-bin spectra the covariance is singular and a 20k×20k
matrix is unusable, so it caps to the 80 most-variable metabolites and warns (the
identified-metabolite table never hits this cap).

**(c) Covariate correlation.** `covariate_correlation()` correlates every metabolite
with one **numeric metadata column** (age/BMI/BP…) via Pearson/Spearman, with BH-FDR
— for metabolite ↔ clinical-variable links.

---

## 6. Function 5 — Feature Selection

**Goal:** keep the panel **small, stable, and leakage-safe** — the properties a
translatable test needs.

**Code:** `biomarker_engine._select_in_fold()` (selection) + `_stable_from_folds()`
(stability). It is the selection stage *inside* Functions 1 and 2, surfaced as its
own reportable output.

**How it works (per fold, train-only):**
1. **Filter — variance:** drop near-constant features (`variance_filter`).
2. **Filter — univariate:** score every feature vs the label (point-biserial t-test
   for binary, ANOVA-F for ≥3 groups) and keep those passing **BH-FDR**; if none pass,
   fall back to the top-k by score.
3. **Top-k:** the k strongest survivors.

**Stability reporting (across folds):**
- `stable_panel` — features chosen in ≥ half of all folds.
- `stable_panel_counts` — selection frequency per feature.
- `topk_stability_jaccard` — mean pairwise Jaccard of per-fold selections (0 = random,
  1 = identical every fold).

**Why it matters:** because selection is fit strictly inside folds, the reported AUC
isn't inflated by peeking; because stability is measured, you know whether the panel
is reproducible or a small-n artifact. (Embedded L1 / tree-importance / RFE selectors
are a documented future addition — currently selection is the leakage-safe *filter*
form.)

---

## 7. Endpoint reference

| Endpoint | Method | Function | Engine |
|---|---|---|---|
| `/track2/preview` | POST | pre-flight file check | parse + derive_labels |
| `/track2/biomarkers` | POST | Biomarker Discovery | `biomarker_engine.discover` (honest) |
| `/track2/discover-with-metadata` | POST | Biomarker Discovery | `biomarker_engine.discover` (honest) |
| `/track2/metadata-columns` | POST | list metadata columns | — |
| `/track2/differential` | POST | Differential Analysis | `differential.differential_analysis` |
| `/track2/correlation` | POST | Correlation / GGM | `correlation.analyze` |
| `/biomarkers-safe` | GET | Discovery (bundled cohorts) | `biomarker_engine.discover` (honest) |
| `/biomarkers-model-suite` | GET/POST | Predictive Model | `model_suite.compare_models` |
| `/biomarkers-projection` | GET/POST | PCA/UMAP (exploratory viz only) | `dimensionality` |
| `/biology` | GET | pathway enrichment on a panel | `biology` |
| `/ncd-screen` | GET | NCD panel (bundled) | `biomarker_engine.discover` |

> ⚠️ `GET /biomarkers` and `POST /biomarkers-upload` use an **older, partly-leaky**
> path (`biomarkers.discover`) that is **not** wired to the UI or any `/track2/*`
> route. Do not quote its bare `roc_auc` — use the honest engine above.

---

## 8. Metrics glossary — what to quote (and what never to)

**Quote these (honest):**
- `honest_roc_auc` (+ `honest_roc_auc_ci95`) — leakage-safe out-of-fold ROC-AUC and
  its group-level bootstrap 95% CI. Multi-class = macro one-vs-rest.
- `permutation_p_value` — significance vs a label-shuffle null.
- `honest_q2` — predictive R² from out-of-fold predictions (binary).
- `classification_metrics` — accuracy, sensitivity, specificity, precision, recall,
  F1 (+ macro forms and per-class recall for multi-class) and the confusion matrix.
- `q_value` — BH-FDR-adjusted per-metabolite significance (differential/correlation).
- `topk_stability_jaccard`, `stable_panel_counts` — panel reproducibility.

**Never quote as accuracy:**
- `leaky_roc_auc` / `leakage_inflation` — diagnostics that *show* the leakage gap.
- PCA/UMAP separation (`/biomarkers-projection`) — exploratory visualization only.
- The in-distribution SSL retrieval score (Track 1) — not external accuracy.

---

## 9. Binary vs multi-class

Every function auto-detects the number of classes from the labels. **Binary** is the
default and is byte-for-byte the original engine. **Multi-class (≥3 groups, e.g.
control / diabetes / hypertension)** switches to: ANOVA-F screening, multinomial
logistic regression, macro one-vs-rest ROC-AUC, macro-F1, per-class recall, and
Kruskal-Wallis/ANOVA in the differential. `derive_labels(multiclass=True)` preserves
all groups instead of collapsing to the two largest; a 2-class column degrades to the
identical binary result.
