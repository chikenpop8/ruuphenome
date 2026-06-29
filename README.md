# RuuPhenome

**An open, auditable alternative to Chenomx for ¹H-NMR metabolomics.**
Built for the BDI Young Innovator Hackathon 2026 — Track: Phenome.

RuuPhenome takes NMR evidence from a cohort, annotates and quantifies
metabolites, then connects that metabolite table to leakage-safe biomarker
discovery and biological interpretation — end to end, in the browser, with every
step explainable.

---

## Why it matters (the problem)

Closed NMR profiling tools like Chenomx are expensive, proprietary, and run only
on the analyst's desktop — a barrier for Thai research labs and a black box for
reviewers. RuuPhenome is:

- **Free and open** — no per-seat licence.
- **On-premise** — runs entirely inside your own VM; no data leaves the host.
- **Auditable** — every annotation, fit, and biomarker score is inspectable,
  not hidden behind a closed model.
- **End-to-end** — it doesn't stop at a metabolite list (Track 1); it carries
  through to biomarkers and pathway biology (Track 2), which Chenomx does not do.

---

## Quick start

```bash
bash backend/nmr_api/run.sh
```

Opens on **http://127.0.0.1:8100**. Set `NMR_HOST=0.0.0.0` to expose it on the competition VM.

### Key URLs

| URL | What |
|---|---|
| `/` | The profiler UI |
| `/docs` | Interactive API (Swagger) |
| `/healthz` | Health check |

---

## What it does

**Track 1 — cohort profiling**
binned NMR matrix → orientation detection → PQN normalization → reference-shift
annotation → NNLS deconvolution with target-decoy / FDR filtering → per-sample
metabolite concentration table → overlay visualization → CSV export.

**Track 2 — biomarkers & biology**
metabolite/feature matrix + metadata → patient-grouped, leakage-safe nested CV →
model comparison (elastic-net, SVM, gradient boosting, PCA variants) with Q²,
VIP, and permutation tests → stable biomarker panel → pathway enrichment and
curated metabolite biology cards.

---

## Reproducibility

- **Deterministic by construction:** single-threaded native math + fixed random
  seeds in all modelling code.
- **38 automated tests:**
  ```bash
  backend/nmr_api/.venv/bin/python -m unittest discover -s backend/nmr_api/tests
  ```

---

## Data governance

RuuPhenome is designed for closed-dataset environments. Set `NMR_OFFLINE=1` to
block every outbound network call so no dataset row can leave the host. Reference
enrichment (PubChem metadata) is served from a **bundled offline cache**; it
never fetches data at demo time.

---

## Honest limitations

We deliberately do **not** overclaim:

- Annotation coverage is generous for demonstration; production claims need
  stricter scoring and comparison against manual / Chenomx-reviewed ground truth.
- Quantification is Chenomx-style estimation, not yet wet-lab validated against
  internal standards.
- The self-supervised encoder / NMRformer are **optional** and off by default;
  the explainable pattern-matching path is the primary method.
- PCA/UMAP plots are exploratory structure, not classifier performance.

See `HANDOFF.md` for the full engineering status and known gaps.
