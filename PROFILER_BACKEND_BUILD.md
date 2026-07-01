# Profiler Backend Build Spec — Confidence-Gated Workflow

For the next agent (codex). **Read `HANDOFF.md` first** — especially the sections
"Profiler workflow — train on H100, reuse, or neither?" and "How to add a new NCD
cohort". This file says *what backend to build, in what order, reusing what.*

The target UX flow (AI proposes, human verifies only the flagged cases):

```
1 QC gate → 2 Auto-profile → 3 Triage → 4 Quantify → 5 NCD panel → 6 Report+provenance
   (machine) (machine)        (HUMAN)   (machine)    (machine)     (HUMAN sign-off)
```

Human touchpoints: **Stage 3** (review only ⚠ metabolites) and **Stage 6** (sign
the report). Stage 1 only needs a human if QC fails. Everything else is automatic.

---

## START HERE (priority order — do not build out of order)

1. **FIRST — the result schema + `POST /profile/auto` (Stage 2).** This is the
   backbone. Stages 3, 4, and 6 are just *views over its output*. Get the
   `MetaboliteResult` object right once and the rest is thin. **Begin here.**
2. **Stage 1 QC gate** — the deterministic gate placed in front of Stage 2.
3. **Stage 3 Triage** — a sort of Stage-2 output into ✓/⚠/✗ (trivial once each
   result carries a confidence + status).
4. **Stage 4 Quantify** — concentrations + uncertainty (mostly already in the
   existing NNLS deconvolution; add bootstrap error bars).
5. **Stage 5 NCD panel** — **already done**: `GET /ncd-screen`. Leave it.
6. **Stage 6 Report + provenance** — assembles everything, last.

---

## The one data contract — build this object first

Every stage reads/writes this. It carries **confidence + provenance on every
metabolite** — that is the product differentiator and the data-sovereignty story,
so it is non-negotiable.

```jsonc
// MetaboliteResult
{
  "name": "Citrate",
  "assignment": { "source": "nmrformer", "prob": 0.97 },   // or "pattern-match"
  "peaks_used": [{ "ppm": 2.54, "amp": 1.2 }, { "ppm": 2.66, "amp": 1.1 }],
  "concentration": { "value": 184.3, "unit": "uM|a.u.", "ci_low": 171.0, "ci_high": 198.5 },
  "confidence": 0.91,                 // 0..1, combined score (see Stage 3)
  "fdr": 0.03,                        // target-decoy q-value
  "status": "accept | review | reject",
  "provenance": {                     // every number must be traceable
    "deconv": "nnls",
    "fit_residual": 0.04,
    "model_version": "onedTrans_0.9782",
    "flags": ["overlap@2.6ppm"]
  }
}
```

`POST /profile/auto` returns `{ "spectrum_meta": {...}, "metabolites": [MetaboliteResult, ...] }`.

---

## Reuse map — do NOT retrain or reinvent (grep before writing new)

| Need | Reuse this (already in `backend/nmr_api/`) | Train? |
|---|---|---|
| QC metrics, peak picking, baseline, phasing | `signal_processing.py` (has `NMRGLUE` guard) | none |
| Deconvolution (NNLS) + target-decoy FDR | the existing Domain-1 single-spectrum profiling code — grep `main.py` for the current profiler endpoints and follow into the module | none |
| Metabolite assignment + probability | `nmrformer_backend.py` (`.status()`, bundled `NMRformer/onedTrans_0.9782`) | reuse only |
| Spectrum fingerprint / confidence feature | `self_supervised.py` + `models/masked_nmr_encoder.pt` | reuse only |
| NCD biomarker panel | `biomarker_engine.discover(...)` via `GET /ncd-screen` | CPU only |

No new training is required to ship this workflow. H100 jobs (field-upconverter,
NN-quantifier) are optional upgrades — see HANDOFF section D.

---

## Per-stage backend spec

### Stage 1 — `POST /profile/qc`  (deterministic)
In: a spectrum (same input the current profiler accepts).
Out: `{ snr, baseline_score, water_residual, referenced: bool, verdict: "pass|warn|fail", reasons: [..] }`.
Build from `signal_processing` metrics + fixed thresholds. Fail fast: if `fail`,
Stage 2 should refuse to run unless `?override=true&reason=...` is passed.

### Stage 2 — `POST /profile/auto`  (the backbone)
Orchestrate for **all** metabolites at once:
deconvolve (NNLS) → assign (nmrformer) → compute `confidence` and `fdr` → set
`status` → fill `provenance`. Return the `MetaboliteResult[]` contract above.
`confidence` = simple, documented combination of assignment prob + (1−fit_residual)
+ FDR pass. Keep it a transparent formula now; a learned calibration is optional
later (CPU, not H100).

### Stage 3 — `GET /profile/triage?hi=0.85&lo=0.5`  (thin)
Bucket the Stage-2 results: `confidence ≥ hi` → accept, `< lo` → reject, else
review. Return `{ accepted: [...], review: [...], rejected: [...] }`. The human
edits only `review`. (Can also be precomputed in `status` during Stage 2.)

### Stage 4 — Quantify  (mostly exists)
Concentrations already come from NNLS. Add **bootstrap uncertainty** → fill
`concentration.ci_low/ci_high`. Surface internal-standard calibration status
(µM vs a.u.). Expose via the same `MetaboliteResult` or `POST /profile/quantify`.

### Stage 5 — NCD panel  → **already built**: `GET /ncd-screen`. No work.

### Stage 6 — `GET /profile/report`  (last)
Assemble QC verdict + accepted/edited metabolites + NCD panel + full provenance
into one JSON, plus a CSV export. Include a `signed_by` / `signed_at` field for the
human sign-off. Every concentration must trace back to its peaks + model + confidence.

---

## Acceptance smoke tests (add to `tests/`)

- `/profile/qc` on a known-good demo TSV → `verdict: "pass"`.
- `/profile/auto` on a demo spectrum → non-empty `metabolites`, each with
  `confidence`, `fdr`, `status`, `provenance`.
- `/profile/triage` → the three buckets partition the metabolite set (no overlap,
  no drops).
- `/profile/report` → contains QC + metabolites + panel + `signed_*` fields.
- Existing suite still green:
  `backend/nmr_api/.venv/bin/python -m unittest discover -s backend/nmr_api/tests -q`

## Rules

- Reuse existing modules; grep before writing anything new.
- Keep deterministic stages deterministic (QC, NNLS, FDR).
- Every `MetaboliteResult` MUST carry `confidence` + `provenance` — non-negotiable.
- Open-data only for any future training; never train on the closed competition set.
- After changes: run the test suite + `py_compile` the package.

---

## Build checklist — do in THIS exact order

Work top to bottom. Do not start a step until the one above passes its check.
Commit after each numbered step so progress is recoverable.

### Step 0 — Orient (no code yet)
- [ ] Read `HANDOFF.md` (sections D "train/reuse/none" and the gaps list) + this file.
- [ ] `git status --short` — preserve unrelated dirty files (CVD cohort, HANDOFF edits).
- [ ] **Locate the existing Domain-1 functions you will reuse** — grep and write
      down the real names/signatures before coding:
  - `grep -rn "nnls\|deconv\|decoy\|fdr\|peak_pick\|pick_peaks" backend/nmr_api/*.py`
  - open `nmrformer_backend.py` and note the assignment call + how it returns a probability.
  - open `signal_processing.py` and note SNR / baseline / phase / peak-pick helpers.
- [ ] Confirm you can run the app + tests locally (use the existing venv).

### Step 1 — The data contract (foundation)
- [ ] Add a `MetaboliteResult` schema (Pydantic model) — fields exactly as the
      contract above. Put it in a new `profile_schema.py` (or top of `main.py`).
- [ ] Add a small builder helper `make_result(...) -> MetaboliteResult`.
- [ ] **Check:** `py_compile` clean; you can construct one dummy result in a REPL.

### Step 2 — Stage 2 backbone: auto-profile (BUILD THIS FIRST of the stages)
- [ ] 2a. Write `auto_profile(spectrum_bytes) -> list[MetaboliteResult]` that
      chains the **existing** pieces found in Step 0:
      peak-pick → NNLS deconvolve → nmrformer assign → compute `confidence`
      (transparent formula: assignment_prob, 1−fit_residual, FDR-pass) → set
      `status` → fill `provenance`.
- [ ] 2b. Add `POST /profile/auto` returning `{spectrum_meta, metabolites:[...]}`.
- [ ] **Check:** on a bundled demo spectrum, returns ≥1 metabolite, each with
      `confidence`, `fdr`, `status`, `provenance` populated.

### Step 3 — Stage 1: QC gate (put in front of Stage 2)
- [ ] Add `POST /profile/qc` → `{snr, baseline_score, water_residual, referenced,
      verdict, reasons[]}` from `signal_processing` metrics + fixed thresholds.
- [ ] Make `/profile/auto` call QC first; if `verdict=="fail"`, return 400 unless
      `?override=true&reason=...`.
- [ ] **Check:** `verdict=="pass"` on a good demo; a deliberately bad input fails.

### Step 4 — Stage 3: triage buckets
- [ ] Ensure `auto_profile` sets `status` via thresholds (accept ≥hi, reject <lo,
      else review).
- [ ] Add `GET /profile/triage?hi=0.85&lo=0.5` returning
      `{accepted, review, rejected}`.
- [ ] **Check:** the three buckets partition the set — no overlap, no dropped items.

### Step 5 — Stage 4: quantify uncertainty
- [ ] Add bootstrap confidence interval to each concentration → fill
      `concentration.ci_low/ci_high`; surface µM-vs-a.u. calibration status.
- [ ] **Check:** `ci_low ≤ value ≤ ci_high` for every metabolite.

### Step 6 — Stage 5: NCD panel (already done — just confirm)
- [ ] Confirm `GET /ncd-screen` still works and returns honest AUCs. No new code.

### Step 7 — Stage 6: report + provenance (last)
- [ ] Add `GET /profile/report` assembling QC + accepted/edited metabolites +
      `/ncd-screen` + full provenance + `signed_by`/`signed_at`. Add CSV export.
- [ ] **Check:** report JSON contains all four sections and the sign-off fields.

### Step 8 — Tests + final verify
- [ ] Add a smoke test per new endpoint (qc, auto, triage, report) under `tests/`.
- [ ] Run: `backend/nmr_api/.venv/bin/python -m unittest discover -s backend/nmr_api/tests -q`
- [ ] `py_compile` the whole package. Update `HANDOFF.md` "implemented" list.

**Stop after Step 8** — that is the complete backend. UI wiring and the optional
H100 upgrades (field-upconverter, NN-quantifier) are separate, later tasks.
