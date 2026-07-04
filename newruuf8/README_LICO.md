# RuuPhenome F8 — GISSMO quantifier training pack (LiCO / H100)

Self-contained pack to train the **F8 GISSMO transformer quantifier** (a ¹H spectrum →
per-compound relative concentration; identity = concentration above a threshold) on open,
physically-exact GISSMO-simulated mixtures. **It downloads nothing and touches no closed
/ patient data** — only the bundled open corpus.

## Contents
```
run_f8.sh                 # one-command training launcher (edit FIELD_MHZ)
requirements-f8.txt       # deps (prefer the node's preinstalled CUDA torch)
nmr_api/                  # the code (Python package)
nmr_api/open_data/
    gissmo_corpus.json            # TRAINING DATA — 94 GISSMO compounds (open)
    bmrb_experimental_peaks.json  # HELD-OUT REAL eval set — 18 BMRB experimental compounds
    bmrb_reference_shifts.json    # reference library (open)
```

## 1. What you're training
The GISSMO transformer quantifier (Conv patch-embed → Transformer encoder → softplus
per-compound relative concentration), on ~250k GISSMO-simulated ¹H mixtures generated on the
fly, with ppm-drift augmentation.

## 2. Why it needs the H100
The cost is the **data generation** (~250k mixtures/epoch × augmentation) and the transformer
over long (4096-bin) spectra + sweeps — not model size.

## 3. Setup (once)
```bash
# Use the node's existing CUDA python if it has torch; otherwise create an env:
python -m pip install -r requirements-f8.txt
# IMPORTANT: keep the node's CUDA build of torch. Only install torch if it is absent,
# and then use the CUDA wheel for the node's driver — a CPU wheel will not use the H100.
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # must print True
```

## 4. Run it
Edit the field to your competition spectrometer, then launch:
```bash
FIELD_MHZ=500 bash run_f8.sh            # full run (~a few hours on 1x H100)
# or, for a 1-GPU / 1-hour LiCO slot:
FIELD_MHZ=500 PROFILE=lico1h bash run_f8.sh
```
Equivalent explicit command (from the pack root):
```bash
python -m nmr_api.train_on_h100 --supervised gissmo-quant \
    --field-mhz 500 --mixtures 250000 --epochs 200 --batch-size 256 --n-bins 4096 --patch 16
```

## 5. Required inputs
`nmr_api/open_data/gissmo_corpus.json` (bundled). No network, no closed data.

## 6. Outputs (written to `nmr_api/models/`)
```
gissmo_quantifier.pt                        # the checkpoint
gissmo_quantifier_<epochs>ep_cuda_b256.pt   # config-tagged copy (auto-named)
gissmo_quantifier_report.json               # loss history + config
gissmo_quantifier_report_<...>.json         # tagged copy
```

## 7. Send back
Copy back **only** the two files (the `.pt` checkpoint + the `.json` report):
```
nmr_api/models/gissmo_quantifier*.pt
nmr_api/models/gissmo_quantifier_report*.json
```
**Never** copy back any patient / closed data (there is none in this pack — keep it that way).

## After it's back (on the dev machine)
Drop `gissmo_quantifier.pt` into `backend/nmr_api/models/`, then run the **independent real
held-out** benchmark so the trained quantifier is judged on BMRB experimental spectra it never
trained on:
```bash
python -m nmr_api.track1_benchmark --validate-bmrb
```

## Honest caveat
Training is on GISSMO simulated patterns (in-distribution for the quantifier). Its **fair
evaluation is on the BMRB experimental set** (a source it did NOT train on), and real held-out
F1 is still modest — this is research-use-only, not clinical validation. F8's job is to narrow
the sim-to-real gap and improve on deterministic matching under real shift/intensity variation.
