# Model checkpoints (committed — required for reproducibility)

These checkpoints are **committed to the repo** so a clean clone reproduces the app's
headline behaviour without a training step. All are derived from **open data only**
(BMRB / GISSMO / synthetic superpositions of open reference shifts) and are
**research-use-only (RUO)** — no closed competition data is used or embedded.

| File | Role | Trained on | Serve path |
|------|------|-----------|------------|
| `pscnn_identifier.pt` | pSCNN learned identification channel (the "hybrid" = FDR ∪ pSCNN) | 30-compound routine ¹H panel, 5000 synthetic mixtures of open reference shifts (seed 2026, 25 epochs) | Loaded by `main._load_pscnn()`; blended in `_identification_channels()` |
| `pscnn_identifier_report.json` | Training metadata for the above (panel, epochs, loss, device) | — | Surfaced context only |
| `masked_nmr_encoder.pt` | Self-supervised masked-spectrum encoder (SSL pretrain artifact) | open BMRB ¹H reference spectra | Not on the serve path today (exhibit / future fine-tune init) |
| `masked_nmr_training.json` | Training metadata for the SSL encoder | — | — |

## Why the pSCNN checkpoint is committed

`main._load_pscnn()` only enables the hybrid identification channel when
`models/pscnn_identifier.pt` exists. Without it the app **silently degrades to
deterministic-only** (NNLS + target–decoy FDR) — the headline "deterministic + pSCNN
(hybrid)" story would not reproduce from a fresh clone. `pscnn.status()` now reports
this state explicitly (`hybrid_active` + a `note`) via `/plugins`, so degradation is
never silent.

## Regenerating `pscnn_identifier.pt`

Deterministic (seed-pinned) — reproduces an equivalent checkpoint from open data:

```bash
# from the repo root, inside the backend venv
python -m nmr_api.train_on_h100 --supervised pscnn      # CUDA/MPS/CPU auto-selected
# writes models/pscnn_identifier.pt + pscnn_identifier_report.json
```

The learned channel is a **precision-respecting recall booster**, not a standalone
classifier: the trustworthy identifications are the target–decoy FDR-confirmed set;
pSCNN only adds hybrid candidates. Best real held-out score (BMRB experimental peak
lists): hybrid **F1 ≈ 0.53 @ precision 0.70** — reported by `track1_benchmark.py`, not
by this checkpoint's own report.

## Governance

Open data only; RUO. Do **not** commit any checkpoint trained on closed competition
data — those stay inside the governed VM (see `finetune_loader.py`, `NMR_OFFLINE=1`).
