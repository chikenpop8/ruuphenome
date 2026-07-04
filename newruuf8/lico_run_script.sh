#!/bin/bash
# ============================================================================
# RuuPhenome F8 — GISSMO quantifier — LiCO Common Job "Run Script"
# Paste this whole block into the Run Script box. Set FIELD_MHZ below.
# Resource Options: 1 node, 1 GPU (H100), a GPU queue/partition.
# ============================================================================
set -uo pipefail

# --- 0. SET THIS -------------------------------------------------------------
FIELD_MHZ=500          # <-- competition spectrometer field in MHz (ASK: 500 / 600 / 700?)
PROFILE=full           # "full" (~a few hours) or "lico1h" (fits a 1-GPU/1-hour slot)

# --- 1. environment ----------------------------------------------------------
# If your cluster uses modules/conda for CUDA PyTorch, load it here (ask your
# admin for exact names). A CPU-only torch will NOT use the H100. Examples:
#   module load anaconda3 cuda pytorch
#   source activate pytorch
PY="${PYTHON:-python}"
echo "python: $($PY -c 'import sys;print(sys.executable)' 2>&1)"
$PY -c 'import torch;print("torch",torch.__version__,"| CUDA available:",torch.cuda.is_available())' \
  || { echo "ERROR: torch not importable — load a CUDA PyTorch module/conda env above."; exit 1; }

# --- 2. locate the pack ------------------------------------------------------
# Works whether the Workspace IS the pack dir, or just contains the tarball.
if [ ! -d nmr_api ]; then
  [ -f ruuphenome_f8_lico_pack.tar.gz ] && tar -xzf ruuphenome_f8_lico_pack.tar.gz
  [ -d ruuphenome_f8_lico_pack ] && cd ruuphenome_f8_lico_pack
fi
[ -d nmr_api ] || { echo "ERROR: nmr_api/ not found in the Workspace."; exit 1; }

# --- 3. light deps (torch/CUDA already provided above) -----------------------
$PY -m pip install --user -q numpy scipy pandas scikit-learn requests || \
  echo "warn: pip install skipped/failed — continuing (deps may already be present)."

# --- 4. train ----------------------------------------------------------------
if [ "$PROFILE" = "lico1h" ]; then MIX=40000; EPOCHS=60; else MIX=250000; EPOCHS=200; fi
echo "=== training: field ${FIELD_MHZ} MHz · ${MIX} mixtures/epoch · ${EPOCHS} epochs ==="
$PY -m nmr_api.train_on_h100 --supervised gissmo-quant \
    --field-mhz "$FIELD_MHZ" --mixtures "$MIX" --epochs "$EPOCHS" \
    --batch-size 256 --n-bins 4096 --patch 16

# --- 5. show what to send back ----------------------------------------------
echo "=== DONE — download ONLY these (open-data outputs; no patient data): ==="
ls -lh nmr_api/models/gissmo_quantifier*.pt nmr_api/models/gissmo_quantifier_report*.json
