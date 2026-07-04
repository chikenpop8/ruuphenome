#!/usr/bin/env bash
# RuuPhenome F8 — train the GISSMO transformer quantifier on 1x H100 (LiCO).
# Run from the pack root (the directory that contains ./nmr_api).
#
#   bash run_f8.sh                 # full run (a few hours on 1x H100)
#   FIELD_MHZ=600 bash run_f8.sh   # set your competition spectrometer field
#   PROFILE=lico1h bash run_f8.sh  # small run for a 1-GPU / 1-hour slot
#
# It downloads NOTHING and reads NO closed data — only the bundled open corpus
# nmr_api/open_data/gissmo_corpus.json.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python}"                      # override with PYTHON=/path/to/python
FIELD_MHZ="${FIELD_MHZ:-500}"               # <-- SET THIS to the competition field (MHz)
PROFILE="${PROFILE:-full}"

if [ "$PROFILE" = "lico1h" ]; then          # 1-GPU / 1-hour slot
  MIX=40000; EPOCHS=60
else                                        # full run
  MIX=250000; EPOCHS=200
fi

echo "RuuPhenome F8 — GISSMO quantifier"
echo "  python      : $($PY -c 'import sys;print(sys.executable)')"
echo "  torch/CUDA  : $($PY -c 'import torch;print(torch.__version__, torch.cuda.is_available())' 2>/dev/null || echo 'torch not importable')"
echo "  field (MHz) : $FIELD_MHZ   profile: $PROFILE   mixtures: $MIX   epochs: $EPOCHS"
echo

$PY -m nmr_api.train_on_h100 --supervised gissmo-quant \
    --field-mhz "$FIELD_MHZ" --mixtures "$MIX" --epochs "$EPOCHS" \
    --batch-size 256 --n-bins 4096 --patch 16

echo
echo "=== DONE. Send back ONLY these files (open data only, never patient data): ==="
ls -lh nmr_api/models/gissmo_quantifier*.pt nmr_api/models/gissmo_quantifier_report*.json 2>/dev/null || true
