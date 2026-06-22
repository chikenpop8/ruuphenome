#!/bin/bash
# Install the open-source NMRTransformer model so /analyze predicts shifts
# for ANY SMILES (not just the HMDB fallback set).
# Usage (from a terminal with internet access): bash backend/nmr_api/setup_nmrtransformer.sh
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

PIP=".venv/bin/pip"
PYTHON=".venv/bin/python"

$PIP install -q --upgrade pip

echo "Installing project requirements..."
$PIP install -q -r requirements.txt

echo "Installing RDKit..."
$PIP install -q rdkit

# Do NOT reinstall torch — use the existing MPS/GPU build already in the venv.
$PYTHON -c "import torch; print('  torch ok:', torch.__version__)" 2>/dev/null \
  || { echo "Installing PyTorch..."; $PIP install -q torch; }

echo "Cloning + installing NMRTransformer..."
if [ ! -d "NMRTransformer" ]; then
  git clone --depth 1 https://github.com/liningtonlab/NMRTransformer.git
fi
$PIP install -q -e NMRTransformer

echo ""
echo "✓ NMRTransformer installed. /analyze will now report backend=NMRTransformer."
echo "  Restart the server (bash run.sh) to activate."
