#!/bin/bash
# RuuPhenome — NMRTransformer Setup Script
# Run once: bash setup.sh

set -e

echo "=== Step 1: Create Python virtual environment ==="
python3 -m venv nmr_env
source nmr_env/bin/activate

echo "=== Step 2: Install core dependencies ==="
pip install --upgrade pip
pip install numpy pandas scipy matplotlib seaborn jupyter ipykernel

echo "=== Step 3: Install RDKit (required by NMRTransformer) ==="
pip install rdkit

echo "=== Step 4: Install PyTorch (CPU version — upgrade to CUDA if GPU available) ==="
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "=== Step 5: Clone NMRTransformer from GitHub ==="
if [ ! -d "NMRTransformer" ]; then
  git clone https://github.com/liningtonlab/NMRTransformer.git
fi
cd NMRTransformer
pip install -e .
cd ..

echo "=== Step 6: Install additional metabolomics tools ==="
pip install nmrglue requests openpyxl

echo ""
echo "✓ Setup complete. Activate env with: source nmr_env/bin/activate"
echo "Then run: jupyter notebook nmr_analysis.ipynb"
