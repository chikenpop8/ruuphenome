#!/bin/bash
# Start the RuuPhenome NMR API (dev mode, auto-reload).
# Usage: bash run.sh
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi
source .venv/bin/activate

pip install -q --upgrade pip
pip install -q -r requirements.txt

# The macOS XGBoost wheel expects Homebrew's libomp. Reuse the compatible
# runtime already bundled with scikit-learn when Homebrew is absent.
if [ "$(uname)" = "Darwin" ] && ! python -c "import xgboost" >/dev/null 2>&1; then
  XGB_LIB="$(python -c 'import site; from pathlib import Path; print(next(str(x) for p in site.getsitepackages() for x in Path(p).glob("xgboost/lib/libxgboost.dylib")))')"
  SKLEARN_OMP="$(python -c 'import sklearn; from pathlib import Path; print(Path(sklearn.__file__).parent / ".dylibs" / "libomp.dylib")')"
  if [ -f "$XGB_LIB" ] && [ -f "$SKLEARN_OMP" ]; then
    install_name_tool -add_rpath "$(dirname "$SKLEARN_OMP")" "$XGB_LIB" 2>/dev/null || true
  fi
fi

echo ""
echo "Starting NMR API on http://127.0.0.1:8100"
echo "Interactive docs:  http://127.0.0.1:8100/docs"
echo ""

# Activate the local SSL-encoder NMRformer adapter (hybrid assignment mode).
export NMRFORMER_ADAPTER_MODULE=nmr_api.nmrformer_adapter

# Run from backend/ so the 'nmr_api' package imports resolve.
cd ..
exec python -m uvicorn nmr_api.main:app --reload --host 127.0.0.1 --port 8100
