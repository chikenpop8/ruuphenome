"""RuuPhenome NMR API — metabolite recognition from NMR metabolomics data."""

# Set the macOS OpenMP-duplicate guard at the earliest possible point: Python
# imports this package (__init__) before importing nmr_api.main, so this runs
# before scikit-learn / XGBoost load their native libomp. Prevents the
# duplicate-libomp segfault on /biomarkers-model-suite. Harmless on Linux.
import os as _os
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
