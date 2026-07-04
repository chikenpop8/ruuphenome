"""RuuPhenome NMR API — metabolite recognition from NMR metabolomics data."""

# Stabilize native math libraries at the earliest possible point: Python imports
# this package (__init__) before importing nmr_api.main, so these run before
# scikit-learn / XGBoost load their native OpenMP. On macOS, XGBoost and sklearn
# each ship their own libomp; multi-threaded parallel regions intermittently
# segfault the server (silent crash on /biomarkers-model-suite). Forcing
# single-threaded OpenMP removes the conflicting threads entirely — datasets here
# are small so there is no meaningful speed cost. Harmless on Linux (the VM).
import os as _os
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    _os.environ.setdefault(_var, "1")
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "0.1.0"
