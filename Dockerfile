# RuuPhenome — reproducible container for the BDI Hackathon 2026 VM.
#
# Builds a lean, torch-free image (~700 MB) that runs the full demo: Track 1
# metabolite profiling, Track 2 biomarker discovery, biology, and the web UI.
#
#   docker build -t ruuphenome .
#   docker run -p 8100:8100 ruuphenome
#
# Then open http://<VM-IP>:8100  (over the KKU VPN).
#
# Determinism: NMR_OFFLINE=1 blocks every outbound network call, so the demo is
# fully reproducible AND cannot send any dataset row off the VM (hackathon
# data-governance policy, Section 2). The single-thread env vars keep native
# math (numpy/scipy/sklearn) deterministic and avoid the macOS/OpenMP crash.

FROM python:3.12-slim

# --- runtime environment ---------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    NMR_OFFLINE=1 \
    NMR_HOST=0.0.0.0 \
    NMR_PORT=8100 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    KMP_DUPLICATE_LIB_OK=TRUE

WORKDIR /app

# --- dependencies (cached layer) -------------------------------------------
# Copy only the requirements first so `docker build` reuses this layer unless
# the dependency list changes. All wheels are prebuilt manylinux — no compiler
# needed. If a future package has no wheel, add build-essential here.
COPY backend/nmr_api/requirements-core.txt ./requirements-core.txt
RUN pip install --upgrade pip && pip install -r requirements-core.txt

# --- application code ------------------------------------------------------
COPY backend/ ./backend/

EXPOSE 8100

# The 'nmr_api' package resolves when uvicorn runs from inside backend/.
WORKDIR /app/backend

# Simple container healthcheck against the app's own /healthz endpoint.
HEALTHCHECK --interval=30s --timeout=4s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8100/healthz').status==200 else 1)"

CMD ["python", "-m", "uvicorn", "nmr_api.main:app", "--host", "0.0.0.0", "--port", "8100"]
