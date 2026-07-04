# RuuPhenome NMR API — production container.
#
# Multi-stage build: the builder stage installs everything into a venv using
# CPU-only torch (the default PyPI wheel bundles CUDA and is ~2x larger; the
# app runs on CPU/nmrglue/sklearn regardless, so CUDA wheels buy nothing here).
# The runtime stage copies only that venv + app code onto a slim base, so the
# compiler toolchain and pip cache never end up in the final image.

FROM python:3.13.2-slim AS builder

# Build deps for scipy/xgboost/nmrglue native extensions.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY nmr_api/requirements.lock.txt ./requirements.lock.txt

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# CPU-only torch first (from its dedicated index), then everything else from
# the pinned lockfile — this is what keeps the image from ballooning with CUDA.
RUN pip install --no-cache-dir --upgrade pip \
    && grep '^torch==' requirements.lock.txt | xargs -I{} pip install --no-cache-dir \
       --index-url https://download.pytorch.org/whl/cpu "{}" \
    && grep -v '^torch==' requirements.lock.txt > requirements.notorch.txt \
    && pip install --no-cache-dir -r requirements.notorch.txt


FROM python:3.13.2-slim AS runtime

# Native runtime libs the wheels above link against (no compilers needed here).
RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0 libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 ruuphenome

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    NMRFORMER_ADAPTER_MODULE=nmr_api.nmrformer_adapter

WORKDIR /app
COPY nmr_api ./nmr_api
RUN chown -R ruuphenome:ruuphenome /app
USER ruuphenome

EXPOSE 8100
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request as u; u.urlopen('http://127.0.0.1:8100/healthz', timeout=3)" || exit 1

CMD ["python", "-m", "uvicorn", "nmr_api.main:app", "--host", "0.0.0.0", "--port", "8100"]
