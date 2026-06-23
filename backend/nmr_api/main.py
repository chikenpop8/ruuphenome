from __future__ import annotations

import json
import os
from pathlib import Path

if not os.environ.get("LOKY_MAX_CPU_COUNT"):
    os.environ["LOKY_MAX_CPU_COUNT"] = "1"

# macOS: XGBoost and scikit-learn each ship their own OpenMP runtime. When
# XGBoost runs inside the server's worker thread, the duplicate libomp load
# hard-crashes (segfault) the process. Allow the duplicate load. Must be set
# before numpy/sklearn/xgboost import below. Harmless on Linux.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

try:
    from . import (
        biology,
        biomarker_engine,
        biomarkers,
        dimensionality,
        laboratory_workflow,
        library,
        model_suite,
        nmrformer_backend,
        open_data,
        pipeline,
        self_supervised,
        signal_processing,
        spectral_cohort,
    )
    from .models import AnalysisResponse, HealthResponse, LaboratoryQCRequest
    from .shifts_db import HMDB_KNOWN_SHIFTS, nmrtransformer_available, predict_shifts
except ImportError:  # pragma: no cover - direct script execution fallback
    import biology  # type: ignore
    import biomarker_engine  # type: ignore
    import biomarkers  # type: ignore
    import dimensionality  # type: ignore
    import laboratory_workflow  # type: ignore
    import library  # type: ignore
    import model_suite  # type: ignore
    import nmrformer_backend  # type: ignore
    import open_data  # type: ignore
    import pipeline  # type: ignore
    import self_supervised  # type: ignore
    import signal_processing  # type: ignore
    import spectral_cohort  # type: ignore
    from models import AnalysisResponse, HealthResponse, LaboratoryQCRequest  # type: ignore
    from shifts_db import HMDB_KNOWN_SHIFTS, nmrtransformer_available, predict_shifts  # type: ignore


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

# Default demo dataset (MTBLS242). Override with NMR_DEFAULT_TSV env var.
DEFAULT_TSV = Path(
    os.environ.get(
        "NMR_DEFAULT_TSV",
        str(APP_DIR / "open_data" / "demo_mtbls242.tsv"),
    )
)

# Bundled real-disease dataset: MTBLS1 (human urine NMR, type-2 diabetes vs
# control). Labels come from the study sample sheet, not the sample IDs.
MTBLS1_TSV = APP_DIR / "open_data" / "demo_mtbls1.tsv"
MTBLS1_LABELS = APP_DIR / "open_data" / "demo_mtbls1_labels.json"

# Registry of selectable demo datasets for the UI switcher.
DATASETS = {
    "mtbls242": {
        "label": "MTBLS242 — gastric-bypass time series (time-point 0 vs 4)",
        "kind": "longitudinal",
        "tsv": DEFAULT_TSV,
    },
    "mtbls1": {
        "label": "MTBLS1 — type-2 diabetes vs control (urine NMR)",
        "kind": "labeled",
        "tsv": MTBLS1_TSV,
        "labels": MTBLS1_LABELS,
        "task": "diabetes vs control",
        "class_names": {0: "control", 1: "diabetes"},
    },
}


def _load_label_map() -> dict:
    """Sample-name → 0/1 disease label for the MTBLS1 dataset."""
    return {k: int(v) for k, v in json.loads(MTBLS1_LABELS.read_text()).items()}

app = FastAPI(
    title="RuuPhenome NMR API",
    version="0.1.0",
    description=(
        "NMR metabolite recognition for the National Phenome Institute. "
        "Predicts ¹H chemical shifts (NMRTransformer, open-source) and matches "
        "them against annotated spectrum peaks to identify and rank metabolites."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/healthz", response_model=HealthResponse)
def healthz():
    available = nmrtransformer_available()
    return HealthResponse(
        ok=True,
        backend="NMRTransformer" if available else "HMDB-fallback",
        nmrtransformer_available=available,
        notes=(
            "NMRTransformer ready — predicting shifts for arbitrary SMILES."
            if available
            else "NMRTransformer not installed; using HMDB known-shift fallback. "
                 "Run backend/nmr_api/setup_nmrtransformer.sh to enable full prediction."
        ),
    )


@app.get("/plugins")
def plugins():
    """Report the status of every connected backend plugin."""
    def _ver(mod):
        try:
            m = __import__(mod)
            return getattr(m, "__version__", "installed")
        except Exception:
            return None
    return {
        "domain1_signal_processing": {
            "nmrglue": _ver("nmrglue"),
            "scipy": _ver("scipy"),
            "ready": signal_processing.NMRGLUE,
            "capabilities": [
                "Bruker digital-filter removal",
                "zero/first-order auto-phase",
                "ALS baseline correction",
                "robust peak picking",
                "quality control",
            ],
        },
        "domain1_peak_assignment": {
            "active": (
                "hybrid-pattern+nmrformer"
                if nmrformer_backend.status()["available"]
                else "reference-pattern-matcher"
            ),
            "nmrformer": nmrformer_backend.status(),
            "notes": (
                "Hybrid inference activates only when a validated local NMRformer "
                "adapter is configured; otherwise transparent pattern matching is used."
            ),
        },
        "domain1_self_supervised": self_supervised.status(),
        "shift_prediction": {
            "nmrtransformer": nmrtransformer_available(),
            "fallback": "HMDB known-shift table",
            "active": "NMRTransformer" if nmrtransformer_available() else "HMDB-fallback",
        },
        "domain2_biomarkers": {
            "scikit_learn": _ver("sklearn"),
            "xgboost": _ver("xgboost"),
            "catboost": _ver("catboost"),
            "deep_learning": _ver("torch"),
            "ready": _ver("sklearn") is not None,
            "models": model_suite.dependency_status(),
        },
        "domain2_dimensionality": dimensionality.dependency_status(),
        "structure_rendering": {"smilesdrawer": "frontend (CDN)", "rdkit": _ver("rdkit")},
        "web": {"fastapi": _ver("fastapi"), "pandas": _ver("pandas"), "numpy": _ver("numpy")},
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    nmr_results_tsv: UploadFile = File(
        ..., description="MetaboLights NMR results TSV (Domain 2), e.g. MTBLS242"
    ),
    tolerance_ppm: float = Form(default=0.05),
):
    """
    Upload a Domain 2 NMR results table and receive ranked metabolite matches
    scored against the default Domain 1 annotated peak list.
    """
    if not nmr_results_tsv.filename.lower().endswith((".tsv", ".txt", ".csv")):
        raise HTTPException(
            status_code=400,
            detail="Expected a .tsv/.txt/.csv MetaboLights results table.",
        )

    raw = await nmr_results_tsv.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        result = pipeline.analyze(raw, tolerance_ppm=tolerance_ppm)
    except Exception as exc:  # pragma: no cover - surfaces parse errors to client
        raise HTTPException(status_code=422, detail=f"Failed to analyze: {exc}")

    return result


@app.get("/domain1-peaks")
def domain1_peaks():
    """Return the default Domain 1 annotated peak list used for scoring."""
    return {"peaks": pipeline.DEFAULT_DOMAIN1_PEAKS}


@app.get("/laboratory-workflow")
def get_laboratory_workflow():
    """Return the end-to-end real-laboratory workflow and release gates."""
    return laboratory_workflow.workflow()


@app.post("/laboratory-workflow/evaluate-qc")
def evaluate_laboratory_qc(observations: LaboratoryQCRequest):
    """Evaluate spectrum and batch observations against visible RUO defaults."""
    return laboratory_workflow.evaluate_qc(**observations.model_dump())


@app.get("/compounds")
def compounds():
    """
    Compound library (Reference Card + compound table data) built from the
    default demo dataset. Powers the Profiler UI on first load.
    """
    if not DEFAULT_TSV.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Default dataset not found at {DEFAULT_TSV}. "
                   "Set NMR_DEFAULT_TSV or use POST /compounds-upload.",
        )
    return library.build_library(DEFAULT_TSV.read_bytes())


@app.post("/compounds-upload")
async def compounds_upload(nmr_results_tsv: UploadFile = File(...)):
    """Build the compound library from an uploaded MetaboLights TSV."""
    raw = await nmr_results_tsv.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        return library.build_library(raw)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=422, detail=f"Failed to parse: {exc}")


# ── Domain 1: NMR signal processing (nmrglue) ──────────────────────────────
def _domain1_reference_library():
    """Return shifts, abundance weights and human-readable compound names."""
    if not DEFAULT_TSV.exists():
        return HMDB_KNOWN_SHIFTS, None, {}
    df_meta, _abundance, _samples = pipeline.load_domain2(DEFAULT_TSV.read_bytes())
    smiles = df_meta["smiles"].fillna("").tolist()
    shifts, _backend = predict_shifts(smiles)
    abundance = {}
    names = {}
    for _, row in df_meta.iterrows():
        smiles_key = row.get("smiles", "")
        if not isinstance(smiles_key, str) or not smiles_key:
            continue
        value = row.get("mean_abundance", 1.0)
        abundance[smiles_key] = float(value) if value and value == value else 1.0
        names[smiles_key] = str(row.get("metabolite_identification", smiles_key))
    normalizer = max(abundance.values()) if abundance else 1.0
    abundance = {key: 0.3 + value / normalizer for key, value in abundance.items()}
    return shifts, abundance, names


@app.get("/demo-spectrum")
def demo_spectrum():
    """
    Synthesize and process a realistic serum ¹H spectrum from the reference
    library — demonstrates the full nmrglue/scipy Domain 1 pipeline without a
    physical FID file. Returns ppm axis, intensity, and picked peaks.
    """
    shifts, abundance, names = _domain1_reference_library()
    return signal_processing.demo_spectrum(shifts, abundance, names)


@app.post("/process-fid")
async def process_fid(
    bruker_zip: UploadFile = File(...),
    line_broadening_hz: float = Form(default=0.3),
    zero_fill_factor: int = Form(default=2),
    snr_threshold: float = Form(default=5.0),
    tolerance_ppm: float = Form(default=0.04),
    assignment_backend: str = Form(default="hybrid"),
):
    """
    Upload a zipped Bruker experiment folder (containing an 'fid' file) and run
    the nmrglue → FFT → phase → baseline → peak-pick pipeline.
    """
    if not signal_processing.NMRGLUE:
        raise HTTPException(status_code=501, detail="nmrglue not installed.")
    if not 0 <= line_broadening_hz <= 10:
        raise HTTPException(status_code=400, detail="line_broadening_hz must be between 0 and 10.")
    if zero_fill_factor not in (1, 2, 4, 8):
        raise HTTPException(status_code=400, detail="zero_fill_factor must be 1, 2, 4, or 8.")
    if not 2 <= snr_threshold <= 50:
        raise HTTPException(status_code=400, detail="snr_threshold must be between 2 and 50.")
    if assignment_backend not in ("hybrid", "pattern-matcher", "nmrformer"):
        raise HTTPException(
            status_code=400,
            detail="assignment_backend must be hybrid, pattern-matcher, or nmrformer.",
        )
    if assignment_backend == "nmrformer" and not nmrformer_backend.status()["available"]:
        raise HTTPException(
            status_code=501,
            detail=nmrformer_backend.status()["reason"],
        )
    raw = await bruker_zip.read()
    try:
        fid, meta = signal_processing.read_bruker_zip(raw)
        shifts, _abundance, names = _domain1_reference_library()
        result = signal_processing.process_fid(
            fid,
            meta["sw_hz"],
            meta["sf_mhz"],
            lb_hz=line_broadening_hz,
            carrier_ppm=meta["carrier_ppm"],
            zero_fill_factor=zero_fill_factor,
            snr=snr_threshold,
            reference_shifts=shifts,
            compound_names=names,
            tolerance_ppm=tolerance_ppm,
            assignment_backend=assignment_backend,
        )
        result["source"] = bruker_zip.filename
        result["acquisition"] = meta
        return result
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"FID processing failed: {exc}")


@app.post("/process-spectrum")
async def process_spectrum(
    spectrum_file: UploadFile = File(...),
    snr_threshold: float = Form(default=5.0),
    tolerance_ppm: float = Form(default=0.04),
    assignment_backend: str = Form(default="hybrid"),
):
    """
    Upload a processed two-column CSV/TSV (ppm, intensity), then run baseline
    correction, peak picking, library assignment and quality control.
    """
    filename = spectrum_file.filename or ""
    if not filename.lower().endswith((".csv", ".tsv", ".txt")):
        raise HTTPException(status_code=400, detail="Expected a .csv, .tsv, or .txt spectrum.")
    if not 2 <= snr_threshold <= 50:
        raise HTTPException(status_code=400, detail="snr_threshold must be between 2 and 50.")
    if assignment_backend not in ("hybrid", "pattern-matcher", "nmrformer"):
        raise HTTPException(
            status_code=400,
            detail="assignment_backend must be hybrid, pattern-matcher, or nmrformer.",
        )
    if assignment_backend == "nmrformer" and not nmrformer_backend.status()["available"]:
        raise HTTPException(
            status_code=501,
            detail=nmrformer_backend.status()["reason"],
        )
    raw = await spectrum_file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded spectrum is empty.")
    try:
        ppm, intensity = signal_processing.parse_processed_spectrum(raw, filename)
        shifts, _abundance, names = _domain1_reference_library()
        result = signal_processing.analyze_spectrum(
            ppm,
            intensity,
            snr=snr_threshold,
            reference_shifts=shifts,
            compound_names=names,
            tolerance_ppm=tolerance_ppm,
            processing_steps=["uploaded processed numeric spectrum"],
            assignment_backend=assignment_backend,
        )
        result["source"] = filename
        return result
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Spectrum processing failed: {exc}")


@app.get("/open-data")
def open_data_status():
    """Report the curated open BMRB corpus and complete provenance."""
    manifest = open_data.load_manifest()
    summary_path = open_data.DATA_DIR / "corpus_summary.json"
    summary = (
        json.loads(summary_path.read_text())
        if summary_path.exists()
        else None
    )
    return {
        "manifest": manifest,
        "corpus_built": open_data.CORPUS_PATH.exists(),
        "summary": summary,
        "provenance_path": str(open_data.PROVENANCE_PATH),
    }


@app.post("/open-data/build")
def build_open_data(force_download: bool = Form(default=False)):
    """Download, checksum and process the curated open BMRB corpus."""
    try:
        return open_data.build_processed_corpus(force_download)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Open-data build failed: {exc}")


@app.get("/self-supervised/status")
def self_supervised_status():
    return self_supervised.status()


@app.post("/self-supervised/train")
def train_self_supervised(
    epochs: int = Form(default=20),
    steps_per_epoch: int = Form(default=32),
    batch_size: int = Form(default=16),
):
    """Train masked-spectrum reconstruction on unlabeled augmented BMRB spectra."""
    if not open_data.CORPUS_PATH.exists():
        open_data.build_processed_corpus()
    try:
        return self_supervised.train(
            epochs=max(1, min(epochs, 200)),
            steps_per_epoch=max(1, min(steps_per_epoch, 500)),
            batch_size=max(2, min(batch_size, 128)),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422, detail=f"Self-supervised training failed: {exc}"
        )


# ── Domain 2: biomarker discovery (scikit-learn) ───────────────────────────
@app.get("/biomarkers")
def biomarkers_default(group_a: int | None = None, group_b: int | None = None):
    """Run biomarker discovery on the default dataset (time-point contrast)."""
    if not DEFAULT_TSV.exists():
        raise HTTPException(status_code=404, detail="Default dataset not found.")
    try:
        return biomarkers.discover(DEFAULT_TSV.read_bytes(), group_a, group_b)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Discovery failed: {exc}")


@app.post("/biomarkers-upload")
async def biomarkers_upload(
    nmr_results_tsv: UploadFile = File(...),
    group_a: int | None = Form(default=None),
    group_b: int | None = Form(default=None),
):
    """Run biomarker discovery on an uploaded MetaboLights TSV."""
    raw = await nmr_results_tsv.read()
    try:
        return biomarkers.discover(raw, group_a, group_b)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Discovery failed: {exc}")


# ── Domain 2 (best version): leakage-safe p>>n discovery ───────────────────
def _biomarker_task(raw: bytes, group_a: int | None, group_b: int | None):
    """Build one binary longitudinal task with patient IDs for grouped CV."""
    import numpy as np
    import pandas as pd

    X, _names, time_groups = biomarkers.build_matrix(raw)
    available = sorted(set(time_groups.values()))
    if len(available) < 2:
        raise ValueError("Need at least two sample groups.")
    if group_a is None or group_b is None:
        group_a, group_b = available[0], available[-1]
    if group_a == group_b or group_a not in available or group_b not in available:
        raise ValueError(f"Choose two different groups from {available}.")
    rows_a = [sample for sample, group in time_groups.items() if group == group_a]
    rows_b = [sample for sample, group in time_groups.items() if group == group_b]
    selected_samples = rows_a + rows_b
    matrix = pd.concat([X.loc[rows_a], X.loc[rows_b]]).dropna(axis=1, how="all")
    y = np.array([0] * len(rows_a) + [1] * len(rows_b))
    patient_map = biomarkers.sample_patient_groups(selected_samples)
    patients = np.array([patient_map[sample] for sample in selected_samples])
    return matrix, y, patients, group_a, group_b


def _labeled_biomarker_task(raw: bytes, label_map: dict):
    """
    Build a binary task from an explicit sample→label map (cross-sectional
    disease-vs-control study, e.g. MTBLS1). Each sample is its own group, so
    grouped CV reduces to stratified CV (no repeated-patient leakage to control).
    """
    import numpy as np

    X, _names, _groups = biomarkers.build_matrix(raw)
    rows = [s for s in X.index if s in label_map]
    if len(set(label_map[s] for s in rows)) < 2:
        raise ValueError("Need both classes present in the labeled samples.")
    matrix = X.loc[rows].dropna(axis=1, how="all")
    y = np.array([label_map[s] for s in rows])
    patients = np.array(rows)            # one sample = one patient
    return matrix, y, patients


def _dimensionality_task(
    raw: bytes,
    group_a: int | None,
    group_b: int | None,
    *,
    include_umap: bool,
    n_neighbors: int,
    min_dist: float,
):
    """Create an explicitly exploratory PCA/UMAP view of one Domain 2 task."""
    Xs, y, patients, group_a, group_b = _biomarker_task(
        raw, group_a, group_b
    )
    labels = [
        f"time-point {group_a}" if value == 0 else f"time-point {group_b}"
        for value in y
    ]
    result = dimensionality.project(
        Xs.values,
        labels,
        sample_names=list(Xs.index),
        patient_ids=patients,
        feature_names=list(Xs.columns),
        include_umap=include_umap,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )
    result["task"] = f"time-point {group_a} vs {group_b}"
    result["outcome_warning"] = (
        "This dataset measures longitudinal surgery time points, not disease vs control."
    )
    return result


@app.get("/datasets")
def list_datasets():
    """Selectable bundled demo datasets for the UI switcher."""
    return {
        "datasets": [
            {"id": k, "label": v["label"], "kind": v["kind"],
             "available": Path(v["tsv"]).exists()}
            for k, v in DATASETS.items()
        ]
    }


@app.get("/biomarkers-safe")
def biomarkers_safe(group_a: int | None = None, group_b: int | None = None,
                    k: int = 8, repeats: int = 3, dataset: str = "mtbls242"):
    """
    Leakage-safe biomarker discovery (fold-internal selection + FDR + nested CV
    + Top-k stability). Reports honest AUC/F1, the leaky AUC for comparison, and
    the stable panel. `dataset=mtbls1` runs the real diabetes-vs-control study.
    """
    if dataset == "mtbls1":
        if not MTBLS1_TSV.exists():
            raise HTTPException(status_code=404, detail="MTBLS1 dataset not bundled.")
        Xs, y, patients = _labeled_biomarker_task(
            MTBLS1_TSV.read_bytes(), _load_label_map()
        )
        task = DATASETS["mtbls1"]["task"]
    else:
        if not DEFAULT_TSV.exists():
            raise HTTPException(status_code=404, detail="Default dataset not found.")
        Xs, y, patients, group_a, group_b = _biomarker_task(
            DEFAULT_TSV.read_bytes(), group_a, group_b
        )
        task = f"time-point {group_a} vs {group_b}"
    res = biomarker_engine.discover(
        Xs.values,
        y,
        k=k,
        repeats=repeats,
        feature_names=list(Xs.columns),
        groups=patients,
    )
    res["task"] = task
    res["dataset"] = dataset
    res["n_patients"] = int(len(set(patients)))
    # Biological interpretation of the stable panel (per-metabolite + pathway enrichment)
    res["biological_interpretation"] = biology.interpret_panel(
        res.get("stable_panel", []),
        background=list(Xs.columns),
    )
    return res


# ── Track 1: binned-spectra cohort pipeline (annotate → visualize → bridge) ──
def _run_cohort_pipeline(X, bin_ppm, label_map=None, normalize="pqn",
                         include_biomarkers=True):
    """Shared engine: binned matrix → normalize → annotate → (optional) Track 2."""
    import numpy as np
    Xn = (spectral_cohort.pqn_normalize(X) if normalize == "pqn"
          else spectral_cohort.total_area_normalize(X) if normalize == "total_area"
          else X)
    ann = spectral_cohort.annotate(Xn, bin_ppm)
    M = ann.pop("annotated_matrix")
    out = {
        "annotation": ann,
        "visualization": spectral_cohort.overlay_traces(Xn),
        "normalization": normalize,
        "sample_metabolite_shape": list(M.shape),
    }
    if include_biomarkers and label_map and not M.empty:
        rows = [s for s in M.index if str(s) in label_map]
        if len(set(label_map[str(s)] for s in rows)) >= 2 and len(rows) >= 8:
            sub = M.loc[rows]
            y = np.array([label_map[str(s)] for s in rows])
            groups = np.array([str(s) for s in rows])
            disc = biomarker_engine.discover(
                sub.values, y, k=8, repeats=3,
                feature_names=list(sub.columns), groups=groups)
            disc["biological_interpretation"] = biology.interpret_panel(
                disc.get("stable_panel", []), background=list(sub.columns))
            out["biomarkers"] = disc
    return out


@app.get("/spectral/demo-pipeline")
def spectral_demo_pipeline():
    """
    Self-contained Track-1 → Track-2 demo: simulate a binned ¹H cohort, annotate
    bins → metabolites, then run biomarker discovery + biological interpretation.
    Shows the full automated pipeline without any upload.
    """
    X, bin_ppm, labels = spectral_cohort.make_demo_binned()
    result = _run_cohort_pipeline(X, bin_ppm, label_map=labels, normalize="pqn")
    result["task"] = "demo: simulated case vs control (BCAA signal planted)"
    result["note"] = "Synthetic cohort for pipeline demonstration, not real data."
    return result


@app.post("/spectral/annotate")
async def spectral_annotate(
    binned_matrix: UploadFile = File(...),
    normalize: str = Form(default="pqn"),
):
    """
    Upload a binned NMR matrix (sample × ppm-bin). Returns metabolite annotation
    (bins → metabolites), multi-sample overlay traces, and the sample × metabolite
    table that feeds Track 2.
    """
    raw = await binned_matrix.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        X, bin_ppm = spectral_cohort.load_binned_matrix(raw)
        return _run_cohort_pipeline(X, bin_ppm, normalize=normalize,
                                    include_biomarkers=False)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Annotation failed: {exc}")


@app.post("/spectral/pipeline")
async def spectral_pipeline(
    binned_matrix: UploadFile = File(...),
    metadata: UploadFile = File(...),
    normalize: str = Form(default="pqn"),
    label_column: str | None = Form(default=None),
):
    """
    Full automated pipeline: binned matrix + metadata (Table 2) → annotate →
    derive labels → biomarker discovery → biological interpretation.
    """
    raw = await binned_matrix.read()
    meta_raw = await metadata.read()
    if not raw or not meta_raw:
        raise HTTPException(status_code=400, detail="Both files are required.")
    try:
        X, bin_ppm = spectral_cohort.load_binned_matrix(raw)
        meta = spectral_cohort.parse_metadata(meta_raw)
        label_map, info = spectral_cohort.derive_labels(
            meta, [str(s) for s in X.index], label_column=label_column)
        result = _run_cohort_pipeline(X, bin_ppm, label_map=label_map,
                                      normalize=normalize)
        result["label_info"] = info
        return result
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Pipeline failed: {exc}")


@app.post("/track2/discover-with-metadata")
async def track2_with_metadata(
    metabolite_table: UploadFile = File(...),
    metadata: UploadFile = File(...),
    label_column: str | None = Form(default=None),
):
    """
    Track 2 on a processed table + separate metadata file. Joins Table 1
    (metabolite × sample) to Table 2 (metadata) on sample ID, derives the binary
    task, and runs leakage-safe discovery + biology.
    """
    import numpy as np
    raw = await metabolite_table.read()
    meta_raw = await metadata.read()
    if not raw or not meta_raw:
        raise HTTPException(status_code=400, detail="Both files are required.")
    try:
        X, _names, _g = biomarkers.build_matrix(raw)     # samples × metabolites
        meta = spectral_cohort.parse_metadata(meta_raw)
        label_map, info = spectral_cohort.derive_labels(
            meta, [str(s) for s in X.index], label_column=label_column)
        rows = [s for s in X.index if str(s) in label_map]
        if len(set(label_map[str(s)] for s in rows)) < 2:
            raise ValueError("Need two classes among matched samples.")
        sub = X.loc[rows].dropna(axis=1, how="all")
        y = np.array([label_map[str(s)] for s in rows])
        groups = np.array([str(s) for s in rows])
        disc = biomarker_engine.discover(
            sub.values, y, k=8, repeats=3,
            feature_names=list(sub.columns), groups=groups)
        disc["label_info"] = info
        disc["biological_interpretation"] = biology.interpret_panel(
            disc.get("stable_panel", []), background=list(sub.columns))
        return disc
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Discovery failed: {exc}")


@app.get("/biology")
def biology_interpret(metabolites: str):
    """
    Biological interpretation for a comma-separated list of metabolite names.

    Returns HMDB-curated per-metabolite roles/disease associations plus
    hypergeometric pathway over-representation analysis. Example:
        /biology?metabolites=lactate,alanine,glucose,pyruvate
    """
    names = [m.strip() for m in metabolites.split(",") if m.strip()]
    if not names:
        raise HTTPException(status_code=422, detail="No metabolite names supplied.")
    return biology.interpret_panel(names)


@app.get("/biomarkers-model-suite")
def biomarkers_model_suite(
    group_a: int | None = None,
    group_b: int | None = None,
    k: int = 8,
    repeats: int = 2,
    dataset: str = "mtbls242",
):
    """
    Nested patient-grouped comparison of elastic-net logistic regression,
    linear SVM, histogram gradient boosting and XGBoost (when installed).
    `dataset=mtbls1` runs the real diabetes-vs-control study.
    """
    try:
        if dataset == "mtbls1":
            if not MTBLS1_TSV.exists():
                raise HTTPException(status_code=404, detail="MTBLS1 dataset not bundled.")
            Xs, y, patients = _labeled_biomarker_task(
                MTBLS1_TSV.read_bytes(), _load_label_map()
            )
            task = DATASETS["mtbls1"]["task"]
            warning = "Cross-sectional diabetes-vs-control study (real disease labels)."
        else:
            if not DEFAULT_TSV.exists():
                raise HTTPException(status_code=404, detail="Default dataset not found.")
            Xs, y, patients, group_a, group_b = _biomarker_task(
                DEFAULT_TSV.read_bytes(), group_a, group_b
            )
            task = f"time-point {group_a} vs {group_b}"
            warning = "This dataset measures longitudinal surgery time points, not disease vs control."
        result = model_suite.compare_models(
            Xs.values,
            y,
            patients,
            k=k,
            repeats=repeats,
            feature_names=list(Xs.columns),
        )
        result["task"] = task
        result["dataset"] = dataset
        result["outcome_warning"] = warning
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Model comparison failed: {exc}")


@app.post("/biomarkers-model-suite-upload")
async def biomarkers_model_suite_upload(
    nmr_results_tsv: UploadFile = File(...),
    group_a: int | None = Form(default=None),
    group_b: int | None = Form(default=None),
    k: int = Form(default=8),
    repeats: int = Form(default=2),
):
    """Run the nested patient-grouped model suite on an uploaded results TSV."""
    raw = await nmr_results_tsv.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        Xs, y, patients, group_a, group_b = _biomarker_task(raw, group_a, group_b)
        result = model_suite.compare_models(
            Xs.values,
            y,
            patients,
            k=k,
            repeats=repeats,
            feature_names=list(Xs.columns),
        )
        result["task"] = f"time-point {group_a} vs {group_b}"
        return result
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Model comparison failed: {exc}")


@app.get("/biomarkers-projection")
def biomarkers_projection(
    group_a: int | None = None,
    group_b: int | None = None,
    include_umap: bool = True,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    dataset: str = "mtbls242",
):
    """Return exploratory PCA scores/loadings and optional UMAP coordinates."""
    try:
        if dataset == "mtbls1":
            if not MTBLS1_TSV.exists():
                raise HTTPException(status_code=404, detail="MTBLS1 dataset not bundled.")
            Xs, y, patients = _labeled_biomarker_task(
                MTBLS1_TSV.read_bytes(), _load_label_map()
            )
            names = DATASETS["mtbls1"]["class_names"]
            labels = [names[int(v)] for v in y]
            result = dimensionality.project(
                Xs.values, labels,
                sample_names=list(Xs.index), patient_ids=patients,
                feature_names=list(Xs.columns),
                include_umap=include_umap, n_neighbors=n_neighbors, min_dist=min_dist,
            )
            result["task"] = DATASETS["mtbls1"]["task"]
            result["dataset"] = dataset
            result["outcome_warning"] = "Real diabetes-vs-control labels; PCA/UMAP remain exploratory."
            return result
        if not DEFAULT_TSV.exists():
            raise HTTPException(status_code=404, detail="Default dataset not found.")
        return _dimensionality_task(
            DEFAULT_TSV.read_bytes(),
            group_a,
            group_b,
            include_umap=include_umap,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Projection failed: {exc}")


@app.post("/biomarkers-projection-upload")
async def biomarkers_projection_upload(
    nmr_results_tsv: UploadFile = File(...),
    group_a: int | None = Form(default=None),
    group_b: int | None = Form(default=None),
    include_umap: bool = Form(default=True),
    n_neighbors: int = Form(default=15),
    min_dist: float = Form(default=0.1),
):
    """Run exploratory PCA/UMAP on an uploaded MetaboLights results TSV."""
    raw = await nmr_results_tsv.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    try:
        return _dimensionality_task(
            raw,
            group_a,
            group_b,
            include_umap=include_umap,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
        )
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Projection failed: {exc}")


@app.get("/biomarkers-benchmark")
def biomarkers_benchmark(n_datasets: int = 100, p: int = 20000, n: int = 200,
                         n_true: int = 8, effect: float = 1.2, k: int = 20):
    """
    Benchmark the leakage-safe engine across `n_datasets` simulated p>>n cohorts
    (the brief's design). Returns the honest AUC distribution, leaky AUC, leakage
    inflation, Top-k stability, and true-biomarker recovery.
    """
    try:
        return biomarker_engine.benchmark(n_datasets=n_datasets, n=n, p=p,
                                          n_true=n_true, effect=effect, k=k)
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=422, detail=f"Benchmark failed: {exc}")


@app.get("/")
def index():
    """Serve the Chenomx-style Profiler web UI."""
    page = STATIC_DIR / "profiler.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="Frontend not built.")
    return FileResponse(page)
