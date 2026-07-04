"""
LiCO competition Track-1 data loader (serum + urine 1D ¹H-NMR).

Targets the EXACT cluster layout the organizers provided:

    nmr-pattern/Human_Serum/MTBLS6213_Metadata.csv
    nmr-pattern/Human_Serum/spectra_intensity_ppm.csv
    nmr-pattern/Human_Urine/MTBLS1_MTBLS694_metadata.csv
    nmr-pattern/Human_Urine/spectra_intensity_ppm.csv

The spectra file is (ppm rows × sample columns Var1..VarN); the metadata is a
SEPARATE file whose rows map to Var1..VarN BY COLUMN ORDER (Var_i = metadata row i),
NOT by id. This module handles every quirk of that schema so the data loads correctly
the first time on the H100/LiCO VM (where the real files cannot be read/exported, only
trained on) — and it is fully exercised off-VM by synthetic fixtures in
tests/test_lico_loader.py that reproduce the schema.

Handled quirks (from the dataset spec):
  * transposed ppm×Var matrix          → auto-oriented to samples×bins
  * positional Var↔metadata-row join   → aligned by order, fails loud on count mismatch
  * serum bracketed boolean labels      → Factor Value[Rheumatoid arthritis] / [Anti-TNF therapy]
  * urine DUPLICATE "Factor Value"      → col 1 = sex/QC, col 2 = condition (pandas → "Factor Value.1")
  * QC + dilution samples               → excluded before any labelling/discovery
  * serum vs urine matrix tag           → drives condition-aware handling downstream

GOVERNANCE: the real nmr-pattern/ data is closed and lives only inside the VM. The
parsers here read a LOCAL path and open NO network. Any TRAINING/EXPORT built on top
must stay on-VM (NMR_OFFLINE=1) and must not export the data — only trained weights.
"""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from . import spectral_cohort as sc
    from . import provenance as prov
except ImportError:  # pragma: no cover - direct execution
    import spectral_cohort as sc  # type: ignore
    import provenance as prov  # type: ignore


# ── QC / dilution detection ──────────────────────────────────────────────────
# QC codes and dilution-series samples are analytical controls, NOT disease
# case/control — they must be dropped before labelling or they corrupt the classes
# and inflate n. Matches the urine metadata codes in the spec + the MTBLS694 design
# (Study Pool, External/Long-Term Reference, DevSet, dilution factors).
_QC_RE = re.compile(
    r"(study\s*pool|external\s*reference|long[-\s]*term\s*reference|\bltr\b|\bsr\b|"
    r"dev\s*set|\bpool\b|\bqc\b|\bblank\b|dilution)", re.I)

_TRUE = {"true", "1", "1.0", "yes", "y", "t", "positive", "pos", "case"}
_FALSE = {"false", "0", "0.0", "no", "n", "f", "negative", "neg", "control"}


def is_qc_value(*values) -> bool:
    """True if any metadata value marks a QC / pool / reference / dilution sample."""
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if s and _QC_RE.search(s):
            return True
    return False


def _read_spectra(path: Path) -> Tuple[pd.DataFrame, np.ndarray, Dict]:
    """Load spectra_intensity_ppm.csv (ppm×Var) → (X samples×bins, bin_ppm, profile).
    Reuses the vetted orientation auto-detect (Var* header non-numeric, ppm index
    numeric → transpose) and matrix profiler."""
    return sc.load_binned_matrix_profiled(Path(path).read_bytes())


def _parse_metadata(raw: bytes) -> pd.DataFrame:
    """Parse a metadata CSV/TSV (bytes). dtype=str + keep_default_na=False so TRUE/FALSE
    stay literal text (not coerced to booleans), 'Sample Name' stays a string, and 'n/a'
    stays a value. Duplicate column names (urine 'Factor Value' ×2) → 'Factor Value' +
    'Factor Value.1' via pandas mangle_dupe_cols. Accepts CSV or TSV (incl. ISA-Tab)."""
    text = raw.decode("utf-8", errors="replace")
    sep = "\t" if text[:4096].count("\t") > text[:4096].count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, dtype=str, keep_default_na=False)


def _read_metadata(path: Path) -> pd.DataFrame:
    return _parse_metadata(Path(path).read_bytes())


def _align_positional(X: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Align metadata rows to spectra sample columns BY ORDER (Var_i = row i).
    Fails loud on a count mismatch — silent mis-alignment is the dominant risk."""
    n_s, n_m = X.shape[0], len(meta)
    if n_s != n_m:
        raise ValueError(
            f"Positional join needs equal counts: {n_s} spectra samples (Var columns) "
            f"vs {n_m} metadata rows. The schema maps Var_i → metadata row i by order, "
            f"so the counts MUST match. Check that neither file has an extra header/blank row.")
    meta = meta.reset_index(drop=True).copy()
    meta.index = list(X.index)     # Var1..VarN, in file order
    return meta


def _find_factor_col(meta: pd.DataFrame, needle: str) -> Optional[str]:
    """Find a bracketed factor column by the text inside the brackets, e.g.
    needle='rheumatoid arthritis' → 'Factor Value[Rheumatoid arthritis]'."""
    n = needle.strip().lower()
    for c in meta.columns:
        if n in str(c).lower():
            return c
    return None


def _factor_value_columns(meta: pd.DataFrame) -> List[str]:
    """The urine metadata has TWO columns literally named 'Factor Value' (pandas →
    'Factor Value' and 'Factor Value.1'). Return them in file order."""
    return [c for c in meta.columns
            if str(c) == "Factor Value" or re.fullmatch(r"Factor Value\.\d+", str(c))]


def _binarize(label_map: Dict[str, str], *, positive: Optional[str] = None
              ) -> Tuple[Dict[str, int], Dict]:
    """Turn a {sample: class-string} map into {sample: 0/1}. If `positive` is given
    (matched case-insensitively / as TRUE-set), that class is 1; otherwise the rarer
    class is positive (matches spectral_cohort.derive_labels' convention)."""
    vals = {k: str(v).strip() for k, v in label_map.items() if v is not None and str(v).strip()}
    classes = sorted(set(vals.values()))
    if len(classes) < 2:
        return {}, {"classes": classes, "note": "only one class present after filtering"}

    def _is_pos(v: str) -> bool:
        vl = v.lower()
        if positive is not None:
            pl = positive.lower()
            if pl in _TRUE or pl in _FALSE:               # boolean-style positive
                return vl in _TRUE
            return vl == pl or pl in vl
        return False

    if positive is not None:
        out = {k: (1 if _is_pos(v) else 0) for k, v in vals.items()}
        pos_name = positive
    else:
        counts = {c: sum(1 for v in vals.values() if v == c) for c in classes}
        pos_name = min(counts, key=counts.get)            # rarer class = positive
        out = {k: (1 if v == pos_name else 0) for k, v in vals.items()}
    info = {"classes": classes, "positive_class": pos_name,
            "class_balance": {c: sum(1 for v in vals.values() if v == c) for c in classes},
            "n_labeled": len(out)}
    return out, info


# ── serum ────────────────────────────────────────────────────────────────────
def _serum_dataset(X: pd.DataFrame, bin_ppm: np.ndarray, profile: Dict,
                   meta: pd.DataFrame) -> Dict:
    """Shared serum builder: positional-join metadata (loud on mismatch) → tasks
    `rheumatoid_arthritis` (main label) + `anti_tnf_therapy` (secondary)."""
    meta = _align_positional(X, meta)          # raises loud on Var/row count mismatch
    tasks: Dict[str, Dict] = {}
    for key, needle in (("rheumatoid_arthritis", "rheumatoid arthritis"),
                        ("anti_tnf_therapy", "anti-tnf")):
        col = _find_factor_col(meta, needle)
        if col is None:
            continue
        raw = {sid: meta.loc[sid, col] for sid in X.index}
        # boolean TRUE/FALSE → positive = TRUE (has RA / on therapy)
        lm, info = _binarize(raw, positive="true")
        tasks[key] = {"label_column": col, "label_map": lm, **info}
    return {
        "matrix": "serum", "X": X, "bin_ppm": bin_ppm, "profile": profile,
        "sample_ids": list(X.index), "n_samples": int(X.shape[0]),
        "tasks": tasks, "n_qc_excluded": 0,
        "metadata": meta,                       # on-VM only; never in safe metrics
        "metadata_columns": [str(c) for c in meta.columns],
    }


def load_serum(root, *, spectra="spectra_intensity_ppm.csv",
               metadata="MTBLS6213_Metadata.csv") -> Dict:
    """Load the serum cohort (MTBLS6213) from a directory. Returns X (samples×bins),
    bin_ppm, matrix profile, and the RA (main) + anti-TNF (secondary) tasks."""
    sd = Path(root) / "Human_Serum"
    X, bin_ppm, profile = _read_spectra(sd / spectra)
    return _serum_dataset(X, bin_ppm, profile, _read_metadata(sd / metadata))


def load_serum_bytes(spectra_raw: bytes, metadata_raw: bytes) -> Dict:
    """Load the serum cohort from uploaded FILE BYTES (the API path): transpose the
    ppm×Var spectra → samples×bins, positional-join the metadata (Var_i = row i),
    derive the RA main label + anti-TNF secondary. Raises ValueError loudly on a
    spectra/metadata count mismatch — never silently mis-joins."""
    X, bin_ppm, profile = sc.load_binned_matrix_profiled(spectra_raw)
    return _serum_dataset(X, bin_ppm, profile, _parse_metadata(metadata_raw))


# ── urine ────────────────────────────────────────────────────────────────────
def load_urine(root, *, spectra="spectra_intensity_ppm.csv",
               metadata="MTBLS1_MTBLS694_metadata.csv") -> Dict:
    """Load the urine cohort (MTBLS1+MTBLS694). Excludes QC/pool/reference/dilution
    samples, keeps sex as a covariate (first 'Factor Value'), and labels the
    condition (second 'Factor Value': Control vs diabetes) on the retained samples."""
    ud = Path(root) / "Human_Urine"
    X, bin_ppm, profile = _read_spectra(ud / spectra)
    meta = _align_positional(X, _read_metadata(ud / metadata))

    fv_cols = _factor_value_columns(meta)
    sex_col = fv_cols[0] if fv_cols else None
    cond_col = fv_cols[1] if len(fv_cols) > 1 else None

    # QC / dilution exclusion — check BOTH factor columns for control codes.
    qc_ids, keep_ids = [], []
    for sid in X.index:
        v_sex = meta.loc[sid, sex_col] if sex_col else None
        v_cond = meta.loc[sid, cond_col] if cond_col else None
        (qc_ids if is_qc_value(v_sex, v_cond) else keep_ids).append(sid)

    Xk = X.loc[keep_ids]
    sex = {sid: str(meta.loc[sid, sex_col]).strip() for sid in keep_ids} if sex_col else {}

    tasks: Dict[str, Dict] = {}
    if cond_col is not None:
        raw = {sid: meta.loc[sid, cond_col] for sid in keep_ids}
        # positive = disease; match 'diabetes' explicitly, else rarer class
        has_diab = any("diab" in str(v).lower() for v in raw.values())
        lm, info = _binarize(raw, positive=("diabetes mellitus" if has_diab else None))
        tasks["diabetes" if has_diab else "condition"] = {
            "label_column": cond_col, "label_map": lm, **info}

    return {
        "matrix": "urine", "X": Xk, "bin_ppm": bin_ppm,
        "profile": sc.profile_matrix(Xk, bin_ppm),
        "sample_ids": keep_ids, "n_samples": int(Xk.shape[0]),
        "sex": sex, "sex_column": sex_col, "condition_column": cond_col,
        "tasks": tasks,
        "n_qc_excluded": len(qc_ids), "qc_sample_ids": qc_ids,
    }


def load_lico(root="nmr-pattern") -> Dict:
    """Load both cohorts from the LiCO layout. Missing arms are skipped (returns only
    what's present) so it works on serum-only or urine-only roots too."""
    root = Path(root)
    out: Dict[str, Dict] = {}
    if (root / "Human_Serum").is_dir():
        out["serum"] = load_serum(root)
    if (root / "Human_Urine").is_dir():
        out["urine"] = load_urine(root)
    if not out:
        raise FileNotFoundError(
            f"No Human_Serum/ or Human_Urine/ under {root}. Expected the LiCO layout "
            f"nmr-pattern/Human_Serum/... and nmr-pattern/Human_Urine/...")
    return out


def build_supervised_set(dataset: Dict, task: str
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Turn a loaded cohort + task into (X_values, y, bin_ppm, feature_ppm) ready for
    training/discovery. One sample = one patient here, so no grouping is needed."""
    t = dataset["tasks"].get(task)
    if not t or not t.get("label_map"):
        raise ValueError(f"task '{task}' not available; have {list(dataset['tasks'])}")
    lm = t["label_map"]
    ids = [s for s in dataset["sample_ids"] if s in lm]
    X = dataset["X"].loc[ids]
    y = np.array([lm[s] for s in ids])
    return X.values, y, np.asarray(dataset["bin_ppm"], dtype=float), list(X.columns)


# ── Track-1 run with SAFE metrics only ───────────────────────────────────────
# On the VM the spectra + concentrations are closed, so this returns ONLY aggregate,
# non-identifying numbers (counts, fit R², confidence, statuses) — never raw spectra,
# per-sample values, or the concentration table itself.
def _safe_metrics(dataset: Dict, task: Optional[str], out: Dict, export: Dict) -> Dict:
    idc = out.get("identification", {}) or {}
    q = out.get("quantification", {}) or {}
    t1q = out.get("track1_quality", {}) or {}
    ann = out.get("annotation", {}) or {}
    tinfo = (dataset.get("tasks", {}) or {}).get(task, {}) if task else {}
    bm = out.get("biomarkers", {}) or {}
    return {
        "matrix": dataset.get("matrix"),
        "n_samples": dataset.get("n_samples"),
        "n_ppm_bins": ann.get("n_bins"),
        "ppm_range": ann.get("ppm_range"),
        "bin_width_est": (dataset.get("profile") or {}).get("bin_width_est"),
        "task": task,
        "label_counts": tinfo.get("class_balance"),
        "n_qc_excluded": dataset.get("n_qc_excluded", 0),
        # identification
        "identification_method": idc.get("method"),
        "n_fdr_confirmed": len(idc.get("deterministic_present", []) or []),
        "n_annotated_before_fdr": idc.get("annotate_present_count"),
        "mean_fit_r2": q.get("mean_fit_r2"),
        "reference_panel": out.get("reference_panel"),
        # pSCNN / hybrid
        "pscnn_status": idc.get("pscnn_status"),
        "hybrid_active": bool(idc.get("hybrid_present") is not None),
        "n_hybrid_present": len(idc.get("hybrid_present", []) or []) if idc.get("hybrid_present") is not None else None,
        "hybrid_added_by_pscnn": idc.get("hybrid_added_by_pscnn"),
        "pscnn_error": idc.get("pscnn_error"),
        # confidence
        "confidence": t1q.get("confidence"),
        "low_confidence_warning": t1q.get("warning"),
        # biomarkers (aggregate only)
        "biomarkers_present": "biomarkers" in out,
        "biomarker_honest_auc": bm.get("honest_roc_auc"),
        "biomarker_permutation_p": bm.get("permutation_p_value"),
        # export + provenance QC
        "concentration_csv_export": export,
        "matrix_warnings": (dataset.get("profile") or {}).get("warnings", []),
    }


def run_track1(dataset: Dict, task: Optional[str] = None, *, fdr: float = 0.05) -> Dict:
    """Run the FULL Track-1 pipeline (PQN → annotate → NNLS+target-decoy FDR → pSCNN
    hybrid → optional discovery) on a loaded cohort and return SAFE aggregate metrics
    only. Uses the app's own `_run_cohort_pipeline`, so the serum run is identical to a
    UI upload. Solvent is D₂O (serum in D₂O buffer) → D₂O guard active. Governance:
    surfaces no raw spectra/concentrations; safe to run on closed data on the VM."""
    from . import main as _main   # lazy: pulls the full pipeline incl. pSCNN hybrid

    X, bin_ppm, matrix = dataset["X"], dataset["bin_ppm"], dataset.get("matrix", "unknown")
    label_map = None
    if task:
        t = (dataset.get("tasks", {}) or {}).get(task, {})
        label_map = t.get("label_map") or None
    profile = dataset.get("profile") or sc.profile_matrix(X, bin_ppm)
    proven = prov.build_provenance(
        profile, {"solvent": "d2o", "sample_type": matrix, "field_mhz": "700"},
        normalization_applied="pqn")

    out = _main._run_cohort_pipeline(
        X, bin_ppm, label_map=label_map, normalize="pqn",
        include_biomarkers=bool(label_map), fdr=fdr, provenance=proven)

    try:
        csv = _main._concentration_csv(X, bin_ppm, provenance=proven)
        export = {"ok": True, "bytes": len(csv.encode("utf-8")), "n_header_lines": csv[:csv.find("\n\n") if "\n\n" in csv else 400].count("#")}
    except Exception as exc:   # never let export failure mask the run
        export = {"ok": False, "error": str(exc)}

    return _safe_metrics(dataset, task, out, export)


# ── training-signal check ────────────────────────────────────────────────────
_ANNOT_HINT = re.compile(
    r"(metabolite|compound|annotation|assignment|identif|hmdb|chebi|inchi|smiles|"
    r"database_identifier|chemical_formula|\bppm\b.*assign)", re.I)


def training_check(dataset: Dict) -> Dict:
    """Report what can be trained on this cohort. Scans the preserved metadata for any
    COMPOUND-ANNOTATION columns (needed to fine-tune the pSCNN compound identifier) vs
    only PHENOTYPE labels (RA / anti-TNF). Honest: says what to train now, not now, later."""
    cols = dataset.get("metadata_columns", [])
    annot_cols = [c for c in cols if _ANNOT_HINT.search(str(c))]
    pheno = {k: v.get("class_balance") for k, v in (dataset.get("tasks", {}) or {}).items()
             if v.get("label_map")}
    has_annot = bool(annot_cols)
    can_now, do_not, wait = [], [], []

    # SSL pre-training on the spectra is always available (unlabeled).
    can_now.append("SSL masked-spectrum pre-training on the serum spectra (unlabeled) — on-VM.")
    if pheno:
        can_now.append(f"Supervised phenotype model(s) on labels {list(pheno)} — on-VM, "
                       f"leakage-safe CV. NOTE small n → report wide CIs, expect overfit risk.")
    if has_annot:
        can_now.append(f"pSCNN compound-ID fine-tune from annotation column(s) {annot_cols} — "
                       f"on-VM via finetune_loader (NMR_OFFLINE=1).")
    else:
        do_not.append("Do NOT fine-tune the pSCNN compound identifier from this file — the serum "
                      "metadata has ONLY phenotype labels (RA / anti-TNF), no per-compound "
                      "annotation columns. Fabricating a training panel would be dishonest.")
        wait.append("pSCNN compound-ID fine-tune: WAIT until an organizer compound-annotation "
                    "file/columns actually appear, then run finetune_loader on the VM.")

    do_not.append("Do NOT train on the closed data OFF the VM, and do NOT export the data — only "
                  "trained weights may leave (open-data governance).")
    wait.append("Any large/deep model: WAIT for the full dataset size (this is a 143-record subset) "
                "to avoid overfitting.")

    return {
        "has_compound_annotations": has_annot,
        "compound_annotation_columns": annot_cols,
        "phenotype_labels": pheno,
        "can_train_now": can_now,
        "do_not_train": do_not,
        "should_wait": wait,
        "governance": "on-VM only (NMR_OFFLINE=1); local read/write; no data export; weights-only out.",
    }


def ingestion_report(root="nmr-pattern") -> Dict:
    """A read-only sanity report: what loaded, class balance, QC excluded, bin width,
    matrix. Run this FIRST on the VM to confirm the data ingests before any training."""
    data = load_lico(root)
    rep: Dict[str, Dict] = {}
    for name, ds in data.items():
        p = ds["profile"]
        rep[name] = {
            "matrix": ds["matrix"], "n_samples": ds["n_samples"],
            "n_bins": p.get("n_bins"), "ppm_range": p.get("ppm_range"),
            "bin_width_est": p.get("bin_width_est"), "coarse_bins": p.get("coarse_bins"),
            "n_qc_excluded": ds.get("n_qc_excluded", 0),
            "tasks": {k: {"classes": v.get("classes"),
                          "balance": v.get("class_balance"),
                          "n_labeled": v.get("n_labeled")} for k, v in ds["tasks"].items()},
            "matrix_warnings": p.get("warnings", []),
        }
    return rep


if __name__ == "__main__":   # on-VM: python -m nmr_api.lico_loader [root]
    import json
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else "nmr-pattern"
    print(json.dumps(ingestion_report(root), indent=2, ensure_ascii=False))
