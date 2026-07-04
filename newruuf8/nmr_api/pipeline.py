"""
Core NMR metabolite-recognition engine.

Domain 1 — observed/annotated peaks from a processed spectrum.
Domain 2 — MetaboLights study table (e.g. MTBLS242) with SMILES + abundances.

The engine predicts ¹H shifts for every Domain 2 candidate (via NMRTransformer
or the HMDB fallback), then scores each candidate against the Domain 1 peak list
and computes per-metabolite abundance statistics for downstream biomarker work.
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .shifts_db import predict_shifts


# Columns in a MetaboLights NMR results table that are NOT sample-abundance columns.
NON_SAMPLE_COLS = {
    "database_identifier", "chemical_formula", "smiles", "inchi",
    "metabolite_identification", "chemical_shift", "multiplicity",
    "taxid", "species", "database", "database_version", "reliability",
    "uri", "search_engine", "search_engine_score",
    "smallmolecule_abundance_sub", "smallmolecule_abundance_stdev_sub",
    "smallmolecule_abundance_std_error_sub",
}

META_COLS = [
    "database_identifier", "chemical_formula", "smiles",
    "inchi", "metabolite_identification", "chemical_shift", "multiplicity",
]

# Aliases for the metabolite-name column (organizers' tables vary).
_NAME_ALIASES = {
    "metabolite_identification", "metabolite_identification_",
    "metabolite", "metabolite_name", "metabolitename",
    "name", "compound", "compound_name", "identification",
}

# Extra missing-value tokens common in real concentration exports. Added ON TOP OF
# pandas' defaults so a blank / "n.d." / "<LOD" cell becomes NaN instead of a
# spurious 0 or a text value that silently coerces to NaN with no trace. Kept to
# tokens that only ever mean "no measurement" in a NUMERIC column — so this list
# is applied to Table 1 (concentrations), not to the categorical metadata table.
NUMERIC_MISSING_TOKENS = [
    "n.d.", "N.D.", "nd", "ND", "n/d", "N/D",
    "<LOD", "<lod", "<LLOQ", "<lloq", "BLQ", "BDL", "b.d.l.",
    "-", "--", "—", "not detected", "Not Detected", "NOT DETECTED",
]


def sniff_separator(head: str) -> str:
    """Most likely delimiter among tab, semicolon, comma (handles EU-style CSVs)."""
    counts = {"\t": head.count("\t"), ";": head.count(";"), ",": head.count(",")}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def coerce_numeric(frame: "pd.DataFrame") -> "pd.DataFrame":
    """to_numeric that also rescues European decimals ('12,34' -> 12.34).

    A comma is only treated as a decimal point when the WHOLE cell is a single
    decimal number ('^\\d+,\\d+$'). Thousands-separated ('1,234') or list-like
    ('a,b') values are left untouched and simply coerce to NaN as before, so this
    can never corrupt data that was already clean.
    """
    def _fix(col):
        # Any non-numeric column may hold "12,34" text (pandas may type it as
        # object OR the newer 'str' dtype — check "not numeric", not "== object").
        if not pd.api.types.is_numeric_dtype(col):
            s = col.astype(str).str.strip()
            mask = s.str.match(r"^\d+,\d+$")
            if mask.any():
                col = s.where(~mask, s.str.replace(",", ".", regex=False))
        return pd.to_numeric(col, errors="coerce")
    return frame.apply(_fix)


def read_results_table(raw: bytes) -> pd.DataFrame:
    """
    Robustly read a Domain-2 results table (Table 1).

    Handles CSV *or* TSV (auto-sniffed) and normalizes the metabolite-name
    column header (e.g. "metabolite identification", "Metabolite", "compound")
    to the canonical ``metabolite_identification``. Sample columns are left as-is.
    """
    text = raw.decode("utf-8", errors="replace")
    sep = sniff_separator(text[:8192])
    df = pd.read_csv(io.StringIO(text), sep=sep, low_memory=False,
                     na_values=NUMERIC_MISSING_TOKENS, keep_default_na=True)

    rename: Dict[str, str] = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_").replace("-", "_")
        if key in _NAME_ALIASES:
            rename[col] = "metabolite_identification"
        elif key in NON_SAMPLE_COLS:
            rename[col] = key
    df = df.rename(columns=rename)

    # Fallback: if no name column matched, use the first non-numeric column.
    if "metabolite_identification" not in df.columns:
        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                df = df.rename(columns={col: "metabolite_identification"})
                break
    return df


# ── Domain 1: default annotated peak list (from the processed-spectrum PDF) ──
DEFAULT_DOMAIN1_PEAKS: List[Dict] = [
    {"metabolite": "Asparagine",      "observed_shift": 2.87, "region": "aliphatic"},
    {"metabolite": "Aspartate",       "observed_shift": 2.65, "region": "aliphatic"},
    {"metabolite": "4-Aminobutyrate", "observed_shift": 1.89, "region": "aliphatic"},
    {"metabolite": "Citrate",         "observed_shift": 2.54, "region": "aliphatic"},
    {"metabolite": "Succinate",       "observed_shift": 2.41, "region": "aliphatic"},
    {"metabolite": "Glutamate",       "observed_shift": 2.35, "region": "aliphatic"},
    {"metabolite": "Acetate",         "observed_shift": 1.92, "region": "aliphatic"},
    {"metabolite": "Alanine",         "observed_shift": 1.48, "region": "aliphatic"},
    {"metabolite": "Threonine",       "observed_shift": 1.33, "region": "aliphatic"},
    {"metabolite": "Valine",          "observed_shift": 0.98, "region": "aliphatic"},
    {"metabolite": "Leucine",         "observed_shift": 0.96, "region": "aliphatic"},
    {"metabolite": "Isoleucine",      "observed_shift": 0.93, "region": "aliphatic"},
    {"metabolite": "Choline",         "observed_shift": 3.21, "region": "oxygenated"},
    {"metabolite": "Meso-Inositol",   "observed_shift": 3.27, "region": "oxygenated"},
    {"metabolite": "Methanol",        "observed_shift": 3.36, "region": "oxygenated"},
    {"metabolite": "Sucrose",         "observed_shift": 5.40, "region": "anomeric"},
    {"metabolite": "Glucose",         "observed_shift": 5.22, "region": "anomeric"},
    {"metabolite": "Xylose",          "observed_shift": 5.17, "region": "anomeric"},
    {"metabolite": "Cytidine",        "observed_shift": 5.90, "region": "anomeric"},
    {"metabolite": "Uridine",         "observed_shift": 5.89, "region": "anomeric"},
    {"metabolite": "Tyrosine",        "observed_shift": 6.89, "region": "aromatic"},
    {"metabolite": "Histidine",       "observed_shift": 7.08, "region": "aromatic"},
    {"metabolite": "Tryptophan",      "observed_shift": 7.33, "region": "aromatic"},
    {"metabolite": "Phenylalanine",   "observed_shift": 7.32, "region": "aromatic"},
    {"metabolite": "Guanine",         "observed_shift": 7.76, "region": "aromatic"},
    {"metabolite": "Xanthurinate",    "observed_shift": 7.82, "region": "aromatic"},
    {"metabolite": "Nicotinate",      "observed_shift": 8.62, "region": "aromatic"},
    {"metabolite": "Formate",         "observed_shift": 8.45, "region": "aromatic"},
    {"metabolite": "Adenosine",       "observed_shift": 8.34, "region": "aromatic"},
    {"metabolite": "Malate",          "observed_shift": 2.65, "region": "aliphatic"},
]


def load_domain2(tsv_bytes: bytes) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Parse a MetaboLights NMR results TSV (raw bytes) into:
      df_meta      — metabolite metadata (one row per metabolite)
      df_abundance — numeric sample-abundance matrix
      sample_cols  — list of sample-column names
    """
    df = read_results_table(tsv_bytes)

    sample_cols = [c for c in df.columns if c not in NON_SAMPLE_COLS]
    df_meta = df[[c for c in META_COLS if c in df.columns]].copy()
    df_abundance = df[sample_cols].apply(pd.to_numeric, errors="coerce")

    df_meta["mean_abundance"] = df_abundance.mean(axis=1)
    df_meta["stdev_abundance"] = df_abundance.std(axis=1)
    df_meta["cv_percent"] = (
        df_meta["stdev_abundance"] / df_meta["mean_abundance"] * 100
    ).abs()
    df_meta["detected_percent"] = (
        df_abundance.notna().sum(axis=1) / max(len(sample_cols), 1) * 100
    )

    return df_meta, df_abundance, sample_cols


def score_candidates(
    df_meta: pd.DataFrame,
    predicted_shifts: Dict[str, List[float]],
    observed_peaks: List[Dict],
    tolerance_ppm: float = 0.05,
) -> pd.DataFrame:
    """
    Score each Domain 2 candidate against the observed peak list.

    match_score = (# predicted shifts that land within tolerance of an
                   observed peak) / (# predicted shifts) * 100, capped at 100.
    """
    records = []
    for _, row in df_meta.iterrows():
        smi = row.get("smiles", "") if pd.notna(row.get("smiles", "")) else ""
        pred = predicted_shifts.get(smi, [])
        name = row.get("metabolite_identification", "")

        matched, detail = 0, []
        for obs in observed_peaks:
            obs_shift = obs["observed_shift"]
            for p in pred:
                if abs(p - obs_shift) <= tolerance_ppm:
                    matched += 1
                    detail.append(f"{obs['metabolite']}≈{p:.2f}ppm")
                    break

        score = min((matched / max(len(pred), 1)) * 100, 100.0) if pred else 0.0

        records.append({
            "metabolite": _safe_str(name),
            "chebi_id": _safe_str(row.get("database_identifier", "")),
            "chemical_formula": _safe_str(row.get("chemical_formula", "")),
            "smiles": smi,
            "predicted_shifts": pred,
            "n_shifts_predicted": len(pred),
            "peaks_matched": matched,
            "match_score": round(score, 1),
            "matched_detail": "; ".join(detail),
            "mean_abundance": _safe_float(row.get("mean_abundance")),
            "cv_percent": _safe_float(row.get("cv_percent")),
            "detected_percent": _safe_float(row.get("detected_percent")),
        })

    return pd.DataFrame(records).sort_values("match_score", ascending=False)


def analyze(
    tsv_bytes: bytes,
    observed_peaks: Optional[List[Dict]] = None,
    tolerance_ppm: float = 0.05,
) -> Dict:
    """
    Full pipeline: parse Domain 2 → predict shifts → score against Domain 1.
    Returns a JSON-serializable dict matching models.AnalysisResponse.
    """
    peaks = observed_peaks or DEFAULT_DOMAIN1_PEAKS
    df_meta, _df_abund, sample_cols = load_domain2(tsv_bytes)

    smiles_list = df_meta["smiles"].fillna("").tolist()
    predicted, backend = predict_shifts(smiles_list)

    df_scores = score_candidates(df_meta, predicted, peaks, tolerance_ppm)

    return {
        "summary": {
            "total_metabolites": int(len(df_meta)),
            "total_samples": int(len(sample_cols)),
            "metabolites_with_smiles": int(df_meta["smiles"].fillna("").astype(bool).sum()),
            "prediction_backend": backend,
            "tolerance_ppm": tolerance_ppm,
        },
        "matches": df_scores.to_dict(orient="records"),
    }


def _safe_str(value) -> str:
    """Coerce to a clean string, mapping NaN/None → empty string."""
    if value is None:
        return ""
    try:
        if isinstance(value, float) and np.isnan(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _safe_float(value) -> Optional[float]:
    """Convert to float, mapping NaN/inf/None → None for clean JSON."""
    try:
        f = float(value)
        if np.isnan(f) or np.isinf(f):
            return None
        return round(f, 4)
    except (TypeError, ValueError):
        return None
