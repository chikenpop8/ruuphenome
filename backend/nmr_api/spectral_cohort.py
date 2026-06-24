"""
Track 1 — multi-sample binned-spectra pipeline.

The hackathon organizers deliver *preprocessed, binned* NMR data (raw FID → FT →
phase/baseline → binning → cut regions → optional align/normalize is done for
us). This module consumes that binned data and performs the scored Track-1 work:

    load binned matrix  →  (optional) normalize  →  ANNOTATE bins → metabolites
                        →  sample × metabolite table  →  visualization

Annotation method: targeted profiling by reference-shift matching — the standard
open-source approach (the transparent core of Chenomx / ASICS / rDolphin). A
metabolite is called present when enough of its characteristic ¹H shifts fall in
occupied bins; its abundance is the intensity carried by those bins. This is
explainable (every call shows which shifts matched) and needs no training.

The output sample × metabolite table feeds straight into the Track-2 biomarker
engine — one automated pipeline across both tracks.
"""

from __future__ import annotations

import io
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# ── reference ¹H shift fingerprints (ppm), HMDB 5.0 — name-keyed ─────────────
# Characteristic singlet/multiplet centers used for targeted profiling.
REFERENCE_SHIFTS: Dict[str, List[float]] = {
    "alanine": [1.48, 3.78],
    "valine": [0.99, 1.04, 2.28, 3.62],
    "leucine": [0.96, 1.71, 3.73],
    "isoleucine": [0.94, 1.01, 1.26, 1.46, 1.98, 3.67],
    "lactate": [1.33, 4.11],
    "glucose": [3.24, 3.41, 3.47, 3.53, 3.71, 3.83, 4.65, 5.23],
    "citrate": [2.54, 2.66],
    "creatinine": [3.05, 4.06],
    "creatine": [3.04, 3.93],
    "glutamine": [2.14, 2.45, 3.78],
    "glutamate": [2.05, 2.12, 2.35, 3.76],
    "glycine": [3.56],
    "histidine": [3.16, 3.24, 7.05, 7.79],
    "tyrosine": [3.05, 3.20, 6.90, 7.19],
    "phenylalanine": [3.13, 3.28, 7.32, 7.37, 7.42],
    "tryptophan": [3.30, 3.49, 7.20, 7.28, 7.54, 7.73],
    "pyruvate": [2.38],
    "acetate": [1.92],
    "acetoacetate": [2.28, 3.45],
    "3-hydroxybutyrate": [1.20, 2.31, 2.41, 4.16],
    "2-hydroxybutyrate": [0.90, 1.69, 1.71, 4.00],
    "2-oxoisovalerate": [1.13, 3.02],
    "threonine": [1.32, 3.58, 4.25],
    "proline": [2.00, 2.07, 2.34, 3.34, 3.42, 4.14],
    "methionine": [2.13, 2.64, 3.86],
    "lysine": [1.47, 1.73, 1.91, 3.03, 3.76],
    "arginine": [1.69, 1.92, 3.24, 3.77],
    "choline": [3.20, 3.52, 4.07],
    "betaine": [3.27, 3.90],
    "taurine": [3.27, 3.43],
    "trimethylamine n-oxide": [3.27],
    "dimethylamine": [2.72],
    "methylamine": [2.61],
    "formate": [8.46],
    "ethanol": [1.19, 3.66],
    "methanol": [3.36],
    "myo-inositol": [3.28, 3.53, 3.62, 4.07],
    "succinate": [2.41],
    "hypoxanthine": [8.19, 8.21],
    "isopropanol": [1.17, 4.02],
    # ── expanded human serum/urine ¹H NMR panel (HMDB/literature reference
    #    shifts) → ~100-metabolite real-world-scale annotation library ──
    # amino acids & derivatives
    "asparagine": [2.85, 2.96, 4.00],
    "aspartate": [2.68, 2.80, 3.89],
    "serine": [3.84, 3.95, 3.98],
    "cysteine": [3.00, 3.12, 3.97],
    "ornithine": [1.81, 1.94, 3.05, 3.78],
    "citrulline": [1.57, 1.86, 3.14, 3.78],
    "hydroxyproline": [2.07, 2.43, 3.42, 4.30],
    "sarcosine": [2.74, 3.61],
    "beta-alanine": [2.55, 3.18],
    "gaba": [1.91, 2.30, 3.01],
    "4-aminobutyrate": [1.91, 2.30, 3.01],
    "1-methylhistidine": [3.15, 3.23, 7.07, 7.78],
    "3-methylhistidine": [3.13, 3.22, 7.06, 7.77],
    "carnosine": [2.66, 3.02, 3.20, 7.10, 8.00],
    "guanidoacetate": [3.80],
    "creatine phosphate": [3.04, 3.94],
    # organic acids
    "fumarate": [6.52],
    "malate": [2.37, 2.67, 4.30],
    "2-oxoglutarate": [2.44, 3.00],
    "alpha-ketoglutarate": [2.44, 3.00],
    "propionate": [1.05, 2.18],
    "butyrate": [0.89, 1.56, 2.16],
    "isobutyrate": [1.13, 2.38],
    "3-hydroxyisovalerate": [1.27, 2.36],
    "2-hydroxyisobutyrate": [1.36],
    "methylmalonate": [1.24, 3.14],
    "malonate": [3.11],
    "glycolate": [3.93],
    "acetone": [2.23],
    "hippurate": [3.97, 7.55, 7.64, 7.84],
    "benzoate": [7.48, 7.55, 7.87],
    "4-hydroxyphenylacetate": [3.45, 6.86, 7.16],
    "phenylacetate": [3.53, 7.30, 7.37],
    "allantoin": [5.39],
    # sugars & polyols
    "fructose": [3.55, 3.80, 3.99, 4.11],
    "galactose": [3.48, 3.70, 3.92, 4.58, 5.27],
    "mannose": [3.38, 3.55, 3.78, 3.88, 5.18],
    "lactose": [3.50, 3.60, 4.45, 5.23],
    "scyllo-inositol": [3.34],
    "mannitol": [3.68, 3.78, 3.87],
    "1,5-anhydroglucitol": [3.25, 3.50, 3.70, 3.85],
    # choline / lipid-related
    "phosphocholine": [3.22, 3.60, 4.18],
    "glycerophosphocholine": [3.23, 3.62, 4.32],
    "acetylcarnitine": [2.14, 3.19, 3.60],
    "o-acetylcarnitine": [2.14, 3.19, 3.60],
    "carnitine": [2.44, 3.23, 3.42, 4.56],
    # amines
    "ethanolamine": [3.13, 3.82],
    "putrescine": [1.77, 3.05],
    "methylguanidine": [2.83],
    # nucleobases / nucleosides / NAD-related
    "xanthine": [7.90],
    "uracil": [5.80, 7.53],
    "uridine": [5.90, 5.92, 7.87],
    "inosine": [6.10, 8.23, 8.34],
    "nicotinamide": [8.25, 8.71, 8.94, 9.28],
    "1-methylnicotinamide": [4.48, 8.19, 8.90, 8.96, 9.28],
    "trigonelline": [4.44, 8.08, 8.84, 9.13],
    # gut-microbial / xenobiotic
    "phenylacetylglutamine": [1.99, 2.27, 3.68, 7.36, 7.42],
    "propylene glycol": [1.13, 3.42, 3.53, 3.87],
    "p-cresol": [2.25, 6.85, 7.15],
    "4-cresol sulfate": [2.34, 7.20, 7.28],
    "indoxyl sulfate": [7.20, 7.28, 7.36, 7.51, 7.70],
    "dimethyl sulfone": [3.14],
    "trimethylamine": [2.88],
    "2-oxoisocaproate": [0.94, 2.10, 2.62],
    "2-oxo-3-methylvalerate": [0.90, 1.10, 1.70, 2.95],
}


def _normalize_name(name: str) -> str:
    return name.strip().lower()


# ── binned-matrix loading ─────────────────────────────────────────────────────
def _looks_numeric(values: Sequence[str]) -> float:
    """Fraction of header tokens that parse as floats (i.e. look like ppm bins)."""
    ok = 0
    for v in values:
        try:
            float(str(v).replace("ppm", "").strip())
            ok += 1
        except ValueError:
            pass
    return ok / max(1, len(values))


def load_binned_matrix(raw: bytes) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load a binned NMR matrix as samples × ppm-bins, auto-detecting orientation.

    Accepts CSV or TSV. Two common layouts are handled:
      (a) rows = samples, columns = ppm-bin centers  (header row is numeric)
      (b) rows = ppm-bin centers, columns = samples   (first column is numeric)

    Returns (X [samples × bins], bin_ppm [bin centers, ascending columns]).
    """
    text = raw.decode("utf-8", errors="replace")
    sep = "\t" if text[:2048].count("\t") >= text[:2048].count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, index_col=0, low_memory=False)

    # Decide orientation from whether the column headers are numeric ppm values.
    header_numeric = _looks_numeric(df.columns.astype(str))
    index_numeric = _looks_numeric(df.index.astype(str))

    if header_numeric < 0.6 and index_numeric >= 0.6:
        df = df.T  # layout (b): bins were rows → transpose so samples are rows

    # Coerce bin labels to float ppm; drop any non-numeric columns.
    ppm = []
    keep = []
    for col in df.columns:
        try:
            ppm.append(float(str(col).replace("ppm", "").strip()))
            keep.append(col)
        except ValueError:
            continue
    if not keep:
        raise ValueError("No numeric ppm-bin columns found in the matrix.")

    X = df[keep].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    bin_ppm = np.asarray(ppm, dtype=float)
    order = np.argsort(bin_ppm)
    X = X.iloc[:, order]
    X.columns = [round(float(p), 5) for p in bin_ppm[order]]
    return X, bin_ppm[order]


# ── normalization (safety net; organizers may already normalize) ─────────────
def total_area_normalize(X: pd.DataFrame) -> pd.DataFrame:
    """Scale each sample so its total integrated area is constant."""
    areas = X.abs().sum(axis=1).replace(0, np.nan)
    return X.div(areas, axis=0).fillna(0.0) * float(np.nanmedian(areas))


def pqn_normalize(X: pd.DataFrame) -> pd.DataFrame:
    """
    Probabilistic Quotient Normalization (Dieterle 2006) — the standard NMR
    metabolomics method robust to dilution. Total-area normalize, build a median
    reference spectrum, then divide each sample by the median of its quotients.
    """
    Xta = total_area_normalize(X)
    reference = Xta.median(axis=0).replace(0, np.nan)
    quotients = Xta.div(reference, axis=1)
    factors = quotients.replace([np.inf, -np.inf], np.nan).median(axis=1)
    factors = factors.replace(0, np.nan).fillna(1.0)
    return Xta.div(factors, axis=0)


# ── annotation: bins → metabolites ───────────────────────────────────────────
def annotate(
    X: pd.DataFrame,
    bin_ppm: np.ndarray,
    *,
    reference_shifts: Optional[Dict[str, List[float]]] = None,
    identified_peaks: Optional[Dict[float, str]] = None,
    tol_ppm: float = 0.03,
    occupancy_quantile: float = 0.75,
    min_coverage: float = 0.5,
) -> Dict:
    """
    Targeted profiling: map occupied ppm bins to metabolites by reference-shift
    matching, producing a sample × metabolite abundance table.

    Args:
        X: samples × bins matrix.
        bin_ppm: ppm center of each column.
        reference_shifts: metabolite → characteristic ¹H shifts (defaults to
            the bundled HMDB fingerprint library).
        identified_peaks: optional organizer-provided {ppm: metabolite} pins,
            used directly and merged with reference matching.
        tol_ppm: how close a bin must be to a reference shift to count.
        occupancy_quantile: a bin is "occupied" if its cohort-median intensity
            exceeds this quantile of all median bin intensities.
        min_coverage: fraction of a metabolite's shifts that must be observed
            to call it present.

    Returns annotation report + sample × metabolite abundance matrix.
    """
    refs = {**(reference_shifts or REFERENCE_SHIFTS)}
    # Fold organizer-identified peaks into the reference library.
    if identified_peaks:
        for ppm, name in identified_peaks.items():
            refs.setdefault(_normalize_name(name), []).append(float(ppm))

    bins = np.asarray(X.columns, dtype=float)
    median_profile = X.median(axis=0).values
    occ_threshold = np.quantile(median_profile, occupancy_quantile)

    def nearest_bin(shift: float) -> Optional[int]:
        j = int(np.argmin(np.abs(bins - shift)))
        return j if abs(bins[j] - shift) <= tol_ppm else None

    metabolites = []
    abundance_cols: Dict[str, np.ndarray] = {}
    for name, shifts in refs.items():
        matched_idx, matched_shifts = [], []
        for s in shifts:
            j = nearest_bin(s)
            if j is not None and median_profile[j] >= occ_threshold:
                matched_idx.append(j)
                matched_shifts.append(round(float(bins[j]), 4))
        coverage = len(matched_idx) / max(1, len(shifts))
        present = coverage >= min_coverage and len(matched_idx) > 0
        if not present:
            continue
        cols = X.iloc[:, sorted(set(matched_idx))]
        abundance = cols.mean(axis=1).values            # per-sample abundance
        abundance_cols[name] = abundance
        metabolites.append({
            "metabolite": name,
            "coverage": round(coverage, 3),
            "expected_shifts": len(shifts),
            "matched_shifts": matched_shifts,
            "matched_count": len(matched_idx),
            "confidence": round(min(100.0, coverage * 100.0), 1),
            "mean_abundance": round(float(np.mean(abundance)), 5),
        })

    metabolites.sort(key=lambda m: (-m["coverage"], -m["matched_count"]))
    annotated = pd.DataFrame(abundance_cols, index=X.index) if abundance_cols \
        else pd.DataFrame(index=X.index)

    return {
        "n_samples": int(X.shape[0]),
        "n_bins": int(X.shape[1]),
        "ppm_range": [round(float(bins.min()), 3), round(float(bins.max()), 3)],
        "occupancy_threshold": round(float(occ_threshold), 6),
        "n_metabolites_annotated": len(metabolites),
        "metabolites": metabolites,
        "annotated_matrix": annotated,        # samples × metabolite (for Track 2)
        "reference_library_size": len(refs),
    }


# ── multi-sample visualization helpers ───────────────────────────────────────
def overlay_traces(X: pd.DataFrame, max_samples: int = 8, points: int = 400) -> Dict:
    """Downsampled overlay traces for a stacked multi-sample spectrum view."""
    bins = np.asarray(X.columns, dtype=float)
    if len(bins) > points:
        idx = np.linspace(0, len(bins) - 1, points).astype(int)
    else:
        idx = np.arange(len(bins))
    ppm = [round(float(bins[i]), 4) for i in idx]
    traces = []
    for name in list(X.index)[:max_samples]:
        row = X.loc[name].values[idx]
        traces.append({"sample": str(name), "intensity": [round(float(v), 5) for v in row]})
    return {"ppm": ppm, "traces": traces, "n_total_samples": int(X.shape[0])}


# ── organizer-provided identified peaks (Track 1) ────────────────────────────
def parse_identified_peaks(raw: bytes) -> Dict[float, str]:
    """
    Parse an organizer-provided identified-peaks file into {ppm: metabolite}.

    Accepts CSV or TSV with a ppm/shift column and a metabolite/name column
    (header names are matched loosely; falls back to first two columns).
    """
    text = raw.decode("utf-8", errors="replace")
    sep = "\t" if text[:2048].count("\t") >= text[:2048].count(",") else ","
    df = pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)
    if df.shape[1] < 2:
        return {}

    def find(colnames, *keys):
        for c in colnames:
            k = str(c).strip().lower()
            if any(key in k for key in keys):
                return c
        return None

    ppm_col = find(df.columns, "ppm", "shift", "bin", "region", "position")
    name_col = find(df.columns, "metabolite", "compound", "name", "assignment", "annotation")
    if ppm_col is None or name_col is None:
        ppm_col, name_col = df.columns[0], df.columns[1]

    out: Dict[float, str] = {}
    for _, row in df.iterrows():
        try:
            ppm = float(str(row[ppm_col]).replace("ppm", "").strip())
        except (ValueError, TypeError):
            continue
        name = str(row[name_col]).strip()
        if name and name.lower() not in ("nan", "none", ""):
            out[ppm] = name
    return out


# ── Track 2: metadata join (Table 1 metabolites + Table 2 metadata) ──────────
def parse_metadata(raw: bytes) -> pd.DataFrame:
    """Load a metadata table (Table 2). Rows = samples, columns = phenotypes."""
    text = raw.decode("utf-8", errors="replace")
    sep = "\t" if text[:2048].count("\t") >= text[:2048].count(",") else ","
    return pd.read_csv(io.StringIO(text), sep=sep, low_memory=False)


def _guess_sample_column(meta: pd.DataFrame, sample_ids: Sequence[str]) -> Optional[str]:
    """Pick the metadata column whose values best overlap the sample IDs."""
    ids = {str(s) for s in sample_ids}
    best, best_overlap = None, 0
    for col in meta.columns:
        overlap = sum(str(v) in ids for v in meta[col])
        if overlap > best_overlap:
            best, best_overlap = col, overlap
    return best if best_overlap >= max(2, 0.3 * len(ids)) else None


def derive_labels(
    meta: pd.DataFrame,
    sample_ids: Sequence[str],
    *,
    label_column: Optional[str] = None,
    sample_column: Optional[str] = None,
    positive_class: Optional[str] = None,
) -> Tuple[Dict[str, int], Dict]:
    """
    Join metadata to samples and derive a binary label map.

    Auto-detects the sample-id column and (if not given) the most informative
    binary-izable phenotype column. Returns ({sample: 0/1}, info).
    """
    sample_column = sample_column or _guess_sample_column(meta, sample_ids)
    if sample_column is None:
        raise ValueError("Could not match a metadata column to the sample IDs.")
    m = meta.set_index(meta[sample_column].astype(str))

    # choose label column: explicit, else the categorical column with 2 classes
    candidates = [c for c in m.columns if c != sample_column]
    if label_column is None:
        scored = []
        for c in candidates:
            vals = m[c].dropna().astype(str)
            nun = vals.nunique()
            if 2 <= nun <= 6:
                scored.append((c, nun))
        if not scored:
            raise ValueError("No categorical phenotype column with 2–6 classes found.")
        scored.sort(key=lambda x: x[1])   # prefer fewest classes (cleanest task)
        label_column = scored[0][0]

    series = m[label_column].astype(str)
    counts = series.value_counts()
    if positive_class is None:
        # two largest classes define the binary task; rarer = positive
        top2 = list(counts.index[:2])
        if len(top2) < 2:
            raise ValueError(f"Column '{label_column}' has <2 usable classes.")
        positive_class = top2[1] if counts[top2[1]] <= counts[top2[0]] else top2[0]
    classes = list(counts.index[:2])
    label_map = {}
    for sid in sample_ids:
        v = series.get(str(sid))
        if v in classes:
            label_map[str(sid)] = 1 if v == positive_class else 0
    info = {
        "sample_column": sample_column,
        "label_column": label_column,
        "positive_class": positive_class,
        "classes": classes,
        "n_labeled": len(label_map),
        "class_balance": {k: int(counts[k]) for k in classes},
    }
    return label_map, info


# ── demo binned cohort (self-contained, no upload needed) ────────────────────
def make_demo_binned(
    n_per_group: int = 30, bin_width: float = 0.01, seed: int = 7
) -> Tuple[pd.DataFrame, np.ndarray, Dict[str, int]]:
    """
    Simulate a two-group binned ¹H cohort from a realistic serum subset of the
    reference fingerprints, with a real, plant-able difference (BCAAs ↑ in the
    'case' group). For demoing the full Track-1 → Track-2 pipeline without
    external files. Uses a fixed core panel (not the whole library) so the
    spectrum density stays realistic regardless of library size.
    """
    rng = np.random.default_rng(seed)
    bins = np.round(np.arange(0.5, 9.0, bin_width), 5)
    names = [f"ctrl_{i:03d}" for i in range(n_per_group)] + \
            [f"case_{i:03d}" for i in range(n_per_group)]
    rows = []
    labels = {}
    elevated = {"valine", "leucine", "isoleucine", "2-oxoisovalerate"}
    # fixed realistic serum panel (~25 metabolites) — keeps demo density stable
    core = ["alanine", "valine", "leucine", "isoleucine", "lactate", "glucose",
            "citrate", "creatinine", "glutamine", "glycine", "histidine",
            "tyrosine", "phenylalanine", "pyruvate", "acetate", "3-hydroxybutyrate",
            "2-oxoisovalerate", "threonine", "methionine", "choline", "taurine",
            "succinate", "formate", "myo-inositol", "betaine"]
    core_refs = {m: REFERENCE_SHIFTS[m] for m in core if m in REFERENCE_SHIFTS}
    for s in names:
        is_case = s.startswith("case")
        labels[s] = 1 if is_case else 0
        spec = rng.normal(0.0, 0.002, len(bins)).clip(min=0)
        for met, shifts in core_refs.items():
            base = rng.uniform(0.2, 1.0)
            if met in elevated and is_case:
                base *= rng.uniform(1.6, 2.4)        # plant the diabetes-like signal
            for sh in shifts:
                j = int(np.argmin(np.abs(bins - sh)))
                width = max(1, int(round(0.012 / bin_width)))
                for k in range(-width, width + 1):
                    if 0 <= j + k < len(bins):
                        spec[j + k] += base * np.exp(-(k * k) / (2 * (width / 2 + 0.5) ** 2))
        spec *= rng.uniform(0.85, 1.15)              # per-sample dilution
        rows.append(spec)
    X = pd.DataFrame(rows, index=names, columns=[round(float(b), 5) for b in bins])
    return X, bins, labels
