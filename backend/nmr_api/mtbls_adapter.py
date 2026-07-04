"""
Adapter: PUBLIC MetaboLights MTBLS6213 (ISA-Tab + Bruker) → the competition serum
schema, for an EXTERNAL real-serum sanity test of the Human-Serum loader + Track-1.

WHY THIS EXISTS — the public study does NOT ship the competition's `spectra_intensity_ppm.csv`.
Public MTBLS6213 provides:
  * s_MTBLS6213.txt              — ISA-Tab sample sheet (has Factor Value[Rheumatoid arthritis],
                                    [Anti-TNF therapy], [Treatment response])
  * a_MTBLS6213_*.txt            — assay (Sample Name → FILES/<n>.zip raw Bruker)
  * m_MTBLS6213_*_maf.tsv        — Chenomx-quantified metabolite table (Track-2-like)
  * FILES/<n>.zip                — per-sample raw Bruker (pdata/1/1r processed spectrum)
  * FILES/TNF_MTBLS6213_normalised.csv — a NAMED-bucket feature matrix (not ppm-indexed)
The competition's ppm×Var binned matrix is an organizer-DERIVED artifact. This adapter
reconstructs an equivalent ppm×Var matrix from the raw Bruker `1r` spectra + the sample
sheet, so the SAME loader/pipeline can be exercised on real serum data.

GOVERNANCE — MTBLS6213 is PUBLIC. This adapter is a public external sanity check only.
It must NOT be pointed at closed competition data, and it prints/returns no closed rows.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# competition metadata target columns
_META_TARGETS = [
    ("Characteristics[Organism part]", "organism part"),
    ("Sample Name", "sample name"),
    ("Factor Value[Rheumatoid arthritis]", "rheumatoid arthritis"),
    ("Factor Value[Anti-TNF therapy]", "anti-tnf"),
]


def isatab_to_serum_metadata(raw) -> pd.DataFrame:
    """ISA-Tab sample sheet (bytes or path) → the competition MTBLS6213_Metadata.csv
    columns, dropping ontology 'plumbing' (Term Source REF / Term Accession / Unit).
    The loader also tolerates raw ISA-Tab, but this yields the exact competition schema."""
    if isinstance(raw, (str, Path)):
        raw = Path(raw).read_bytes()
    text = raw.decode("utf-8", errors="replace")
    # dtype=str + keep_default_na=False: keep TRUE/FALSE as literal text (pandas would
    # otherwise coerce them to booleans), keep 'Sample Name' as strings, and keep 'n/a'
    # as a value rather than NaN — faithful to the source metadata.
    df = pd.read_csv(io.StringIO(text), sep="\t", dtype=str, keep_default_na=False)
    keep: Dict[str, pd.Series] = {}
    for target, needle in _META_TARGETS:
        col = next((c for c in df.columns if needle in str(c).lower()), None)
        if col is not None:
            keep[target] = df[col].astype(str).str.strip()
    if "Sample Name" not in keep:
        raise ValueError("ISA-Tab sample sheet has no 'Sample Name' column.")
    return pd.DataFrame(keep)


def bruker_zip_to_spectrum(zip_path) -> Tuple[np.ndarray, np.ndarray]:
    """Extract a per-sample Bruker zip and read the processed real spectrum (pdata/1/1r)
    → (ppm ascending, intensity). Requires nmrglue."""
    import tempfile
    import nmrglue as ng
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(tmp)
        pdirs = list(Path(tmp).glob("*/pdata/1")) or list(Path(tmp).glob("**/pdata/1"))
        if not pdirs:
            raise FileNotFoundError(f"no pdata/1 in {zip_path}")
        d, data = ng.bruker.read_pdata(str(pdirs[0]))
        uc = ng.fileiobase.uc_from_udic(ng.bruker.guess_udic(d, data))
        ppm = np.asarray(uc.ppm_scale(), dtype=float)
        data = np.asarray(data, dtype=float)
        order = np.argsort(ppm)                    # ascending ppm
        return ppm[order], data[order]


def bin_to_grid(ppm: np.ndarray, intensity: np.ndarray,
                lo: float = 0.5, hi: float = 9.5, width: float = 0.005
                ) -> Tuple[np.ndarray, np.ndarray]:
    """Bucket a native spectrum onto a common ppm grid by summing intensity per bucket
    (standard NMR bucketing) → (bin centers, binned intensity)."""
    n = int(round((hi - lo) / width))
    edges = np.linspace(lo, hi, n + 1)
    idx = np.digitize(ppm, edges) - 1
    m = (idx >= 0) & (idx < n)
    out = np.zeros(n, dtype=float)
    np.add.at(out, idx[m], intensity[m])
    centers = np.round((edges[:-1] + edges[1:]) / 2.0, 5)
    return centers, out


def build_competition_serum(zip_map: Sequence[Tuple[str, str]], samplesheet,
                            out_root, *, lo=0.5, hi=9.5, width=0.005) -> Path:
    """Reconstruct the competition serum layout from public MTBLS6213.

    zip_map: [(sample_name, zip_path), ...] in the desired Var order.
    Writes  <out_root>/Human_Serum/spectra_intensity_ppm.csv  (ppm × Var1..VarN)
            <out_root>/Human_Serum/MTBLS6213_Metadata.csv      (rows in the SAME order
                                                                → positional Var↔row join).
    Returns the Human_Serum dir. Uses PUBLIC data only."""
    meta_full = isatab_to_serum_metadata(samplesheet)
    by_name = meta_full.set_index(meta_full["Sample Name"].astype(str))

    centers: Optional[np.ndarray] = None
    cols: Dict[str, np.ndarray] = {}
    ordered: List[str] = []
    for i, (name, zp) in enumerate(zip_map):
        ppm, inten = bruker_zip_to_spectrum(zp)
        c, b = bin_to_grid(ppm, inten, lo=lo, hi=hi, width=width)
        centers = c
        cols[f"Var{i + 1}"] = b
        ordered.append(str(name))

    spectra = pd.DataFrame(cols)
    spectra.insert(0, "ppm", centers)
    meta_ordered = by_name.loc[ordered].reset_index(drop=True)

    out = Path(out_root) / "Human_Serum"
    out.mkdir(parents=True, exist_ok=True)
    spectra.to_csv(out / "spectra_intensity_ppm.csv", index=False)
    meta_ordered.to_csv(out / "MTBLS6213_Metadata.csv", index=False)
    return out
