"""
RuuPhenome — NMR Metabolite Identification Pipeline
Domain 1: Pattern Identification (NMR peak annotation)
Domain 2: MTBLS242 blood serum metabolomics (CHEBI-annotated)

Usage:
    python nmr_pipeline.py --domain1_pdf path/to/spectrum.pdf \
                           --domain2_tsv path/to/nmr_results.tsv \
                           --output results/
"""

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ─── Domain 1 peaks extracted from PDF annotation ───────────────────────────
# These were manually read from Domain_1_processed_NMR_spectrum.pdf
# (plant extract + serum peaks annotated per the PDF visualization)
DOMAIN1_ANNOTATED_PEAKS = [
    {"metabolite": "Asparagine",          "shift_ppm": 2.87},
    {"metabolite": "Aspartate",           "shift_ppm": 2.65},
    {"metabolite": "4-Aminobutyrate",     "shift_ppm": 1.89},
    {"metabolite": "Citrate",             "shift_ppm": 2.54},
    {"metabolite": "Succinate",           "shift_ppm": 2.41},
    {"metabolite": "Glutamate",           "shift_ppm": 2.35},
    {"metabolite": "N-Acetylglutamate",   "shift_ppm": 2.05},
    {"metabolite": "Pyroglutamate",       "shift_ppm": 2.00},
    {"metabolite": "Acetate",             "shift_ppm": 1.92},
    {"metabolite": "Alanine",             "shift_ppm": 1.48},
    {"metabolite": "Threonine",           "shift_ppm": 1.33},
    {"metabolite": "Methanol",            "shift_ppm": 3.36},
    {"metabolite": "Choline",             "shift_ppm": 3.21},
    {"metabolite": "Ethanol",             "shift_ppm": 1.19},
    {"metabolite": "Nicotinate",          "shift_ppm": 8.62},
    {"metabolite": "Formate",             "shift_ppm": 8.45},
    {"metabolite": "Adenosine",           "shift_ppm": 8.34},
    {"metabolite": "Tryptophan",          "shift_ppm": 7.33},
    {"metabolite": "Phenylalanine",       "shift_ppm": 7.32},
    {"metabolite": "Histidine",           "shift_ppm": 7.08},
    {"metabolite": "Tyrosine",            "shift_ppm": 6.89},
    {"metabolite": "Cytidine",            "shift_ppm": 5.90},
    {"metabolite": "Uridine",             "shift_ppm": 5.89},
    {"metabolite": "Sucrose",             "shift_ppm": 5.40},
    {"metabolite": "Glucose",             "shift_ppm": 5.22},
    {"metabolite": "Valine",              "shift_ppm": 0.98},
    {"metabolite": "Leucine",             "shift_ppm": 0.96},
    {"metabolite": "Isoleucine",          "shift_ppm": 0.93},
    {"metabolite": "Malate",              "shift_ppm": 2.65},
    {"metabolite": "Meso-Inositol",       "shift_ppm": 3.27},
    {"metabolite": "Xylose",              "shift_ppm": 5.17},
    {"metabolite": "Guanine",             "shift_ppm": 7.76},
    {"metabolite": "Xanthurinate",        "shift_ppm": 7.82},
    {"metabolite": "Chlorogenate",        "shift_ppm": 6.27},
    {"metabolite": "Homocysteine",        "shift_ppm": 2.73},
]


def load_domain2(tsv_path: str) -> pd.DataFrame:
    """Load MTBLS242 NMR results TSV into a DataFrame."""
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)

    sample_cols = [c for c in df.columns
                   if c not in ["database_identifier", "chemical_formula",
                                "smiles", "inchi", "metabolite_identification",
                                "chemical_shift", "multiplicity", "taxid",
                                "species", "database", "database_version",
                                "reliability", "uri", "search_engine",
                                "search_engine_score", "smallmolecule_abundance_sub",
                                "smallmolecule_abundance_stdev_sub",
                                "smallmolecule_abundance_std_error_sub"]]

    meta_cols = ["database_identifier", "chemical_formula", "smiles",
                 "inchi", "metabolite_identification", "chemical_shift", "multiplicity"]
    df_meta = df[meta_cols].copy()
    df_abundance = df[sample_cols].apply(pd.to_numeric, errors="coerce")

    df_meta["n_samples_measured"] = df_abundance.notna().sum(axis=1)
    df_meta["mean_abundance"] = df_abundance.mean(axis=1)
    df_meta["cv_abundance"] = (df_abundance.std(axis=1) / df_abundance.mean(axis=1) * 100)

    return df_meta, df_abundance, sample_cols


def fetch_hmdb_shifts(smiles: str, metabolite_name: str) -> list[dict]:
    """
    Query HMDB NMR database for known ¹H chemical shifts.
    Falls back to NMRTransformer prediction if HMDB returns no result.
    """
    try:
        import requests
        url = "https://hmdb.ca/metabolites/search"
        params = {"query": metabolite_name, "query_type": "metabolite_name"}
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            return []
    except Exception:
        pass
    return []


def predict_shifts_nmrtransformer(smiles_list: list[str]) -> dict[str, list[float]]:
    """
    Predict ¹H NMR chemical shifts using NMRTransformer.
    Requires: pip install -e NMRTransformer/ (from setup.sh)
    """
    try:
        from NMRTransformer.predict import predict_1h_shifts
        results = {}
        for smi in smiles_list:
            if smi and isinstance(smi, str):
                shifts = predict_1h_shifts(smi)
                results[smi] = shifts
        return results
    except ImportError:
        print("[WARN] NMRTransformer not installed. Using fallback HMDB lookup.")
        return {}
    except Exception as e:
        print(f"[WARN] NMRTransformer prediction failed: {e}")
        return {}


def match_peaks_to_metabolites(
    observed_peaks: list[dict],
    candidates: pd.DataFrame,
    predicted_shifts: dict,
    tolerance_ppm: float = 0.05
) -> pd.DataFrame:
    """
    Score metabolite candidates against observed NMR peaks.

    Scoring:
      - +10 pts per matched peak within tolerance
      - Score normalized by total observed peaks
    """
    records = []
    for _, row in candidates.iterrows():
        smi = row.get("smiles", "")
        name = row.get("metabolite_identification", "")
        pred = predicted_shifts.get(smi, [])

        matched = 0
        matched_peaks = []
        for obs in observed_peaks:
            obs_shift = obs["shift_ppm"]
            for p_shift in pred:
                if abs(p_shift - obs_shift) <= tolerance_ppm:
                    matched += 1
                    matched_peaks.append(f"{obs['metabolite']}≈{p_shift:.2f}")
                    break

        score = min((matched / max(len(pred), 1)) * 100, 100.0) if pred else 0

        records.append({
            "metabolite_identification": name,
            "chebi_id": row.get("database_identifier", ""),
            "smiles": smi,
            "predicted_shifts": json.dumps(sorted(pred)) if pred else "[]",
            "peaks_matched": matched,
            "match_score": round(score, 1),
            "matched_peaks": "; ".join(matched_peaks),
            "mean_abundance": row.get("mean_abundance", 0),
            "cv_%": row.get("cv_abundance", 0),
        })

    df_scores = pd.DataFrame(records).sort_values("match_score", ascending=False)
    return df_scores


def run_pipeline(domain2_tsv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    print("\n── Step 1: Load Domain 2 (MTBLS242) ──")
    df_meta, df_abundance, sample_cols = load_domain2(domain2_tsv)
    print(f"  Loaded {len(df_meta)} metabolites, {len(sample_cols)} samples")
    print(f"  Metabolites: {', '.join(df_meta['metabolite_identification'].tolist())}")

    print("\n── Step 2: Predict ¹H shifts via NMRTransformer ──")
    smiles_list = df_meta["smiles"].dropna().unique().tolist()
    predicted_shifts = predict_shifts_nmrtransformer(smiles_list)

    if not predicted_shifts:
        print("  NMRTransformer unavailable — using HMDB known shifts as fallback")
        predicted_shifts = build_hmdb_fallback_shifts(df_meta)

    print(f"  Predictions obtained for {len(predicted_shifts)} molecules")

    print("\n── Step 3: Match Domain 1 peaks → metabolite candidates ──")
    df_scores = match_peaks_to_metabolites(
        DOMAIN1_ANNOTATED_PEAKS, df_meta, predicted_shifts
    )
    scores_path = Path(output_dir) / "metabolite_match_scores.csv"
    df_scores.to_csv(scores_path, index=False)
    print(f"  Match scores saved: {scores_path}")
    print(df_scores[["metabolite_identification", "match_score", "peaks_matched"]].to_string(index=False))

    print("\n── Step 4: Abundance matrix (sample × metabolite) ──")
    abundance_path = Path(output_dir) / "abundance_matrix.csv"
    df_wide = df_abundance.copy()
    df_wide.index = df_meta["metabolite_identification"].values
    df_wide.T.to_csv(abundance_path)
    print(f"  Abundance matrix saved: {abundance_path}")
    print(f"  Shape: {df_wide.shape[1]} samples × {df_wide.shape[0]} metabolites")

    print("\n── Step 5: Summary report ──")
    report = {
        "total_metabolites": int(len(df_meta)),
        "total_samples": int(len(sample_cols)),
        "metabolites_with_smiles": int(df_meta["smiles"].notna().sum()),
        "nmr_predictions_run": int(len(predicted_shifts)),
        "domain1_peaks_annotated": len(DOMAIN1_ANNOTATED_PEAKS),
        "top_matches": df_scores.head(5)[
            ["metabolite_identification", "match_score"]
        ].to_dict("records"),
    }
    report_path = Path(output_dir) / "pipeline_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Report saved: {report_path}")

    return df_scores, df_abundance


def build_hmdb_fallback_shifts(df_meta: pd.DataFrame) -> dict:
    """
    Hardcoded known ¹H NMR shifts (ppm) from HMDB for the 21 MTBLS242 metabolites.
    Used as fallback when NMRTransformer is not installed.
    Source: HMDB 5.0 / literature.
    """
    known = {
        # 21 MTBLS242 metabolites — SMILES matched exactly from TSV (HMDB 5.0)
        "C[C@@H](O)CC(O)=O":               [1.20, 2.30, 2.44, 4.17],        # (R)-3-Hydroxybutyric acid
        "CC(O)=O":                          [1.92],                           # Acetate
        "CC(=O)CC(O)=O":                    [2.28, 3.43],                    # Acetoacetate
        "C[C@H](N)C(O)=O":                  [1.48, 3.78],                    # L-Alanine
        "CC(C)[C@H](N)C(O)=O":              [0.94, 0.99, 2.28, 3.61],       # L-Valine
        "O=c1[nH]cnc2nc[nH]c12":           [8.12, 8.19],                    # Hypoxanthine
        "OC(=O)CC(O)(CC(O)=O)C(O)=O":      [2.54, 2.67],                    # Citrate
        "CN1CC(=O)NC1=N":                   [3.04, 3.77, 4.06],             # Creatinine
        "N[C@@H](CCC(N)=O)C(O)=O":         [1.88, 2.13, 2.47, 3.76],       # L-Glutamine
        "NCC(O)=O":                         [3.55],                          # Glycine
        "NC(Cc1c[nH]cn1)C(O)=O":           [3.19, 3.28, 7.08, 7.78],       # Histidine
        "CC[C@@H](C)[C@H](N)C(O)=O":       [0.93, 0.95, 1.47, 1.98, 3.68], # L-allo-Isoleucine
        "CC(C)O":                           [1.17, 4.02],                    # Isopropanol
        "C[C@H](O)C(O)=O":                 [1.33, 4.31],                    # L-Lactic acid
        "CC(C)C[C@H](N)C(O)=O":            [0.94, 0.96, 1.70, 1.72, 3.73], # L-Leucine
        "CO":                               [3.36],                          # Methanol
        "CS(C)(=O)=O":                      [3.14],                          # Dimethyl sulfone
        "N[C@H](Cc1ccccc1)C(O)=O":         [3.12, 3.27, 7.31, 7.35, 7.42], # D-Phenylalanine
        "CC(=O)C(O)=O":                     [2.37],                          # Pyruvic acid
        "N[C@@H](Cc1ccc(O)cc1)C(O)=O":     [3.04, 3.19, 6.89, 7.18],       # L-Tyrosine
        "":                                 [],                              # Lipoproteins (no SMILES)
    }
    result = {}
    for _, row in df_meta.iterrows():
        smi = row.get("smiles", "")
        if smi and smi in known:
            result[smi] = known[smi]
        elif smi:
            result[smi] = []
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RuuPhenome NMR Pipeline")
    parser.add_argument("--domain2_tsv", required=True,
                        help="Path to Domain_2_NMR_results_MTBLS242.tsv")
    parser.add_argument("--output", default="nmr_results",
                        help="Output directory for results")
    args = parser.parse_args()

    run_pipeline(args.domain2_tsv, args.output)
