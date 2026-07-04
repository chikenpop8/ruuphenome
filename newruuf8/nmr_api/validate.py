"""
Validation harness — measure how well RuuPhenome matches real NMR ground truth.

Run any of four checks:

  1. peaks    — picked peaks (Domain 1) vs a reference peak list (ppm matching)
  2. ids      — metabolite identifications vs a reference table (precision/recall/F1)
  3. quant    — concentrations vs a reference table (Pearson/Spearman + Bland-Altman)
  4. shifts   — predicted ¹H shifts vs experimental HMDB/literature shifts (MAE)

The "reference" is normally a Chenomx export (CSV) on the SAME spectra, an HMDB
shift table, or known spiked-standard concentrations.

Usage
-----
  # Identification + quantification vs a Chenomx export
  python -m nmr_api.validate ids   --pred ruuphenome_results.csv --ref chenomx_export.csv
  python -m nmr_api.validate quant --pred ruuphenome_results.csv --ref chenomx_export.csv

  # Domain-1 peak picking vs a reference peak list (one ppm value per line)
  python -m nmr_api.validate peaks --pred picked_peaks.csv --ref reference_peaks.csv --tol 0.02

  # Shift prediction vs experimental shifts
  python -m nmr_api.validate shifts --pred predicted_shifts.csv --ref hmdb_shifts.csv

Each command prints a metrics report and exits non-zero if accuracy is below the
pass threshold, so it can gate a CI pipeline.
"""

from __future__ import annotations

import argparse
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ── helpers ────────────────────────────────────────────────────────────────
def _norm(name: str) -> str:
    """Normalize a metabolite name for matching (lowercase, strip L-/D-, spaces)."""
    s = str(name).strip().lower()
    for p in ("l-", "d-", "(r)-", "(s)-", "dl-"):
        if s.startswith(p):
            s = s[len(p):]
    return s.replace(" ", "").replace("-", "").replace("_", "")


def _load(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ── 1. Peak picking accuracy ───────────────────────────────────────────────
def check_peaks(pred_csv: str, ref_csv: str, tol: float = 0.02) -> Dict:
    """Match picked peaks to reference peaks within `tol` ppm."""
    pred = _read_ppm_column(pred_csv)
    ref = _read_ppm_column(ref_csv)
    matched, used = 0, set()
    errors = []
    for r in ref:
        best, bi = tol + 1, -1
        for i, p in enumerate(pred):
            if i in used:
                continue
            d = abs(p - r)
            if d < best:
                best, bi = d, i
        if bi >= 0 and best <= tol:
            matched += 1; used.add(bi); errors.append(best)
    recall = matched / len(ref) if ref else 0
    precision = matched / len(pred) if pred else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        "reference_peaks": len(ref),
        "predicted_peaks": len(pred),
        "matched": matched,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "mean_ppm_error": round(float(np.mean(errors)), 4) if errors else None,
        "max_ppm_error": round(float(np.max(errors)), 4) if errors else None,
        "pass": f1 >= 0.8,
    }


def _read_ppm_column(path: str) -> List[float]:
    df = pd.read_csv(path)
    for col in ("ppm", "shift", "chemical_shift", "position"):
        if col in df.columns:
            return sorted(pd.to_numeric(df[col], errors="coerce").dropna().tolist())
    # fall back to first numeric column
    num = df.select_dtypes("number")
    return sorted(num.iloc[:, 0].dropna().tolist()) if num.shape[1] else []


# ── 2. Identification accuracy ─────────────────────────────────────────────
def check_ids(pred_csv: str, ref_csv: str) -> Dict:
    """Precision / recall / F1 of identified metabolites vs reference."""
    pred = _id_set(pred_csv)
    ref = _id_set(ref_csv)
    tp = pred & ref
    precision = len(tp) / len(pred) if pred else 0
    recall = len(tp) / len(ref) if ref else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return {
        "predicted_ids": len(pred),
        "reference_ids": len(ref),
        "true_positives": len(tp),
        "false_positives": sorted(pred - ref)[:20],
        "false_negatives": sorted(ref - pred)[:20],
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "pass": f1 >= 0.8,
    }


def _id_set(path: str) -> set:
    df = pd.read_csv(path)
    col = _first_col(df, ["metabolite", "compound", "name", "metabolite_identification",
                          "Compound Name"])
    # only count rows that were actually identified (concentration present if a column exists)
    conc = _first_col(df, ["concentration_uM", "concentration", "Concentration (µM)"], required=False)
    if conc:
        df = df[pd.to_numeric(df[conc], errors="coerce").notna()]
    return {_norm(x) for x in df[col].dropna()}


# ── 3. Quantification accuracy ─────────────────────────────────────────────
def check_quant(pred_csv: str, ref_csv: str) -> Dict:
    """Correlate predicted vs reference concentrations on shared metabolites."""
    p = _conc_map(pred_csv); r = _conc_map(ref_csv)
    shared = sorted(set(p) & set(r))
    if len(shared) < 3:
        return {"error": "fewer than 3 shared metabolites with concentrations", "shared": shared}
    pv = np.array([p[k] for k in shared]); rv = np.array([r[k] for k in shared])
    from scipy.stats import pearsonr, spearmanr
    pear = pearsonr(pv, rv); spear = spearmanr(pv, rv)
    diff = pv - rv
    return {
        "shared_metabolites": len(shared),
        "pearson_r": round(float(pear[0]), 3),
        "pearson_p": round(float(pear[1]), 4),
        "spearman_r": round(float(spear[0]), 3),
        "bland_altman_bias": round(float(np.mean(diff)), 3),
        "bland_altman_sd": round(float(np.std(diff)), 3),
        "mean_abs_pct_error": round(float(np.mean(np.abs(diff) / (rv + 1e-9)) * 100), 1),
        "pass": float(pear[0]) >= 0.9,
    }


def _conc_map(path: str) -> Dict[str, float]:
    df = pd.read_csv(path)
    name = _first_col(df, ["metabolite", "compound", "name", "metabolite_identification",
                           "Compound Name"])
    conc = _first_col(df, ["concentration_uM", "concentration", "Concentration (µM)"])
    out = {}
    for _, row in df.iterrows():
        v = pd.to_numeric(row[conc], errors="coerce")
        if pd.notna(v):
            out[_norm(row[name])] = float(v)
    return out


# ── 4. Shift-prediction accuracy ───────────────────────────────────────────
def check_shifts(pred_csv: str, ref_csv: str, tol: float = 0.05) -> Dict:
    """Mean absolute error of predicted ¹H shifts vs experimental, per metabolite."""
    p = _shift_map(pred_csv); r = _shift_map(ref_csv)
    shared = sorted(set(p) & set(r))
    all_err, per = [], []
    for k in shared:
        pred_sh, ref_sh = sorted(p[k]), sorted(r[k])
        errs = []
        for rs in ref_sh:
            if pred_sh:
                errs.append(min(abs(rs - ps) for ps in pred_sh))
        if errs:
            per.append({"metabolite": k, "mae_ppm": round(float(np.mean(errs)), 4)})
            all_err += errs
    return {
        "shared_metabolites": len(shared),
        "overall_mae_ppm": round(float(np.mean(all_err)), 4) if all_err else None,
        "within_tol_pct": round(float(np.mean([e <= tol for e in all_err]) * 100), 1) if all_err else None,
        "per_metabolite": sorted(per, key=lambda x: x["mae_ppm"], reverse=True)[:15],
        "pass": bool(all_err) and float(np.mean(all_err)) <= tol,
    }


def _shift_map(path: str) -> Dict[str, List[float]]:
    df = pd.read_csv(path)
    name = _first_col(df, ["metabolite", "compound", "name", "metabolite_identification"])
    shifts = _first_col(df, ["predicted_1H_shifts_ppm", "shifts", "chemical_shift", "ppm"])
    out: Dict[str, List[float]] = {}
    for _, row in df.iterrows():
        raw = str(row[shifts])
        vals = [float(x) for x in raw.replace(",", " ").split() if _isfloat(x)]
        if vals:
            out.setdefault(_norm(row[name]), []).extend(vals)
    return out


# ── shared utils ───────────────────────────────────────────────────────────
def _first_col(df: pd.DataFrame, names: List[str], required: bool = True):
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise SystemExit(f"None of the expected columns {names} found in: {list(df.columns)}")
    return None


def _isfloat(x: str) -> bool:
    try:
        float(x); return True
    except ValueError:
        return False


def _report(title: str, metrics: Dict) -> int:
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"  {k:24s}: {v}")
    ok = metrics.get("pass")
    print(f"\n  RESULT: {'PASS ✅' if ok else 'FAIL / review ⚠️'}")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="RuuPhenome validation harness")
    ap.add_argument("check", choices=["peaks", "ids", "quant", "shifts"])
    ap.add_argument("--pred", required=True, help="RuuPhenome output CSV")
    ap.add_argument("--ref", required=True, help="Ground-truth CSV (Chenomx export / HMDB / standards)")
    ap.add_argument("--tol", type=float, default=0.02, help="ppm tolerance (peaks/shifts)")
    args = ap.parse_args()

    if args.check == "peaks":
        sys.exit(_report("Peak picking (Domain 1)", check_peaks(args.pred, args.ref, args.tol)))
    elif args.check == "ids":
        sys.exit(_report("Identification accuracy", check_ids(args.pred, args.ref)))
    elif args.check == "quant":
        sys.exit(_report("Quantification accuracy", check_quant(args.pred, args.ref)))
    elif args.check == "shifts":
        sys.exit(_report("Shift-prediction accuracy", check_shifts(args.pred, args.ref, args.tol)))
