"""
External cross-cohort validation — reproducibility / RUO dev tool.

**NOT part of the served app.** This standalone script downloads PUBLIC
MetaboLights ¹H-NMR studies (open data) and runs the exact leakage-safe Track-2
engine (`biomarker_engine.discover`) on cohorts the project has never bundled or
trained on, reporting HONEST metrics — ROC-AUC with a bootstrap 95% CI, a
permutation p-value, the leaky-vs-honest gap, and a whole-matrix differential
table. It is network-using **by design** (external validation must fetch the
external cohort); the served FastAPI app stays offline (`NMR_OFFLINE=1`).

Data governance: only OPEN MetaboLights data; the closed competition dataset is
never used here. Results are research-use-only (RUO), not clinical validation.

Usage (from `backend/`):
    python -m nmr_api.external_validation                # run all cohorts
    python -m nmr_api.external_validation --cohort MTBLS161_serum
    python -m nmr_api.external_validation --list

Each cohort records its accession, FTP URL, label column, biofluid filter and
positive class, so the numbers reproduce from a clean checkout + internet.
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from . import biomarker_engine, biomarkers, differential, spectral_cohort
except ImportError:  # direct execution fallback
    import biomarker_engine, biomarkers, differential, spectral_cohort  # type: ignore

FTP = "https://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public"
CACHE = Path(tempfile.gettempdir()) / "ruuphenome_extval"

# ── external validation cohorts (public MetaboLights ¹H NMR; NOT bundled) ─────
# Add a cohort = one entry here. `matrix` filters on Characteristics[Organism part].
COHORTS: List[Dict] = [
    {
        "id": "MTBLS161_serum",
        "accession": "MTBLS161",
        "disease": "ME/CFS vs healthy control",
        "biofluid": "serum",
        "matrix": "blood serum",
        "label_column": "Factor Value[Chronic Fatigue Syndrome]",
    },
    {
        "id": "MTBLS161_urine",
        "accession": "MTBLS161",
        "disease": "ME/CFS vs healthy control",
        "biofluid": "urine",
        "matrix": "urine",
        "label_column": "Factor Value[Chronic Fatigue Syndrome]",
    },
    {
        # Metabolomics Workbench, 1D ¹H NMR in D₂O @ 600 MHz — the closest genuine
        # per-sample NMR *metabolic* cohort found (no open binary T2D-vs-control NMR
        # set exists; see docs/IMPACT_AND_VALIDATION.md §7). 3-group, exercises the
        # multi-class path on real external NMR data. Not disease-vs-control.
        "id": "ST004325_urine",
        "accession": "ST004325",
        "source": "workbench",
        "disease": "Type-1 diabetes by disease duration (T1D-S/M/L, 3-group)",
        "biofluid": "urine",
        "matrix": None,
        "factor_key": "Duration_group",
        "multiclass": True,
    },
]

_MW = "https://www.metabolomicsworkbench.org/rest/study/study_id"

_ANN = {
    "database_identifier", "chemical_formula", "smiles", "inchi",
    "metabolite_identification", "chemical_shift", "multiplicity", "taxid",
    "species", "database", "database_version", "reliability", "uri",
    "search_engine", "search_engine_score", "smallmolecule_abundance_sub",
    "smallmolecule_abundance_stdev_sub", "smallmolecule_abundance_std_error_sub",
}


def _curl(url: str, timeout: int = 90) -> str:
    return subprocess.run(["curl", "-sSL", "--max-time", str(timeout), url],
                          capture_output=True, text=True, timeout=timeout + 10).stdout


def _download(acc: str) -> Path:
    """Fetch the NMR MAF + sample file for an accession into the cache; return dir."""
    d = CACHE / acc
    d.mkdir(parents=True, exist_ok=True)
    listing = _curl(f"{FTP}/{acc}/")
    mafs = sorted(set(re.findall(r'm_[A-Za-z0-9_.\-]*maf\.tsv', listing)))
    sfiles = sorted(set(re.findall(r's_[A-Za-z0-9_.\-]*\.txt', listing)))
    nmr = [m for m in mafs if "NMR" in m.upper()] or mafs
    if not nmr or not sfiles:
        raise RuntimeError(f"{acc}: could not locate an NMR MAF + sample file on the FTP.")
    maf_path, s_path = d / "maf.tsv", d / "s.txt"
    if not maf_path.exists():
        maf_path.write_text(_curl(f"{FTP}/{acc}/{nmr[0]}"))
    if not s_path.exists():
        s_path.write_text(_curl(f"{FTP}/{acc}/{sfiles[0]}"))
    return d


def _organism_part_col(sdf: pd.DataFrame) -> Optional[str]:
    for c in sdf.columns:
        if str(c).strip().lower() in ("characteristics[organism part]", "organism part"):
            return c
    return None


def _build_tables(cfg: Dict) -> Dict:
    """Download + assemble Table 1 (metabolite×sample, matrix-filtered) and a
    Sample→condition map. Returns bytes + a small provenance dict."""
    d = _download(cfg["accession"])
    maf = pd.read_csv(d / "maf.tsv", sep="\t", low_memory=False)
    sdf = pd.read_csv(d / "s.txt", sep="\t", low_memory=False)
    sn = "Sample Name"
    lab = cfg["label_column"]
    if sn not in sdf.columns or lab not in sdf.columns:
        raise RuntimeError(f"{cfg['id']}: sample file missing '{sn}' or '{lab}'.")
    op = _organism_part_col(sdf)
    smap = {}
    for _, r in sdf.iterrows():
        name = str(r[sn]).strip()
        matrix = str(r[op]).strip().lower() if op else ""
        smap[name] = (matrix, str(r[lab]).strip())
    sample_cols = [c for c in maf.columns if c not in _ANN]
    want_matrix = (cfg.get("matrix") or "").lower()
    keep = [c for c in sample_cols
            if str(c).strip() in smap
            and (not want_matrix or smap[str(c).strip()][0] == want_matrix)
            and smap[str(c).strip()][1] and smap[str(c).strip()][1].lower() != "nan"]
    if len(keep) < 10:
        raise RuntimeError(f"{cfg['id']}: only {len(keep)} matched samples after matrix filter.")
    t1 = maf[["metabolite_identification"] + keep]
    b1 = io.StringIO(); t1.to_csv(b1, sep="\t", index=False)
    t2 = pd.DataFrame({"Sample Name": keep,
                       "Condition": [smap[str(c).strip()][1] for c in keep]})
    b2 = io.StringIO(); t2.to_csv(b2, sep="\t", index=False)
    return {"t1": b1.getvalue().encode(), "t2": b2.getvalue().encode(),
            "ftp": f"{FTP}/{cfg['accession']}/",
            "study_url": f"https://www.ebi.ac.uk/metabolights/{cfg['accession']}"}


def _mw_json(url: str):
    txt = _curl(url, 60)
    return json.loads(txt) if txt.strip() else {}


def _build_tables_mw(cfg: Dict) -> Dict:
    """Assemble Table 1 + Table 2 from a Metabolomics Workbench NMR study via its
    REST API (`/factors` for the group of each sample, `/data` for the per-sample
    named-metabolite matrix). Groups are parsed from the `factor_key` field."""
    sid = cfg["accession"]
    key = cfg["factor_key"].lower()
    facs = _mw_json(f"{_MW}/{sid}/factors")
    fitems = list(facs.values()) if isinstance(facs, dict) else facs
    smap = {}
    for it in fitems:
        name = str(it.get("local_sample_id", "")).strip()
        grp = None
        for part in str(it.get("factors", "")).split("|"):
            if key in part.lower() and ":" in part:
                grp = part.split(":", 1)[1].strip()
        if name and grp:
            smap[name] = grp
    data = _mw_json(f"{_MW}/{sid}/data")
    ditems = list(data.values()) if isinstance(data, dict) else data
    all_samples = set()
    for d in ditems:
        all_samples.update((d.get("DATA") or {}).keys())
    samples = [s for s in sorted(all_samples) if s in smap]
    if len(samples) < 10 or not ditems:
        raise RuntimeError(f"{cfg['id']}: only {len(samples)} labeled samples / {len(ditems)} metabolites from MW.")
    t1 = "metabolite_identification\t" + "\t".join(samples) + "\n"
    for d in ditems:
        nm = str(d.get("metabolite_name", "unknown"))
        dd = d.get("DATA") or {}
        t1 += nm + "\t" + "\t".join(str(dd.get(s, "")) for s in samples) + "\n"
    t2 = "Sample Name\tCondition\n" + "".join(f"{s}\t{smap[s]}\n" for s in samples)
    return {"t1": t1.encode(), "t2": t2.encode(),
            "ftp": f"{_MW}/{sid}/data",
            "study_url": f"https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&StudyID={sid}"}


def run_cohort(cfg: Dict, *, k: int = 8, repeats: int = 5,
               permutations: int = 200, ci_boot: int = 1000) -> Dict:
    tb = _build_tables_mw(cfg) if cfg.get("source") == "workbench" else _build_tables(cfg)
    multiclass = bool(cfg.get("multiclass"))
    X, _names, _g = biomarkers.build_matrix(tb["t1"])
    meta = spectral_cohort.parse_metadata(tb["t2"])
    label_map, info = spectral_cohort.derive_labels(
        meta, [str(s) for s in X.index], label_column="Condition", multiclass=multiclass)
    rows = [s for s in X.index if str(s) in label_map]
    sub = X.loc[rows].dropna(axis=1, how="all")
    y = np.array([label_map[str(s)] for s in rows])
    groups = np.array([str(s) for s in rows])   # 1 sample = 1 patient
    res = biomarker_engine.discover(
        sub.values, y, k=k, repeats=repeats, feature_names=list(sub.columns),
        groups=groups, permutations=permutations, ci_boot=ci_boot, panel_sizes=(1, 3, 5))
    cn = info.get("class_names") or {}
    if not cn:
        pos = info.get("positive_class")
        neg = next((c for c in info.get("classes", []) if c != pos), "other")
        cn = {0: neg, 1: pos}
    dv = differential.differential_analysis(
        sub.values, y, list(sub.columns), class_names={int(kk): v for kk, v in cn.items()})
    return {"cfg": cfg, "info": info, "n": len(rows), "n_metabolites": sub.shape[1],
            "discover": res, "differential": dv, **{k: tb[k] for k in ("ftp", "study_url")}}


def _print(rep: Dict) -> None:
    cfg, res, info, dv = rep["cfg"], rep["discover"], rep["info"], rep["differential"]
    m = res["classification_metrics"]
    ci = res.get("honest_roc_auc_ci95")
    ci_s = f"  95% CI [{ci[0]}–{ci[1]}]" if ci else ""
    bal = ", ".join(f"{k}: {v}" for k, v in info.get("class_balance", {}).items())
    print(f"\n=== {cfg['accession']} · {cfg['disease']} · {cfg['biofluid']} (¹H NMR) — EXTERNAL, never bundled ===")
    print(f"source: {rep['study_url']}   (data: {rep['ftp']})")
    print(f"n={rep['n']} ({bal}) · {rep['n_metabolites']} metabolites · condition '{info.get('label_column')}'")
    if res["task_type"] == "multiclass":
        pcr = " · ".join(f"{k}:{v}" for k, v in (m.get("per_class_recall") or {}).items())
        print(f"macro ROC-AUC (one-vs-rest) {res['honest_roc_auc']}{ci_s}  ·  leaky {res['leaky_roc_auc']} "
              f"(inflation {res['leakage_inflation']})  ·  permutation p {res['permutation_p_value']} ({res['n_permutations']} perms)")
        print(f"accuracy {m['accuracy']} · macro F1/prec/recall {m['f1_macro']}/{m['precision_macro']}/{m['recall_macro']} · per-class recall {pcr}")
    else:
        print(f"honest ROC-AUC {res['honest_roc_auc']}{ci_s}  ·  leaky {res['leaky_roc_auc']} "
              f"(inflation {res['leakage_inflation']})  ·  permutation p {res['permutation_p_value']} ({res['n_permutations']} perms)")
        print(f"accuracy {m['accuracy']} · sensitivity {m['sensitivity']} · specificity {m['specificity']} · "
              f"precision {m['precision']} · recall {m['recall']} · F1 {m['f1']}")
    print(f"stable panel: {', '.join(res['stable_panel']) or '—'}")
    sweep = " · ".join(f"top-{e['k']}→AUC {e['honest_roc_auc']}" for e in res['panel_sweep'])
    print(f"minimal-panel sweep: {sweep}")
    sig = [r for r in dv["table"] if r["q_value"] is not None and r["q_value"] < 0.05][:6]
    print(f"differential (q<0.05, {dv['n_significant']} total): " +
          "; ".join(f"{r['metabolite']} (q {r['q_value']}, {r['direction']}"
                    + (f", log2FC {r['log2_fold_change']}" if r.get('log2_fold_change') is not None else "") + ")"
                    for r in sig))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="External cross-cohort validation (open MetaboLights ¹H NMR).")
    ap.add_argument("--cohort", help="run a single cohort id (default: all)")
    ap.add_argument("--list", action="store_true", help="list configured cohorts")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--permutations", type=int, default=200)
    args = ap.parse_args(argv)
    if args.list:
        for c in COHORTS:
            print(f"{c['id']:20s} {c['accession']:10s} {c['disease']} ({c['biofluid']})")
        return 0
    cohorts = [c for c in COHORTS if (not args.cohort or c["id"] == args.cohort)]
    if not cohorts:
        print(f"No cohort matches '{args.cohort}'. Use --list.", file=sys.stderr)
        return 2
    print("RuuPhenome external validation — RUO, open MetaboLights data only.\n"
          "Leakage-safe nested CV · bootstrap 95% CI · permutation test · BH-FDR differential.")
    for c in cohorts:
        try:
            _print(run_cohort(c, repeats=args.repeats, permutations=args.permutations))
        except Exception as exc:
            print(f"\n=== {c['id']} — FAILED: {exc} ===")
    print("\nNote: research-use-only. External cohorts prove the pipeline is honest and reproducible "
          "on unseen data; they are not clinical validation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
