"""
Track-1 identification benchmark — the reproducible DETERMINISTIC baseline (RUO).

Generates labeled synthetic ¹H-NMR mixtures from the OPEN reference library
(known component identities at their reference shifts, with per-sample
concentration variation, peak-width, ppm drift, and baseline noise), runs the
deterministic identification (`annotate` + NNLS `deconvolve` + target-decoy FDR),
and scores identification **precision / recall / F1** against the planted ground
truth. This is the number any LEARNED classifier (pSCNN, the GISSMO H100 hero)
must beat before it earns its place.

**Honesty (read before quoting a number).** This is an **in-distribution
synthetic** benchmark: mixtures are built from the SAME reference library the
matcher uses, so recall is *optimistic* — it measures overlap-resolution and
false-positive control under realistic noise/drift, NOT real cross-instrument or
D2O-blood accuracy. A held-out REAL benchmark is still required (see
docs/TRACK1_PLAN.md open questions). The ground-truth panel is restricted to
compounds that are actually observable in D2O (≥1 non-exchangeable resonance),
consistent with the D2O guard.

Usage (from `backend/`):
    python -m nmr_api.track1_benchmark
    python -m nmr_api.track1_benchmark --cohorts 12 --present 15 --bins 2000
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

try:
    from . import identification_quality as idq
    from . import spectral_cohort as sc
except ImportError:  # pragma: no cover - direct execution
    import identification_quality as idq  # type: ignore
    import spectral_cohort as sc  # type: ignore


def _identifiable_library(refs: Dict[str, List[float]], lo: float, hi: float) -> List[str]:
    """Library compounds observable in D2O within [lo, hi]: ≥2 shifts in range and
    ≥1 non-exchangeable (excludes water-only / exchangeable-dominant entries)."""
    out = []
    for nm, shifts in refs.items():
        in_range = [s for s in shifts if lo <= s <= hi]
        robust = [s for s in in_range if idq.classify_shift(s) == "non_exchangeable"]
        if len(in_range) >= 2 and robust:
            out.append(nm)
    return out


def simulate_cohort(bin_ppm: np.ndarray, refs: Dict[str, List[float]], *,
                    n_present: int, n_samples: int, seed: int,
                    noise: float = 0.01, sigma_ppm: float = 0.012) -> Tuple[pd.DataFrame, Set[str]]:
    """One labeled cohort: `n_present` known compounds planted across `n_samples`
    (varying concentration + ppm drift + noise). Returns (matrix, truth set)."""
    rng = np.random.default_rng(seed)
    lo, hi = float(bin_ppm.min()), float(bin_ppm.max())
    candidates = _identifiable_library(refs, lo, hi)
    present = list(rng.choice(candidates, size=min(n_present, len(candidates)), replace=False))
    rows = []
    for _ in range(n_samples):
        spec = np.abs(rng.normal(noise, noise * 0.3, size=len(bin_ppm)))   # baseline
        for nm in present:
            amp = float(rng.lognormal(0.0, 0.4))                            # concentration
            for sh in refs[nm]:
                if idq.classify_shift(sh) != "non_exchangeable":
                    continue                                               # D2O: skip exchangeable
                jitter = float(rng.normal(0.0, 0.002))                     # ppm drift
                spec += amp * np.exp(-((bin_ppm - (sh + jitter)) ** 2) / (2.0 * sigma_ppm ** 2))
        rows.append(spec)
    X = pd.DataFrame(np.asarray(rows), columns=np.round(bin_ppm, 5),
                     index=[f"m{i}" for i in range(n_samples)])
    return X, set(present)


def score(called: Sequence[str], truth: Set[str]) -> Dict:
    called, truth = set(called), set(truth)
    tp, fp, fn = len(called & truth), len(called - truth), len(truth - called)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def _agg(rows: List[Dict]) -> Dict:
    return {k: round(float(np.mean([r[k] for r in rows])), 3)
            for k in ("precision", "recall", "f1")}


def run_baseline(*, n_cohorts: int = 10, n_present: int = 15, n_samples: int = 6,
                 n_bins: int = 2000, lo: float = 0.5, hi: float = 9.5, seed: int = 0) -> Dict:
    bin_ppm = np.linspace(lo, hi, n_bins)
    refs = sc.REFERENCE_SHIFTS
    ann, fdr = [], []
    for c in range(n_cohorts):
        X, truth = simulate_cohort(bin_ppm, refs, n_present=n_present,
                                   n_samples=n_samples, seed=seed + c)
        a = sc.annotate(X, bin_ppm)
        ann.append(score({m["metabolite"] for m in a["metabolites"]}, truth))
        d = sc.deconvolve(X, bin_ppm)
        fdr.append(score({m["metabolite"] for m in d["metabolites"] if m["passes_fdr"]}, truth))
    return {
        "config": {"cohorts": n_cohorts, "compounds_planted": n_present,
                   "samples_per_cohort": n_samples, "bins": n_bins, "ppm": [lo, hi]},
        "annotate_baseline": _agg(ann),          # permissive reference-shift matching
        "deconvolve_fdr_baseline": _agg(fdr),    # NNLS + target-decoy FDR (controlled)
        "note": ("IN-DISTRIBUTION synthetic (library == matcher), so recall is optimistic; "
                 "measures overlap-resolution + FP control, not real cross-instrument/D2O "
                 "accuracy. Ground truth restricted to D2O-observable compounds. A held-out "
                 "REAL benchmark is still required before any accuracy claim."),
    }


def run_comparison(*, panel_names: List[str] = None, n_test: int = 20,
                   n_bins: int = 1000, epochs: int = 25, seed: int = 0) -> Dict:
    """Deterministic (NNLS+FDR) vs learned pSCNN vs hybrid evidence, on an EASY
    (in-distribution) and a HARD (ppm-drift-beyond-tolerance) condition — the
    honest picture of where a learned channel adds value."""
    from . import pscnn
    refs = sc.REFERENCE_SHIFTS
    default = ["glucose", "lactate", "valine", "leucine", "isoleucine", "alanine",
               "citrate", "creatinine", "acetate", "pyruvate", "glutamine", "tyrosine"]
    panel = {n: refs[n] for n in (panel_names or default) if n in refs}
    names = list(panel)
    pgrid = pscnn.make_grid(384)
    model, meta = pscnn.train(panel, grid=pgrid, n_mixtures=350, epochs=epochs,
                              lr=3e-3, batch_size=64, seed=seed, save=False)
    bins = np.linspace(0.5, 9.5, n_bins)

    def make_test(cond, s):
        rng = np.random.default_rng(10_000 + s)
        k = int(rng.integers(3, min(8, len(names))))
        present = set(rng.choice(names, size=k, replace=False))
        spec = np.abs(rng.normal(cond["noise"], cond["noise"] * 0.3, len(bins))).astype(float)
        for n in present:
            d = float(rng.normal(0.0, cond["drift"]))
            for sh in panel[n]:
                if idq.classify_shift(sh) != "non_exchangeable":
                    continue
                spec += float(rng.lognormal(0, 0.4)) * np.exp(-((bins - (sh + d)) ** 2) / (2 * 0.012 ** 2))
        return spec, present

    conds = {"easy (low noise, no drift)": {"noise": 0.02, "drift": 0.0},
             "hard (noise + 0.04 ppm drift)": {"noise": 0.05, "drift": 0.04}}
    out = {}
    for cname, cond in conds.items():
        det, psc, hyb = [], [], []
        for s in range(n_test):
            spec, truth = make_test(cond, s)
            Xdf = pd.DataFrame(spec[None, :], columns=np.round(bins, 5), index=["t0"])
            d = sc.deconvolve(Xdf, bins)
            det_called = {m["metabolite"] for m in d["metabolites"] if m["passes_fdr"]} & set(names)
            probs = pscnn.identify((model, meta), bins, spec)
            psc_called = {n for n in names if probs.get(n, 0.0) > 0.5}
            hyb_called = {n for n in names if (n in det_called) or probs.get(n, 0.0) > 0.6}
            det.append(score(det_called, truth))
            psc.append(score(psc_called, truth))
            hyb.append(score(hyb_called, truth))
        out[cname] = {"deterministic": _agg(det), "pscnn": _agg(psc), "hybrid": _agg(hyb)}
    return {
        "panel_size": len(panel), "n_test_per_condition": n_test, "n_bins": n_bins,
        "pscnn_final_loss": meta["loss_history"][-1],
        "conditions": out,
        "note": ("Synthetic. EASY is IN-DISTRIBUTION (both saturate — an optimistic number, "
                 "not evidence of superiority). HARD applies ppm drift BEYOND the deterministic "
                 "fixed tolerance, where the drift-augmented pSCNN should hold up better — the "
                 "value case for a learned channel. The DECISIVE test is real held-out BMRB / "
                 "GISSMO spectra (endpoints in docs/TRACK1_PLAN.md); this is not clinical validation."),
    }


_LIB_SYN = {
    "glutamate": ["glutamate", "l-glutamic acid", "glutamic acid"],
    "lactate": ["lactate", "l-lactic acid", "lactic acid"],
    "citrate": ["citrate", "citric acid"],
    "acetate": ["acetate", "acetic acid"],
    "pyruvate": ["pyruvate", "pyruvic acid"],
    "formate": ["formate", "formic acid"],
    "succinate": ["succinate", "succinic acid"],
}


def _match_library(kw: str, refs: Dict[str, List[float]]) -> Optional[str]:
    """Map a GISSMO compound key to a reference-library key (synonyms + substring)."""
    for cand in _LIB_SYN.get(kw, [kw]):
        if cand in refs:
            return cand
    hits = [n for n in refs if kw in n.lower()]
    return sorted(hits, key=len)[0] if hits else None


def run_real_validation(*, n_test: int = 25, n_bins: int = 1000, epochs: int = 30,
                        seed: int = 0) -> Dict:
    """HELD-OUT: reference library + pSCNN training use our (HMDB-derived) shifts;
    the TEST spectra use INDEPENDENT GISSMO real ¹H shifts. Compares deterministic
    vs pSCNN vs hybrid on real-shift spectra, clean and with a realistic referencing
    offset + noise."""
    from . import external_reference as ext
    from . import pscnn
    refs = sc.REFERENCE_SHIFTS
    real = ext.real_shifts()                      # {compound: GISSMO real shifts} (cached)
    panel, real_map = {}, {}
    for cname, rshifts in real.items():
        libkey = _match_library(cname, refs)
        rin = [s for s in rshifts if 0.5 <= s <= 9.5]
        if libkey is None or len(rin) < 2:
            continue
        panel[libkey] = refs[libkey]
        real_map[libkey] = rin
    names = list(panel)
    if len(names) < 4:
        return {"error": f"only {len(names)} compounds fetched from GISSMO — check network.",
                "compounds": names}
    model, meta = pscnn.train(panel, grid=pscnn.make_grid(384), epochs=epochs, save=False, seed=seed)
    bins = np.linspace(0.5, 9.5, n_bins)

    def make_test(offset, noise, s):
        rng = np.random.default_rng(30_000 + s)
        k = int(rng.integers(3, min(8, len(names) + 1)))
        present = set(rng.choice(names, size=min(k, len(names)), replace=False))
        spec = np.abs(rng.normal(noise, noise * 0.3, len(bins))).astype(float)
        for n in present:
            amp = float(rng.lognormal(0, 0.4))
            for sh in real_map[n]:
                if idq.classify_shift(sh) != "non_exchangeable":
                    continue
                spec += amp * np.exp(-((bins - (sh + offset)) ** 2) / (2 * 0.012 ** 2))
        return spec, present

    conds = {"GISSMO real shifts (clean)": (0.0, 0.02),
             "GISSMO real + 0.03 ppm referencing offset + noise": (0.03, 0.05)}
    out = {}
    for cname, (off, noise) in conds.items():
        det, psc, hyb = [], [], []
        for s in range(n_test):
            spec, truth = make_test(off, noise, s)
            Xdf = pd.DataFrame(spec[None, :], columns=np.round(bins, 5), index=["t0"])
            d = sc.deconvolve(Xdf, bins)
            dc = {m["metabolite"] for m in d["metabolites"] if m["passes_fdr"]} & set(names)
            probs = pscnn.identify((model, meta), bins, spec)
            pc = {n for n in names if probs.get(n, 0.0) > 0.5}
            hc = {n for n in names if (n in dc) or probs.get(n, 0.0) > 0.6}
            det.append(score(dc, truth)); psc.append(score(pc, truth)); hyb.append(score(hc, truth))
        out[cname] = {"deterministic": _agg(det), "pscnn": _agg(psc), "hybrid": _agg(hyb)}
    return {
        "n_compounds": len(names), "compounds": names,
        "source": "GISSMO physically-exact ¹H shifts (spin systems fit to experimental BMRB spectra)",
        "conditions": out,
        "note": ("HELD-OUT external check: reference library + pSCNN training use our HMDB-derived "
                 "shifts; TEST spectra use INDEPENDENT GISSMO real shifts. 'clean' = real shifts as-is; "
                 "the second adds a realistic global referencing offset + noise. Physically-exact "
                 "simulated (from experimental BMRB fits), not a clinical claim — the strongest "
                 "external check available without raw patient spectra."),
    }


def _render_real(peaks: List[List[float]], grid: np.ndarray, *,
                 conc: float = 1.0, offset: float = 0.0, sigma: float = 0.008) -> np.ndarray:
    """Render a REAL experimental peak list ([[ppm, intensity], ...]) onto a grid:
    an intensity-weighted sum of narrow Gaussians. Unlike the library fingerprint,
    this carries the measured relative intensities and multiplet splitting."""
    spec = np.zeros(len(grid), dtype=float)
    for ppm, inten in peaks:
        spec += conc * float(inten) * np.exp(-((grid - (float(ppm) + offset)) ** 2) / (2.0 * sigma ** 2))
    return spec


_CANON_SYN = {
    "glutamicacid": "glutamate", "lglutamicacid": "glutamate", "glutamic": "glutamate",
    "citricacid": "citrate", "lacticacid": "lactate", "aceticacid": "acetate",
    "succinicacid": "succinate", "pyruvicacid": "pyruvate", "formicacid": "formate",
}


def _canonical(name: str) -> str:
    """Normalise a compound name to a comparison key: strip stereo/charge prefixes,
    lowercase, drop spaces/hyphens, map acid→conjugate-base synonyms. Lets the
    GISSMO quantifier's vocabulary ('(+/-)-Glucose', 'L-alanine') map to the
    reference-library keys ('glucose', 'alanine') for scoring, without the
    substring collisions ('alanine' ⊂ 'phenylalanine') a naive match would cause."""
    s = str(name).strip().lower().replace("_", " ")
    for pre in ("(+/-)-", "(+)-", "(-)-", "(r)-", "(s)-", "(2s)-", "dl-", "d-", "l-"):
        while s.startswith(pre):
            s = s[len(pre):]
    s = s.strip().replace(" ", "").replace("-", "")
    return _CANON_SYN.get(s, s)


def run_bmrb_validation(*, n_test: int = 25, n_bins: int = 1000, epochs: int = 30,
                        seed: int = 0, quant_threshold: float = 0.05) -> Dict:
    """HELD-OUT on REAL MEASURED spectra. Test mixtures are built from BMRB
    metabolomics **experimental** ¹H peak lists (real positions + real intensities
    + real multiplet structure, measured on real spectrometers, DSS-referenced) —
    information our HMDB-derived library and the pSCNN's centroid-Gaussian training
    never encoded. Compares deterministic (NNLS+FDR) vs pSCNN vs hybrid, clean and
    with a realistic referencing offset + noise. The strongest independent check
    available without raw patient spectra (RUO; not clinical validation)."""
    try:
        from . import bmrb_experimental as be
        from . import pscnn
    except ImportError:  # pragma: no cover
        import bmrb_experimental as be  # type: ignore
        import pscnn  # type: ignore
    refs = sc.REFERENCE_SHIFTS
    real = be.load_bundle()                       # {compound: [[ppm, intensity], ...]}
    panel = {n: refs[n] for n in real if n in refs}
    real_map = {n: real[n] for n in panel}
    names = list(panel)
    if len(names) < 6:
        return {"error": f"only {len(names)} BMRB experimental peak lists available — "
                         f"run `python -m nmr_api.bmrb_experimental` off-VM to build the bundle.",
                "compounds": names}
    model, meta = pscnn.train(panel, grid=pscnn.make_grid(384), epochs=epochs, save=False, seed=seed)
    bins = np.linspace(0.5, 9.5, n_bins)

    # Optional: the trained GISSMO quantifier (F8), scored on this REAL held-out set
    # it never trained on. Loads only if a checkpoint is present (else skipped, so
    # the benchmark runs with or without it). Its 94-compound GISSMO vocabulary is
    # mapped to the panel's library keys via canonical names.
    quant = None
    q2panel: Dict[str, str] = {}
    try:
        from . import quantifier as _quant
        if _quant.available() and _quant.CHECKPOINT_PATH.exists():
            quant = _quant.load_checkpoint()
            canon2panel = {_canonical(p): p for p in names}
            q2panel = {q: canon2panel[_canonical(q)] for q in quant[1]["names"]
                       if _canonical(q) in canon2panel}
    except Exception:
        quant = None

    def make_test(offset, noise, s):
        rng = np.random.default_rng(40_000 + s)
        k = int(rng.integers(3, min(8, len(names) + 1)))
        present = set(rng.choice(names, size=min(k, len(names)), replace=False))
        spec = np.abs(rng.normal(noise, noise * 0.3, len(bins))).astype(float)
        for n in present:
            spec += _render_real(real_map[n], bins, conc=float(rng.lognormal(0, 0.4)), offset=offset)
        m = float(spec.max())
        if m > 0:
            spec /= m
        return spec, present

    conds = {"BMRB real spectra (clean)": (0.0, 0.02),
             "BMRB real + 0.03 ppm referencing offset + noise": (0.03, 0.05)}
    out = {}
    for cname, (off, noise) in conds.items():
        perm, det, psc, hyb = [], [], [], []
        qnt, hyq = [], []
        for s in range(n_test):
            spec, truth = make_test(off, noise, s)
            Xdf = pd.DataFrame(spec[None, :], columns=np.round(bins, 5), index=["t0"])
            a = sc.annotate(Xdf, bins)
            ac = {m["metabolite"] for m in a["metabolites"]} & set(names)
            d = sc.deconvolve(Xdf, bins)
            dc = {m["metabolite"] for m in d["metabolites"] if m["passes_fdr"]} & set(names)
            probs = pscnn.identify((model, meta), bins, spec)
            pc = {n for n in names if probs.get(n, 0.0) > 0.5}
            hc = {n for n in names if (n in dc) or probs.get(n, 0.0) > 0.6}
            perm.append(score(ac, truth)); det.append(score(dc, truth))
            psc.append(score(pc, truth)); hyb.append(score(hc, truth))
            if quant is not None:                          # F8 GISSMO quantifier evidence
                qconc = _quant.identify(quant, bins, spec, threshold=quant_threshold)
                qc = {q2panel[q] for q in qconc if q in q2panel}
                qnt.append(score(qc, truth)); hyq.append(score(hc | qc, truth))
        row = {"deterministic_permissive": _agg(perm), "deterministic_fdr": _agg(det),
               "pscnn": _agg(psc), "hybrid": _agg(hyb)}
        if quant is not None:
            row["quantifier"] = _agg(qnt); row["hybrid+quant"] = _agg(hyq)
        out[cname] = row
    return {
        "n_compounds": len(names), "compounds": names,
        "quantifier_loaded": quant is not None,
        "quantifier_vocab_mapped": len(q2panel),
        "source": ("BMRB metabolomics EXPERIMENTAL 1D ¹H peak lists (measured spectra, "
                   "DSS-referenced) — bundled open_data/bmrb_experimental_peaks.json"),
        "conditions": out,
        "note": ("HELD-OUT on REAL measured data: test spectra use BMRB experimental peak "
                 "positions + intensities + multiplet structure (not our HMDB centroids, not "
                 "GISSMO sims). Shift positions are shift-correlated with the library (BMRB feeds "
                 "HMDB), but intensities/multiplets/noise are genuinely new. RUO; not clinical."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Track-1 deterministic identification baseline (RUO, synthetic).")
    ap.add_argument("--cohorts", type=int, default=10)
    ap.add_argument("--present", type=int, default=15)
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--bins", type=int, default=2000)
    ap.add_argument("--compare", action="store_true",
                    help="deterministic vs pSCNN vs hybrid (easy + hard/drift, synthetic)")
    ap.add_argument("--validate-real", action="store_true",
                    help="HELD-OUT: deterministic vs pSCNN vs hybrid on real GISSMO shifts")
    ap.add_argument("--validate-bmrb", action="store_true",
                    help="HELD-OUT: deterministic vs pSCNN vs hybrid on REAL BMRB experimental spectra")
    args = ap.parse_args(argv)
    if args.validate_bmrb:
        res = run_bmrb_validation(n_test=25, n_bins=1000)
        if "error" in res:
            print("BMRB validation unavailable:", res["error"]); return 1
        print("RuuPhenome — Track-1 HELD-OUT validation on REAL BMRB experimental ¹H spectra (RUO).")
        print(f"panel ({res['n_compounds']}): {', '.join(res['compounds'])}")
        print(f"source: {res['source']}")
        if res.get("quantifier_loaded"):
            print(f"F8 GISSMO quantifier: LOADED ({res['quantifier_vocab_mapped']} vocab compounds mapped to the panel)")
        else:
            print("F8 GISSMO quantifier: not loaded (drop gissmo_quantifier.pt into models/ to include it)")
        for cond, m in res["conditions"].items():
            print(f"\n[{cond}]")
            for method, s in m.items():
                print(f"   {method:24s} precision {s['precision']} · recall {s['recall']} · F1 {s['f1']}")
        print(f"\n{res['note']}")
        return 0
    if args.validate_real:
        res = run_real_validation(n_test=25, n_bins=1000)
        if "error" in res:
            print("Real validation unavailable:", res["error"]); return 1
        print("RuuPhenome — Track-1 HELD-OUT validation on real GISSMO ¹H shifts (RUO).")
        print(f"panel ({res['n_compounds']}): {', '.join(res['compounds'])}")
        print(f"source: {res['source']}")
        for cond, m in res["conditions"].items():
            print(f"\n[{cond}]")
            for method in ("deterministic", "pscnn", "hybrid"):
                s = m[method]
                print(f"   {method:14s} precision {s['precision']} · recall {s['recall']} · F1 {s['f1']}")
        print(f"\n{res['note']}")
        return 0
    if args.compare:
        res = run_comparison(n_test=20, n_bins=1000)
        print("RuuPhenome — Track-1 identification: deterministic vs pSCNN vs hybrid (RUO, synthetic).")
        print(f"panel {res['panel_size']} · {res['n_test_per_condition']} test mixtures/condition · pSCNN loss {res['pscnn_final_loss']}")
        for cond, m in res["conditions"].items():
            print(f"\n[{cond}]")
            for method in ("deterministic", "pscnn", "hybrid"):
                s = m[method]
                print(f"   {method:14s} precision {s['precision']} · recall {s['recall']} · F1 {s['f1']}")
        print(f"\n{res['note']}")
        return 0
    res = run_baseline(n_cohorts=args.cohorts, n_present=args.present,
                       n_samples=args.samples, n_bins=args.bins)
    a, f = res["annotate_baseline"], res["deconvolve_fdr_baseline"]
    print("RuuPhenome — Track-1 identification baseline (deterministic; RUO, synthetic).")
    print(f"config: {res['config']}")
    print(f"annotate (reference-shift matching):     precision {a['precision']} · recall {a['recall']} · F1 {a['f1']}")
    print(f"deconvolve + target-decoy FDR (controlled): precision {f['precision']} · recall {f['recall']} · F1 {f['f1']}")
    print(f"\n{res['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
