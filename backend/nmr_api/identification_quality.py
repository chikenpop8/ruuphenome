"""
Track-1 identification-quality layer — honesty / bioinformatics-correctness.

Two enforced, auditable guards applied to every reference-shift compound match so
identifications are defensible and never fabricate chemistry:

1. **MSI identification level** (Metabolomics Standards Initiative; Sumner et al.
   2007, Metabolomics 3:211, doi:10.1007/s11306-007-0082-2). RuuPhenome matches
   observed ppm to a curated reference library WITHOUT an in-house authentic-
   standard spike-in and without orthogonal (J-coupling) confirmation, so a
   confident match is **Level 2 — "putatively annotated" by spectral similarity —
   never Level 1** (which requires an authentic standard run under identical
   conditions). Weak/single-resonance matches drop to Level 3; matches with no
   reliable non-exchangeable evidence are not identified (Level 4 / rejected).
   Organizer-provided annotations are reported as a separate "provided" provenance.

2. **D2O / exchangeable-proton guard.** The competition matrix is blood in D2O.
   Exchangeable protons (-COOH / -OH / -NH / -NH2) exchange with deuterium and
   shift or vanish, and residual water/HDO dominates ~4.7-4.9 ppm (Haslauer et al.
   2019, Anal Chem 91:11063). We therefore, WITHOUT fabricating per-compound
   solvent corrections:
     - flag matches in the residual-water/HDO window and in the downfield
       exchangeable region (COOH/NH, > ~9.5 ppm) and EXCLUDE them from evidence;
     - compute coverage/abundance from NON-exchangeable C-H resonances only
       (prefer non-exchangeable resonances — the honest fix);
     - reject an identification that has no reliable non-exchangeable resonance;
     - keep **DSS** (pH-independent) as the 0.00 ppm reference, not TSP.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Residual water / HDO in D2O (temperature-dependent, ~4.79 ppm at 25 °C). Peaks
# here are solvent, not metabolite — standard practice excludes this window.
WATER_HDO_WINDOW = (4.70, 4.90)
# Downfield exchangeable region: carboxylic-acid / amide / NH3+ / aldehyde protons
# typically appear > ~9.5 ppm in H2O and are usually absent in D2O.
DOWNFIELD_EXCHANGEABLE_MIN = 9.5
REFERENCE_STANDARD = "DSS @ 0.00 ppm (pH-independent; preferred over protein-binding, pH-sensitive TSP)"

MSI_SCHEME = (
    "Metabolomics Standards Initiative identification levels (Sumner et al. 2007): "
    "L1 = identified (authentic standard, ≥2 orthogonal properties); "
    "L2 = putatively annotated (spectral-similarity to a library, no standard); "
    "L3 = putative compound-class; L4 = unknown."
)


def classify_shift(ppm: float) -> str:
    """'water_hdo', 'exchangeable_risk', or 'non_exchangeable' for one ¹H shift."""
    if WATER_HDO_WINDOW[0] <= ppm <= WATER_HDO_WINDOW[1]:
        return "water_hdo"
    if ppm >= DOWNFIELD_EXCHANGEABLE_MIN:
        return "exchangeable_risk"
    return "non_exchangeable"


# ── structure-based exchangeability (the chemically-correct basis) ────────────
# Exchangeability is a property of the BONDED ATOM, not the ppm value: protons on
# O / N / S (–OH, –NH, –COOH, –SH) exchange with deuterium in D₂O and vanish, while
# C–H protons persist. We therefore count exchangeable vs non-exchangeable protons
# directly from each metabolite's structure (SMILES) — a far more faithful D₂O
# reliability signal than a fixed ppm cutoff. (No RDKit dependency; a minimal,
# validated implicit-H counter over the organic subset. Validated against
# glucose 5-OH/7-CH, urea 4-NH/0-CH, citrate, taurine sulfonate, creatinine C=N.)
_ORGANIC_VALENCE = {"B": 3, "C": 4, "N": 3, "O": 2, "P": 3, "S": 2,
                    "F": 1, "Cl": 1, "Br": 1, "I": 1}
_SMILES_TOKEN = re.compile(r"\[[^\]]+\]|Br|Cl|[BCNOPSFI]|[bcnops]|=|#|:|/|\\|\.|\(|\)|%\d\d|\d")


def _parse_bracket_atom(tok: str):
    body = tok[1:-1]
    m = re.match(r"\d*([A-Z][a-z]?|[bcnops])", body)
    el = m.group(1) if m else "C"
    aromatic = el[0].islower()
    hm = re.search(r"H(\d*)", body)
    h = (int(hm.group(1)) if hm.group(1) else 1) if hm else 0
    return (el.capitalize() if aromatic else el), aromatic, h


def exchangeable_protons(smiles: Optional[str]) -> Optional[Dict]:
    """Count exchangeable (O/N/S–H) vs non-exchangeable (C–H) protons from a SMILES.
    Returns {'exchangeable', 'nonexchangeable_ch', 'total_h'} or None if unparseable."""
    if not smiles:
        return None
    atoms: List[Dict] = []
    ring: Dict[str, tuple] = {}
    last = None
    stack: List[Optional[int]] = []
    pending = 1
    for t in _SMILES_TOKEN.findall(smiles):
        if t == "(":
            stack.append(last)
        elif t == ")":
            last = stack.pop() if stack else last
        elif t in ("=", "#", ":", "/", "\\"):
            pending = {"=": 2, "#": 3}.get(t, 1)
        elif t == ".":
            last = None; pending = 1
        elif t.startswith("%") or (len(t) == 1 and t.isdigit()):
            if t in ring:
                ai, bo = ring.pop(t)
                atoms[ai]["b"] += bo; atoms[ai]["d"] += 1
                if last is not None:
                    atoms[last]["b"] += bo; atoms[last]["d"] += 1
            else:
                ring[t] = (last, pending)
            pending = 1
        else:
            if t.startswith("["):
                el, aromatic, h = _parse_bracket_atom(t); bracket = True
            else:
                el = t.upper() if t.islower() else t
                aromatic = t.islower(); h = None; bracket = False
            atoms.append({"el": el, "ar": aromatic, "br": bracket, "exH": h, "b": 0, "d": 0})
            i = len(atoms) - 1
            if last is not None:
                atoms[last]["b"] += pending; atoms[last]["d"] += 1
                atoms[i]["b"] += pending; atoms[i]["d"] += 1
            last = i; pending = 1
    if not atoms:
        return None
    exch = ch = 0
    for a in atoms:
        el = a["el"]
        if a["br"]:
            h = a["exH"] or 0
        elif a["ar"]:
            h = (1 if a["d"] <= 2 else 0) if el == "C" else 0   # aromatic C–H; ring N/O/S bare → 0
        else:
            v = _ORGANIC_VALENCE.get(el)
            h = 0 if v is None else max(0, v - round(a["b"]))
        if el in ("O", "N", "S"):
            exch += h
        elif el == "C":
            ch += h
    return {"exchangeable": int(exch), "nonexchangeable_ch": int(ch), "total_h": int(exch + ch)}


_SMILES_INDEX: Optional[Dict[str, str]] = None
_EXCH_CACHE: Dict[str, Optional[Dict]] = {}


def _canon_name(s: str) -> str:
    s = str(s).strip().lower()
    for p in ("(+/-)-", "(+)-", "(-)-", "(r)-", "(s)-", "(2s)-", "(2r)-", "dl-", "d-", "l-"):
        while s.startswith(p):
            s = s[len(p):]
    return s.replace(" ", "").replace("-", "").replace("_", "")


def _smiles_for(name: str) -> Optional[str]:
    """Offline SMILES lookup from the bundled PubChem cache (no network)."""
    global _SMILES_INDEX
    if _SMILES_INDEX is None:
        _SMILES_INDEX = {}
        try:
            cache = json.loads((Path(__file__).resolve().parent / "cache" /
                                "pubchem_cache.json").read_text())
            for k, v in cache.items():
                sm = v.get("smiles") if isinstance(v, dict) else None
                if sm:
                    _SMILES_INDEX.setdefault(_canon_name(k), sm)
        except Exception:
            _SMILES_INDEX = {}
    return _SMILES_INDEX.get(_canon_name(name))


def metabolite_exchangeable(name: str) -> Optional[Dict]:
    """Structural exchangeable-proton counts for a metabolite by name (cached,
    offline). None if no SMILES is available for it."""
    if name in _EXCH_CACHE:
        return _EXCH_CACHE[name]
    r = exchangeable_protons(_smiles_for(name))
    _EXCH_CACHE[name] = r
    return r


def d2o_assessment(matched_shifts: Sequence[float], expected_shifts: int, *,
                   structure: Optional[Dict] = None, apply_guard: bool = True) -> Dict:
    """Split a metabolite's matched shifts into reliable (non-exchangeable C-H) vs
    D2O-unreliable (water/HDO or downfield-exchangeable), and — when the metabolite's
    STRUCTURE is known (`structure` from `metabolite_exchangeable`) — fold in the
    chemically-correct exchangeable-proton inventory (O/N/S–H exchange out; C–H stays).

    `apply_guard` is condition-aware: it must be True ONLY for aqueous/D2O. For
    non-aqueous solvents (DMSO-d6, CDCl3, CD3OD, CD3CN, acetone-d6, benzene-d6,
    pyridine-d5) the exchangeable protons do NOT disappear, so the guard is disabled —
    every matched shift counts and no exchangeability downgrade is applied.

    Returns the reliable subset (`robust_shifts`), auditability flags, a per-molecule
    `d2o_grade` (reliable / caution / weak / invisible / not_applicable) and a caveat."""
    if not apply_guard:
        # Non-D2O solvent: exchangeable protons remain observable → all matched shifts
        # are usable; report the structural inventory as INFORMATION only, never reject.
        robust = [float(s) for s in matched_shifts]
        struct_exch = struct_ch = obs_frac = None
        if structure and structure.get("total_h"):
            struct_exch = int(structure["exchangeable"])
            struct_ch = int(structure["nonexchangeable_ch"])
            obs_frac = round(struct_ch / structure["total_h"], 2)
        return {
            "robust_shifts": [round(s, 4) for s in robust],
            "n_nonexchangeable": len(robust),
            "n_water_hdo": 0,
            "n_exchangeable_risk": 0,
            "robust_coverage": round(len(robust) / max(1, expected_shifts), 3),
            "usable_in_d2o": len(robust) > 0,
            "d2o_grade": "not_applicable",
            "structural_exchangeable": struct_exch,
            "structural_ch": struct_ch,
            "observable_fraction": obs_frac,
            "d2o_caveat": ("Non-aqueous solvent: exchangeable OH/NH/SH protons remain observable; "
                           "the D₂O disappearance rule is not applied. (Reference shifts are "
                           "aqueous/HMDB-derived; organic-solvent positions are not modeled.)"),
            "reference_standard": "TMS @ 0.00 ppm (non-aqueous convention)",
            "guard_applied": False,
        }
    flags = [classify_shift(float(s)) for s in matched_shifts]
    robust = [float(s) for s, f in zip(matched_shifts, flags) if f == "non_exchangeable"]
    n_water = flags.count("water_hdo")
    n_exch = flags.count("exchangeable_risk")
    robust_coverage = len(robust) / max(1, expected_shifts)

    # structure-derived counts (chemically-correct exchangeability)
    struct_exch = struct_ch = obs_frac = None
    if structure and structure.get("total_h"):
        struct_exch = int(structure["exchangeable"])
        struct_ch = int(structure["nonexchangeable_ch"])
        obs_frac = round(struct_ch / structure["total_h"], 2)

    # grade: a molecule with NO C-H is invisible in D2O; otherwise weak (no robust
    # matched resonance), caution (some matched peaks are exchangeable/water), or reliable.
    usable = len(robust) > 0
    if struct_ch == 0 and (struct_exch or 0) > 0:   # entirely OH/NH/SH — nothing to see in D2O
        grade, usable = "invisible", False
    elif not usable:
        grade = "weak"
    elif n_water or n_exch:
        grade = "caution"
    else:
        grade = "reliable"

    # dynamic caveat — leads with the real per-molecule exchangeable inventory
    parts: List[str] = []
    if struct_exch is not None:
        obs = f" ({int(obs_frac * 100)}% of its ¹H observable in D₂O)" if obs_frac is not None else ""
        parts.append(f"{struct_exch} exchangeable (OH/NH/SH) proton(s) exchange out in D₂O; "
                     f"{struct_ch} non-exchangeable C–H proton(s) carry the reliable signal{obs}.")
    if grade == "invisible":
        parts.append("No non-exchangeable C–H protons — not observable in D₂O.")
    elif not robust and (n_water or n_exch):
        parts.append("Matched only via residual-water/HDO or exchangeable-proton signals — "
                     "unreliable in D₂O; rejected.")
    elif n_water or n_exch:
        sub = []
        if n_water:
            sub.append(f"{n_water} matched shift(s) in the residual-water/HDO window "
                       f"({WATER_HDO_WINDOW[0]}–{WATER_HDO_WINDOW[1]} ppm)")
        if n_exch:
            sub.append(f"{n_exch} in the exchangeable region (>{DOWNFIELD_EXCHANGEABLE_MIN} ppm)")
        parts.append("; ".join(sub) + " excluded from evidence; identification rests on "
                     "non-exchangeable C–H resonances.")
    caveat = " ".join(parts) if parts else None

    return {
        "robust_shifts": [round(s, 4) for s in robust],
        "n_nonexchangeable": len(robust),
        "n_water_hdo": n_water,
        "n_exchangeable_risk": n_exch,
        "robust_coverage": round(robust_coverage, 3),
        "usable_in_d2o": usable,
        "d2o_grade": grade,
        "structural_exchangeable": struct_exch,
        "structural_ch": struct_ch,
        "observable_fraction": obs_frac,
        "d2o_caveat": caveat,
        "reference_standard": REFERENCE_STANDARD,
        "guard_applied": True,
    }


def msi_level(robust_coverage: float, n_nonexchangeable: int, *,
              provided: bool = False, has_authentic_standard: bool = False) -> Dict:
    """MSI identification level for one match (honest — Level 1 only with an
    authentic standard, which this pipeline does not run)."""
    if provided:
        return {
            "msi_level": 2,
            "msi_label": "putatively annotated (organizer-provided annotation)",
            "msi_rationale": ("Position provided by the organizer as a training/reference "
                              "annotation; reported as provenance, still MSI Level 2 unless "
                              "confirmed against an authentic standard."),
        }
    if has_authentic_standard and n_nonexchangeable >= 2:
        level, label = 1, "identified (authentic-standard match)"
    elif n_nonexchangeable >= 2 and robust_coverage >= 0.5:
        level, label = 2, "putatively annotated (spectral-similarity to reference library)"
    elif n_nonexchangeable >= 1:
        level, label = 3, "putative characterization (single/weak non-exchangeable resonance)"
    else:
        level, label = 4, "not reliably identified (no non-exchangeable evidence in D2O)"
    return {
        "msi_level": level,
        "msi_label": label,
        "msi_rationale": ("Chemical-shift match to a curated reference library without an "
                          "in-house authentic-standard spike-in or orthogonal J-coupling "
                          "confirmation → MSI Level 2 at best (Sumner 2007); Level 1 requires "
                          "an authentic standard run under identical conditions."),
    }
