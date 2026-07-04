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

from typing import Dict, List, Sequence

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


def d2o_assessment(matched_shifts: Sequence[float], expected_shifts: int) -> Dict:
    """Split a metabolite's matched shifts into reliable (non-exchangeable C-H) vs
    D2O-unreliable (water/HDO or downfield-exchangeable), and report a caveat.

    Returns the reliable subset (`robust_shifts`) that downstream coverage /
    abundance should be computed from, plus flags for auditability."""
    flags = [classify_shift(float(s)) for s in matched_shifts]
    robust = [float(s) for s, f in zip(matched_shifts, flags) if f == "non_exchangeable"]
    n_water = flags.count("water_hdo")
    n_exch = flags.count("exchangeable_risk")
    robust_coverage = len(robust) / max(1, expected_shifts)
    caveat = None
    if not robust and (n_water or n_exch):
        caveat = ("identified only via residual-water/HDO or exchangeable-proton "
                  "signals — unreliable in D2O; rejected.")
    elif n_water or n_exch:
        parts = []
        if n_water:
            parts.append(f"{n_water} matched shift(s) in the residual-water/HDO window "
                         f"({WATER_HDO_WINDOW[0]}–{WATER_HDO_WINDOW[1]} ppm)")
        if n_exch:
            parts.append(f"{n_exch} in the exchangeable region (>{DOWNFIELD_EXCHANGEABLE_MIN} ppm)")
        caveat = ("; ".join(parts) + " excluded from evidence (unreliable in D2O); "
                  "identification rests on non-exchangeable C-H resonances.")
    return {
        "robust_shifts": [round(s, 4) for s in robust],
        "n_nonexchangeable": len(robust),
        "n_water_hdo": n_water,
        "n_exchangeable_risk": n_exch,
        "robust_coverage": round(robust_coverage, 3),
        "usable_in_d2o": len(robust) > 0,
        "d2o_caveat": caveat,
        "reference_standard": REFERENCE_STANDARD,
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
