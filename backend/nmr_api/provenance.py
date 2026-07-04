"""
Track-1 provenance & condition handling.

A binned ¹H-NMR matrix carries sample IDs, ppm bins, and intensities — it does NOT
carry pH, solvent, temperature, buffer, salt, sample concentration, or preparation.
Those come from a separate metadata file / README / methods document / Bruker files /
MetaboLights ISA-Tab / MAF / organizer documentation, and they materially change ¹H
CHEMICAL SHIFTS (pH sets the protonation state of ionizable groups; solvent sets
referencing and whether exchangeable OH/NH/SH protons are even observable; temperature
moves the water resonance and exchange rates; ionic strength and concentration cause
smaller shifts).

This module:
  * assembles what the matrix DOES tell us (shape, ppm range, bin width, QC warnings —
    computed in spectral_cohort and passed in as `matrix_profile`),
  * normalizes user / optional-metadata condition fields (never failing on missing ones),
  * decides CONDITION-AWARE behaviour — crucially, whether the D₂O exchangeable-proton
    guard applies for the stated solvent.

Scientific honesty (must stay true): RuuPhenome provides provenance-aware WARNINGS and
D₂O-aware exchangeable-proton RULES. It does NOT have a curated pH/solvent/temperature-
specific spectral library and does NOT perform Chenomx-style pH/solvent-aware lineshape
fitting. Reference shifts are aqueous/HMDB-derived and are not re-modelled per condition.
"""

from __future__ import annotations

import re
from typing import Dict, Optional


# ── solvent model ────────────────────────────────────────────────────────────
# `guard=True` → apply the D₂O exchangeable-proton disappearance rule (O/N/S–H
# vanish, C–H persists). Only aqueous/D₂O keeps it; named organic solvents disable
# it (exchangeable protons remain observable). "unknown" defaults to the aqueous
# rule (the norm for ¹H metabolomics) but is flagged as an assumption.
NMR_SOLVENTS: Dict[str, Dict] = {
    "d2o":         {"label": "D₂O (aqueous)",             "guard": True,  "aqueous": True,  "reference": "DSS or TSP"},
    "aqueous_d2o": {"label": "Aqueous buffer + D₂O",      "guard": True,  "aqueous": True,  "reference": "DSS or TSP"},
    "dmso-d6":     {"label": "DMSO-d₆",                   "guard": False, "aqueous": False, "reference": "TMS"},
    "cdcl3":       {"label": "CDCl₃",                     "guard": False, "aqueous": False, "reference": "TMS"},
    "cd3od":       {"label": "CD₃OD (methanol-d₄)",       "guard": False, "aqueous": False, "reference": "TMS"},
    "cd3cn":       {"label": "CD₃CN (acetonitrile-d₃)",   "guard": False, "aqueous": False, "reference": "TMS"},
    "acetone-d6":  {"label": "Acetone-d₆",                "guard": False, "aqueous": False, "reference": "TMS"},
    "benzene-d6":  {"label": "Benzene-d₆ (C₆D₆)",         "guard": False, "aqueous": False, "reference": "TMS"},
    "pyridine-d5": {"label": "Pyridine-d₅",               "guard": False, "aqueous": False, "reference": "TMS"},
    "unknown":     {"label": "unknown",                   "guard": True,  "aqueous": None,  "reference": "unknown",
                    "assumed_aqueous": True},
}

# Sample types that commonly appear in ¹H metabolomics (for the UI + provenance).
SAMPLE_TYPES = ("urine", "serum", "plasma", "csf", "cell extract", "tissue extract",
                "plant extract", "marine extract", "food/beverage", "unknown")


def _clean(s) -> str:
    return re.sub(r"[\s_\-]", "", str(s or "").strip().lower())


_SOLVENT_ALIASES: Dict[str, str] = {_clean(k): v for k, v in {
    "d2o": "d2o", "deuterium oxide": "d2o", "deuterated water": "d2o",
    "aqueous": "aqueous_d2o", "aqueous d2o": "aqueous_d2o", "aqueous_d2o": "aqueous_d2o",
    "water": "aqueous_d2o", "h2o": "aqueous_d2o", "h2o+d2o": "aqueous_d2o",
    "h2o/d2o": "aqueous_d2o", "buffer": "aqueous_d2o", "phosphate buffer": "aqueous_d2o",
    "dmso": "dmso-d6", "dmso-d6": "dmso-d6", "dmso d6": "dmso-d6", "d6-dmso": "dmso-d6",
    "cdcl3": "cdcl3", "chloroform-d": "cdcl3", "chloroform": "cdcl3", "cdcl₃": "cdcl3",
    "cd3od": "cd3od", "methanol-d4": "cd3od", "methanol": "cd3od", "meod": "cd3od", "cd₃od": "cd3od",
    "cd3cn": "cd3cn", "acetonitrile-d3": "cd3cn", "acetonitrile": "cd3cn", "cd₃cn": "cd3cn",
    "acetone-d6": "acetone-d6", "acetone": "acetone-d6",
    "benzene-d6": "benzene-d6", "c6d6": "benzene-d6", "benzene": "benzene-d6",
    "pyridine-d5": "pyridine-d5", "pyridine": "pyridine-d5",
}.items()}


def normalize_solvent(value) -> str:
    """Map a free-text solvent to a canonical key; unrecognized/empty → 'unknown'."""
    c = _clean(value)
    if not c or c in ("unknown", "na", "none", "notprovided", "n/a"):
        return "unknown"
    return _SOLVENT_ALIASES.get(c, "unknown")


def solvent_info(value) -> Dict:
    return NMR_SOLVENTS.get(normalize_solvent(value), NMR_SOLVENTS["unknown"])


def applies_d2o_guard(value) -> bool:
    """Whether the D₂O exchangeable-proton disappearance rule should apply."""
    return bool(solvent_info(value)["guard"])


def _num(v) -> Optional[float]:
    """Lenient numeric parse ('7.4', '600 MHz', '25 °C' → float); else None."""
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("na", "n/a", "none", "unknown", "not provided", "not_provided"):
        return None
    m = re.search(r"-?\d+\.?\d*", s.replace(",", "."))
    return float(m.group()) if m else None


def _txt(v) -> Optional[str]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("na", "n/a", "none", "not provided", "not_provided"):
        return None
    return s


# ── optional conditions file ─────────────────────────────────────────────────
# NOT a full ISA-Tab / MAF / Bruker parser — a best-effort key:value (or 2-column
# CSV/TSV) reader for a small conditions/README-style file, with common synonyms
# incl. a few MetaboLights ISA-style keys. Richer parsers are future work.
_FIELD_SYNONYMS: Dict[str, str] = {_clean(k): v for k, v in {
    "solvent": "solvent", "nmr solvent": "solvent", "nmr_solvent": "solvent", "chromatography solvent": "solvent",
    "ph": "ph", "sample ph": "ph",
    "temperature": "temperature", "temp": "temperature", "temperature_c": "temperature",
    "temperature_k": "temperature", "acquisition temperature": "temperature",
    "field": "field_mhz", "field_mhz": "field_mhz", "frequency": "field_mhz",
    "spectrometer frequency": "field_mhz", "mhz": "field_mhz", "magnetic field strength": "field_mhz",
    "sample type": "sample_type", "sample_type": "sample_type", "matrix": "sample_type",
    "biofluid": "sample_type", "organism part": "sample_type", "material type": "sample_type",
    "buffer": "buffer_notes", "buffer_notes": "buffer_notes", "preparation": "buffer_notes",
    "sample preparation": "buffer_notes", "extraction": "buffer_notes",
    "salt": "salt_ionic_strength", "ionic strength": "salt_ionic_strength", "salt concentration": "salt_ionic_strength",
    "concentration": "sample_concentration", "sample concentration": "sample_concentration",
    "dilution": "sample_concentration", "dilution factor": "sample_concentration",
    "normalization": "normalization", "normalisation": "normalization", "preprocessing": "normalization",
    "excluded regions": "excluded_regions", "excluded": "excluded_regions", "exclusion": "excluded_regions",
    "internal standard": "internal_standard", "reference": "internal_standard",
    "reference compound": "internal_standard", "chemical shift reference": "internal_standard",
    "standard": "internal_standard",
}.items()}


def parse_condition_file(raw: bytes) -> Dict[str, str]:
    """Best-effort parse of a small conditions file into user-field values.
    Accepts 'field: value', 'field<TAB>value', or 'field,value' lines. First value
    wins per field. Unknown keys are ignored. Not an ISA/Bruker parser."""
    out: Dict[str, str] = {}
    if not raw:
        return out
    text = raw.decode("utf-8", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key = val = None
        for sep in ("\t", ":", ","):
            if sep in line:
                k, _, v = line.partition(sep)
                key, val = k, v
                break
        if key is None:
            continue
        field = _FIELD_SYNONYMS.get(_clean(key))
        v = (val or "").strip()
        if field and v and field not in out:
            out[field] = v
    return out


# ── provenance assembly ──────────────────────────────────────────────────────
HONESTY_NOTE = (
    "RuuPhenome provides provenance-aware warnings and D₂O-aware exchangeable-proton "
    "rules. It does NOT have a fully curated pH/solvent/temperature-specific spectral "
    "library and does not perform Chenomx-style pH/solvent-aware lineshape fitting — "
    "reference shifts are aqueous/HMDB-derived and are not re-modelled per condition."
)


def build_provenance(matrix_profile: Optional[Dict] = None,
                     user: Optional[Dict] = None,
                     *, normalization_applied: Optional[str] = None) -> Dict:
    """Assemble the Track-1 provenance block from the auto-detected matrix profile and
    user/optional-metadata condition fields. Never fails on missing fields — records
    'not provided' and emits condition-aware warnings instead.

    `user` keys (all optional): solvent, ph, temperature, field_mhz, sample_type,
    buffer_notes, salt_ionic_strength, sample_concentration, normalization,
    excluded_regions, internal_standard (advanced/optional only)."""
    matrix_profile = matrix_profile or {}
    user = user or {}

    solvent_raw = _txt(user.get("solvent"))
    solvent = normalize_solvent(solvent_raw)
    sinfo = NMR_SOLVENTS.get(solvent, NMR_SOLVENTS["unknown"])
    apply_d2o = bool(sinfo["guard"])

    ph = _num(user.get("ph"))
    temperature = _num(user.get("temperature"))
    field_mhz = _num(user.get("field_mhz"))

    conditions = {
        "solvent": solvent,
        "solvent_label": sinfo["label"],
        "solvent_raw": solvent_raw,
        "apply_d2o_guard": apply_d2o,
        "solvent_assumed_aqueous": bool(sinfo.get("assumed_aqueous")),
        "ph": ph,
        "temperature_c": temperature if temperature is not None else "not provided",
        "field_mhz": field_mhz,
        "sample_type": (_txt(user.get("sample_type")) or "unknown"),
        "buffer_notes": _txt(user.get("buffer_notes")),
        "salt_ionic_strength": _txt(user.get("salt_ionic_strength")),
        "sample_concentration": _txt(user.get("sample_concentration")),
        "normalization": (_txt(user.get("normalization")) or normalization_applied or "unknown"),
        "excluded_regions": _txt(user.get("excluded_regions")),
        # ADVANCED / optional only — NOT a required main condition field.
        "internal_standard": (_txt(user.get("internal_standard")) or "unknown"),
    }

    warnings = list(matrix_profile.get("warnings", []))
    # condition-aware, NEVER-FAIL warnings
    if ph is None:
        warnings.append(
            "pH not provided — pH-dependent peak shifts cannot be corrected. Ionizable groups "
            "(carboxylates, amines, histidine, phosphate, citrate) move with pH; matching uses "
            "fixed aqueous reference shifts.")
    if solvent == "unknown":
        warnings.append(
            "Solvent not specified — assuming aqueous/D₂O exchangeable-proton behaviour (the norm "
            "for ¹H metabolomics); solvent-specific chemical-shift effects are not modeled. Set the "
            "solvent to change D₂O handling.")
    elif not apply_d2o:
        warnings.append(
            f"Solvent {sinfo['label']}: the D₂O-specific exchangeable-proton disappearance rule is "
            "NOT applied — OH/NH/SH protons remain observable. Note the reference library is "
            "aqueous/HMDB-derived, so organic-solvent shift positions differ and are not modeled.")
    if temperature is None:
        warnings.append(
            "Temperature not provided — recorded as 'not provided'. Temperature shifts the water "
            "resonance and changes exchange rates.")

    return {
        "matrix": matrix_profile,
        "conditions": conditions,
        "warnings": warnings,
        "capabilities": {
            "provenance_aware_warnings": True,
            "d2o_aware_exchangeable_rules": True,
            "ph_solvent_temperature_curated_library": False,
            "chenomx_style_condition_aware_fitting": False,
        },
        "honesty": HONESTY_NOTE,
    }
