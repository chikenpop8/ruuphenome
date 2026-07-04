"""
Compound-library enrichment for the Chenomx-style Profiler frontend.

Takes a MetaboLights NMR results table and produces, per metabolite:
  - identity (name, ChEBI, formula, SMILES, InChI)
  - molecular weight (computed from formula)
  - predicted ¹H shifts (NMRTransformer or HMDB fallback)
  - a pseudo-concentration (µM) derived from mean abundance
  - external database reference URLs (HMDB, KEGG, PubChem, ChEBI)

This is what feeds the Reference Card + compound table in the web UI.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

import pandas as pd

from .biology import annotate as annotate_biology
from .enrich import enrich
from .pipeline import load_domain2
from .shifts_db import predict_shifts


# Standard atomic weights (g/mol) for elements common in metabolites.
ATOMIC_WEIGHTS = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974,
    "S": 32.06, "Na": 22.990, "K": 39.098, "Cl": 35.45, "Ca": 40.078,
    "Mg": 24.305, "F": 18.998, "Fe": 55.845, "Zn": 65.38,
}

_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def molecular_weight(formula: str) -> Optional[float]:
    """Compute molecular weight from a chemical formula string (e.g. C6H6N2O)."""
    if not formula or not isinstance(formula, str):
        return None
    total = 0.0
    found = False
    for element, count in _FORMULA_TOKEN.findall(formula):
        if element not in ATOMIC_WEIGHTS:
            continue
        n = int(count) if count else 1
        total += ATOMIC_WEIGHTS[element] * n
        found = True
    return round(total, 4) if found else None


def external_refs(chebi_id: str, name: str) -> Dict[str, Optional[str]]:
    """Build external-database reference URLs, Chenomx-style."""
    chebi_num = ""
    if chebi_id and chebi_id.upper().startswith("CHEBI:"):
        chebi_num = chebi_id.split(":", 1)[1]
    q = (name or "").replace(" ", "+")
    return {
        "hmdb": f"https://hmdb.ca/unearth/q?query={q}&searcher=metabolites",
        "kegg": f"https://www.genome.jp/dbget-bin/www_bget?compound+{q}" if name else None,
        "pubchem": f"https://pubchem.ncbi.nlm.nih.gov/#query={q}" if name else None,
        "chebi": f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:{chebi_num}" if chebi_num else None,
    }


def build_library(tsv_bytes: bytes) -> Dict:
    """
    Parse a MetaboLights TSV and return an enriched compound library plus
    spectrometer metadata for the Profiler UI.
    """
    df_meta, _df_abund, sample_cols = load_domain2(tsv_bytes)

    smiles_list = (df_meta["smiles"].fillna("").tolist()
                   if "smiles" in df_meta.columns else [""] * len(df_meta))
    predicted, backend = predict_shifts(smiles_list)

    # Scale mean abundance into a readable pseudo-concentration (µM).
    abund = df_meta["mean_abundance"].fillna(0)
    max_abund = abund.max() or 1.0

    compounds: List[Dict] = []
    for _, row in df_meta.iterrows():
        smi = row.get("smiles", "") if pd.notna(row.get("smiles", "")) else ""
        name = _s(row.get("metabolite_identification", ""))
        formula = _s(row.get("chemical_formula", ""))
        chebi = _s(row.get("database_identifier", ""))
        shifts = predicted.get(smi, [])

        mean_ab = row.get("mean_abundance", 0) or 0
        # Only assign a concentration when we actually have a peak fit.
        conc = round((mean_ab / max_abund) * 500, 2) if shifts and mean_ab else None

        inchi = _s(row.get("inchi", ""))
        meta = enrich(name, chebi, inchi)        # PubChem-enriched, cached

        compounds.append({
            "name": name,
            "chebi_id": chebi,
            "formula": formula,
            "molecular_weight": molecular_weight(formula),
            "smiles": smi,
            "inchi": inchi,
            "iupac_name": meta.get("iupac_name"),
            "alternate_names": meta.get("alternate_names", []),
            "cas_registry": meta.get("cas_registry"),
            "pubchem_cid": meta.get("pubchem_cid"),
            "inchikey": meta.get("inchikey"),
            "predicted_shifts": shifts,
            "concentration_uM": conc,
            "identified": bool(shifts),
            "cv_percent": _round(row.get("cv_percent")),
            "external_refs": meta.get("external_refs") or external_refs(chebi, name),
            "biology": annotate_biology(name),   # HMDB-curated biological role/disease/pathways
        })

    # Identified (has fit) first, then by concentration desc, like Chenomx.
    compounds.sort(key=lambda c: (not c["identified"], -(c["concentration_uM"] or 0)))

    return {
        "spectrometer": {
            "frequency_mhz": 599.48,   # typical 600 MHz serum metabolomics rig
            "ph": 7.00,                # assumed physiological default — NOT measured
            "ph_measured": False,      # nothing in this pipeline reads real sample pH yet
            "compound_count": len(compounds),
            "sample_count": len(sample_cols),
            "prediction_backend": backend,
        },
        "compounds": compounds,
    }


def _s(value) -> str:
    if value is None:
        return ""
    try:
        if isinstance(value, float) and pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _round(value) -> Optional[float]:
    try:
        f = float(value)
        if pd.isna(f):
            return None
        return round(f, 2)
    except (TypeError, ValueError):
        return None
