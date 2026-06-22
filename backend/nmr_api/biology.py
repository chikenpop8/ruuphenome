"""
Biological interpretation layer for RuuPhenome.

Two capabilities, both fully offline (no network on demo day):

  Option A — per-metabolite biology cards
      annotate(name) -> {role, disease_associations, pathways, direction_note}
      Curated from HMDB 5.0 metabolite cards for the serum metabolites in the
      MTBLS242 set (and common NMRformer compounds). This answers the judges'
      "why does this biomarker matter biologically?" question.

  Option B — pathway over-representation (enrichment) analysis
      pathway_enrichment(biomarkers, background) -> ranked pathways with p-values
      Standard hypergeometric (Fisher) over-representation test, the same method
      MetaboAnalyst uses. Answers "are my biomarkers concentrated in a specific
      biological pathway, more than chance?" — quantitative, reproducible.

Scaling note: enrichment cost is linear in the number of pathways, NOT
combinatorial in the number of metabolites. Adding metabolites only grows the
curation table; the statistics stay instant.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence

from scipy.stats import hypergeom


# ── name normalization ───────────────────────────────────────────────────────
# Metabolite names arrive in many forms: "L-Alanine", "L-Lactic acid",
# "(R)-3-Hydroxybutyric acid". Normalize to a canonical lookup key.
_STEREO = re.compile(r"^(l|d|dl|r|s|\(r\)|\(s\)|\(rs\)|beta|alpha|n|o|pi)[\-\s]+", re.I)


def normalize(name: str) -> str:
    """Reduce a metabolite name to a canonical lookup key."""
    if not name:
        return ""
    key = name.strip().lower()
    key = key.replace("(r)-", "").replace("(s)-", "").replace("(rs)-", "")
    # strip a single leading stereo/locant prefix
    key = _STEREO.sub("", key)
    # unify "...ic acid" / "...ate" naming and drop trailing " acid"
    key = re.sub(r"\s+acid$", "", key)
    key = key.replace("lactic", "lactate").replace("pyruvic", "pyruvate")
    key = key.replace("hydroxybutyric", "hydroxybutyrate")
    key = key.strip()
    return key


# ── Option A: curated per-metabolite biology (HMDB-sourced) ──────────────────
# Each entry: role, disease_associations, pathways (KEGG/SMPDB names), direction.
METABOLITE_BIOLOGY: Dict[str, Dict] = {
    "3-hydroxybutyrate": {
        "role": "Primary ketone body; alternative brain/heart fuel during fasting or low insulin.",
        "disease_associations": "↑ in diabetic ketoacidosis, prolonged fasting, type-1/2 diabetes.",
        "pathways": ["Ketone body metabolism", "Butanoate metabolism"],
    },
    "acetate": {
        "role": "Short-chain fatty acid; product of gut microbial fermentation and ethanol metabolism.",
        "disease_associations": "↑ with alcohol intake, altered gut microbiome; energy substrate.",
        "pathways": ["Pyruvate metabolism", "Ketone body metabolism"],
    },
    "acetoacetate": {
        "role": "Ketone body produced in the liver from fatty-acid breakdown.",
        "disease_associations": "↑ in diabetic ketoacidosis and starvation ketosis.",
        "pathways": ["Ketone body metabolism", "Butanoate metabolism"],
    },
    "alanine": {
        "role": "Glucogenic amino acid; carries nitrogen from muscle to liver (glucose–alanine cycle).",
        "disease_associations": "↑ in insulin resistance and type-2 diabetes; reflects protein catabolism.",
        "pathways": ["Alanine, aspartate and glutamate metabolism", "Glucose–alanine cycle"],
    },
    "valine": {
        "role": "Branched-chain amino acid (BCAA); essential, used for muscle protein and energy.",
        "disease_associations": "↑ BCAAs strongly associated with insulin resistance and type-2 diabetes risk.",
        "pathways": ["Valine, leucine and isoleucine degradation", "BCAA metabolism"],
    },
    "leucine": {
        "role": "Branched-chain amino acid (BCAA); potent activator of mTOR / muscle protein synthesis.",
        "disease_associations": "↑ BCAAs predict insulin resistance and future type-2 diabetes.",
        "pathways": ["Valine, leucine and isoleucine degradation", "BCAA metabolism"],
    },
    "isoleucine": {
        "role": "Branched-chain amino acid (BCAA); essential, glucogenic and ketogenic.",
        "disease_associations": "↑ BCAAs associated with insulin resistance and metabolic syndrome.",
        "pathways": ["Valine, leucine and isoleucine degradation", "BCAA metabolism"],
    },
    "hypoxanthine": {
        "role": "Purine-degradation intermediate; marker of ATP turnover and tissue hypoxia.",
        "disease_associations": "↑ in hypoxia, ischemia, oxidative stress and gout-related purine turnover.",
        "pathways": ["Purine metabolism"],
    },
    "citrate": {
        "role": "Central TCA-cycle intermediate; links carbohydrate, fat and energy metabolism.",
        "disease_associations": "Altered in mitochondrial dysfunction, renal stone risk, metabolic disease.",
        "pathways": ["Citric acid cycle (TCA cycle)"],
    },
    "creatinine": {
        "role": "Breakdown product of muscle creatine; excreted by the kidney at a steady rate.",
        "disease_associations": "↑ in reduced kidney function / diabetic nephropathy; used to gauge renal clearance.",
        "pathways": ["Arginine and proline metabolism", "Creatine metabolism"],
    },
    "glutamine": {
        "role": "Most abundant blood amino acid; major nitrogen and carbon carrier between tissues.",
        "disease_associations": "Glutamine/glutamate ratio shifts in insulin resistance and type-2 diabetes.",
        "pathways": ["Alanine, aspartate and glutamate metabolism", "Nitrogen metabolism"],
    },
    "glycine": {
        "role": "Simplest amino acid; one-carbon metabolism, glutathione and heme synthesis.",
        "disease_associations": "↓ glycine associated with insulin resistance, obesity and type-2 diabetes.",
        "pathways": ["Glycine, serine and threonine metabolism", "One-carbon metabolism"],
    },
    "histidine": {
        "role": "Essential amino acid; precursor of histamine, antioxidant and metal-chelating roles.",
        "disease_associations": "↓ histidine linked to inflammation, oxidative stress and chronic kidney disease.",
        "pathways": ["Histidine metabolism"],
    },
    "isopropanol": {
        "role": "Exogenous solvent / disinfectant; not endogenously produced.",
        "disease_associations": "Detected after isopropanol exposure; usually a sample-handling artifact.",
        "pathways": ["Xenobiotic / exogenous"],
    },
    "lactate": {
        "role": "End product of anaerobic glycolysis; key energy shuttle (Cori cycle).",
        "disease_associations": "↑ in hypoxia, sepsis, mitochondrial dysfunction and insulin resistance.",
        "pathways": ["Glycolysis / Gluconeogenesis", "Pyruvate metabolism"],
    },
    "methanol": {
        "role": "Exogenous alcohol; trace dietary/microbial origin.",
        "disease_associations": "Mostly exogenous; elevated after methanol exposure.",
        "pathways": ["Xenobiotic / exogenous"],
    },
    "dimethyl sulfone": {
        "role": "Sulfur-containing compound from diet and gut microbial metabolism (MSM).",
        "disease_associations": "Mostly dietary; reflects sulfur metabolism and supplement intake.",
        "pathways": ["Sulfur / methionine metabolism"],
    },
    "phenylalanine": {
        "role": "Essential aromatic amino acid; precursor of tyrosine and catecholamines.",
        "disease_associations": "↑ in insulin resistance, type-2 diabetes and (extreme) in phenylketonuria.",
        "pathways": ["Phenylalanine, tyrosine and tryptophan biosynthesis", "Aromatic amino acid metabolism"],
    },
    "pyruvate": {
        "role": "End product of glycolysis; gateway to TCA cycle, lactate and alanine.",
        "disease_associations": "↑ with impaired oxidative metabolism and insulin resistance.",
        "pathways": ["Glycolysis / Gluconeogenesis", "Pyruvate metabolism", "Citric acid cycle (TCA cycle)"],
    },
    "tyrosine": {
        "role": "Aromatic amino acid; precursor of dopamine, adrenaline and thyroid hormones.",
        "disease_associations": "↑ tyrosine associated with insulin resistance and type-2 diabetes risk.",
        "pathways": ["Phenylalanine, tyrosine and tryptophan biosynthesis", "Aromatic amino acid metabolism"],
    },
    "glucose": {
        "role": "Primary circulating sugar and energy source; tightly regulated by insulin.",
        "disease_associations": "↑ fasting glucose is the defining biomarker of diabetes mellitus.",
        "pathways": ["Glycolysis / Gluconeogenesis"],
    },
}


# ── Option B: pathway → metabolite membership (for enrichment) ────────────────
# Built from the curated table above so the two stay consistent. Pathway names
# follow KEGG / SMPDB conventions.
def _build_pathway_index() -> Dict[str, set]:
    index: Dict[str, set] = {}
    for metabolite, info in METABOLITE_BIOLOGY.items():
        for pathway in info.get("pathways", []):
            index.setdefault(pathway, set()).add(metabolite)
    return index


PATHWAY_INDEX: Dict[str, set] = _build_pathway_index()


# ── public API ───────────────────────────────────────────────────────────────
def annotate(name: str) -> Optional[Dict]:
    """Return curated biology for one metabolite, or None if not in the table."""
    info = METABOLITE_BIOLOGY.get(normalize(name))
    if not info:
        return None
    return {
        "role": info["role"],
        "disease_associations": info["disease_associations"],
        "pathways": list(info["pathways"]),
        "source": "HMDB 5.0 (curated)",
    }


def pathway_enrichment(
    biomarkers: Sequence[str],
    background: Optional[Sequence[str]] = None,
    *,
    min_overlap: int = 2,
) -> List[Dict]:
    """
    Hypergeometric over-representation test.

    Asks, for each pathway: are the biomarkers enriched in this pathway more
    than expected by chance, given the background set of measured metabolites?

    Args:
        biomarkers: the discovered biomarker metabolite names.
        background: all measured metabolites (the statistical universe).
                    Defaults to the full curated set when not supplied.
        min_overlap: minimum biomarkers-in-pathway to report (avoids 1-hit noise).

    Returns ranked pathways with p-value, overlap and the matching metabolites.
    """
    bm = {normalize(b) for b in biomarkers if normalize(b)}
    if background:
        universe = {normalize(b) for b in background if normalize(b)}
        universe |= bm  # biomarkers are always part of the universe
    else:
        universe = set(METABOLITE_BIOLOGY.keys()) | bm

    N = len(universe)            # population size
    n = len(bm & universe)       # number of "successes" drawn (biomarkers)
    if N == 0 or n == 0:
        return []

    results = []
    for pathway, members in PATHWAY_INDEX.items():
        members_in_universe = members & universe
        K = len(members_in_universe)          # successes in population
        if K == 0:
            continue
        hits = bm & members_in_universe
        k = len(hits)                          # observed overlap
        if k < min_overlap:
            continue
        # P(overlap >= k) under hypergeometric null
        p_value = float(hypergeom.sf(k - 1, N, K, n))
        fold_enrichment = (k / n) / (K / N) if (n and K) else 0.0
        results.append({
            "pathway": pathway,
            "p_value": round(p_value, 5),
            "overlap": k,
            "pathway_size": K,
            "fold_enrichment": round(fold_enrichment, 2),
            "metabolites": sorted(hits),
        })

    results.sort(key=lambda r: (r["p_value"], -r["overlap"]))
    return results


def interpret_panel(
    biomarkers: Sequence[str],
    background: Optional[Sequence[str]] = None,
) -> Dict:
    """
    Full biological interpretation of a biomarker panel: per-metabolite cards
    plus pathway enrichment. This is the one call the API/UI uses.
    """
    annotations = []
    for name in biomarkers:
        bio = annotate(name)
        annotations.append({"metabolite": name, "biology": bio})

    enrichment = pathway_enrichment(biomarkers, background)
    n_annotated = sum(1 for a in annotations if a["biology"])

    return {
        "metabolite_biology": annotations,
        "pathway_enrichment": enrichment,
        "coverage": {
            "annotated": n_annotated,
            "total": len(annotations),
        },
        "top_pathway": enrichment[0]["pathway"] if enrichment else None,
    }
