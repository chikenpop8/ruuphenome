"""
Biological interpretation layer for RuuPhenome.

Two capabilities, both fully offline (no network on demo day):

  Option A — per-metabolite biology cards
      annotate(name) -> {role, disease_associations, pathways, category, direction}
      Curated from HMDB 5.0 metabolite cards for the serum metabolites in the
      MTBLS242 set plus the common NMRformer serum panel. Answers the judges'
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
    key = _STEREO.sub("", key)
    key = re.sub(r"\s+acid$", "", key)
    key = key.replace("lactic", "lactate").replace("pyruvic", "pyruvate")
    key = key.replace("hydroxybutyric", "hydroxybutyrate")
    key = key.replace("hydroxyisobutyric", "hydroxyisobutyrate")
    key = key.replace("isovaleric", "isovalerate")
    key = key.replace("n,n-dimethylglycine", "dimethylglycine")
    key = key.strip()
    return key


# ── category metadata (drives UI grouping + colour) ──────────────────────────
CATEGORY_COLORS: Dict[str, str] = {
    "Energy / glycolysis": "#2e8b57",
    "Ketone bodies": "#c2741c",
    "TCA cycle": "#1f7a8c",
    "Amino acid — BCAA": "#7048b6",
    "Amino acid — aromatic": "#9c3587",
    "Amino acid — other": "#3f6fb0",
    "Kidney / muscle": "#b0413e",
    "Gut microbiome": "#5a8f29",
    "Purine metabolism": "#8c6d1f",
    "Choline / one-carbon": "#2c8c9c",
    "Lipid / membrane": "#a85a2a",
    "Exogenous / xenobiotic": "#888888",
}


# ── Option A: curated per-metabolite biology (HMDB-sourced) ──────────────────
# Each entry: role, disease_associations, pathways, category, direction.
METABOLITE_BIOLOGY: Dict[str, Dict] = {
    # ── Energy / glycolysis ──────────────────────────────────────────────
    "glucose": {
        "role": "Primary circulating sugar and energy source; tightly regulated by insulin.",
        "disease_associations": "↑ fasting glucose is the defining biomarker of diabetes mellitus.",
        "pathways": ["Glycolysis / Gluconeogenesis"],
        "category": "Energy / glycolysis", "direction": "up",
    },
    "lactate": {
        "role": "End product of anaerobic glycolysis; key energy shuttle (Cori cycle).",
        "disease_associations": "↑ in hypoxia, sepsis, mitochondrial dysfunction and insulin resistance.",
        "pathways": ["Glycolysis / Gluconeogenesis", "Pyruvate metabolism"],
        "category": "Energy / glycolysis", "direction": "up",
    },
    "pyruvate": {
        "role": "End product of glycolysis; gateway to TCA cycle, lactate and alanine.",
        "disease_associations": "↑ with impaired oxidative metabolism and insulin resistance.",
        "pathways": ["Glycolysis / Gluconeogenesis", "Pyruvate metabolism", "Citric acid cycle (TCA cycle)"],
        "category": "Energy / glycolysis", "direction": "up",
    },
    "glycerol": {
        "role": "Backbone of triglycerides; released during lipolysis (fat breakdown).",
        "disease_associations": "↑ in fasting, uncontrolled diabetes and increased lipolysis.",
        "pathways": ["Glycerolipid metabolism", "Glycolysis / Gluconeogenesis"],
        "category": "Energy / glycolysis", "direction": "up",
    },
    # ── Ketone bodies ────────────────────────────────────────────────────
    "3-hydroxybutyrate": {
        "role": "Primary ketone body; alternative brain/heart fuel during fasting or low insulin.",
        "disease_associations": "↑ in diabetic ketoacidosis, prolonged fasting, type-1/2 diabetes.",
        "pathways": ["Ketone body metabolism", "Butanoate metabolism"],
        "category": "Ketone bodies", "direction": "up",
    },
    "acetoacetate": {
        "role": "Ketone body produced in the liver from fatty-acid breakdown.",
        "disease_associations": "↑ in diabetic ketoacidosis and starvation ketosis.",
        "pathways": ["Ketone body metabolism", "Butanoate metabolism"],
        "category": "Ketone bodies", "direction": "up",
    },
    "acetone": {
        "role": "Volatile ketone body formed by spontaneous decarboxylation of acetoacetate.",
        "disease_associations": "↑ in ketosis, uncontrolled diabetes and prolonged fasting.",
        "pathways": ["Ketone body metabolism"],
        "category": "Ketone bodies", "direction": "up",
    },
    "acetate": {
        "role": "Short-chain fatty acid; product of gut microbial fermentation and ethanol metabolism.",
        "disease_associations": "↑ with alcohol intake and altered gut microbiome; energy substrate.",
        "pathways": ["Pyruvate metabolism", "Ketone body metabolism"],
        "category": "Gut microbiome", "direction": "up",
    },
    # ── TCA cycle ────────────────────────────────────────────────────────
    "citrate": {
        "role": "Central TCA-cycle intermediate; links carbohydrate, fat and energy metabolism.",
        "disease_associations": "Altered in mitochondrial dysfunction, renal stone risk, metabolic disease.",
        "pathways": ["Citric acid cycle (TCA cycle)"],
        "category": "TCA cycle", "direction": "either",
    },
    "succinate": {
        "role": "TCA-cycle intermediate; also an inflammatory and hypoxia signalling molecule.",
        "disease_associations": "↑ in inflammation, hypoxia and mitochondrial dysfunction.",
        "pathways": ["Citric acid cycle (TCA cycle)"],
        "category": "TCA cycle", "direction": "up",
    },
    # ── Amino acids — BCAA ───────────────────────────────────────────────
    "valine": {
        "role": "Branched-chain amino acid (BCAA); essential, used for muscle protein and energy.",
        "disease_associations": "↑ BCAAs strongly associated with insulin resistance and type-2 diabetes risk.",
        "pathways": ["Valine, leucine and isoleucine degradation", "BCAA metabolism"],
        "category": "Amino acid — BCAA", "direction": "up",
    },
    "leucine": {
        "role": "Branched-chain amino acid (BCAA); potent activator of mTOR / muscle protein synthesis.",
        "disease_associations": "↑ BCAAs predict insulin resistance and future type-2 diabetes.",
        "pathways": ["Valine, leucine and isoleucine degradation", "BCAA metabolism"],
        "category": "Amino acid — BCAA", "direction": "up",
    },
    "isoleucine": {
        "role": "Branched-chain amino acid (BCAA); essential, glucogenic and ketogenic.",
        "disease_associations": "↑ BCAAs associated with insulin resistance and metabolic syndrome.",
        "pathways": ["Valine, leucine and isoleucine degradation", "BCAA metabolism"],
        "category": "Amino acid — BCAA", "direction": "up",
    },
    # ── Amino acids — aromatic ───────────────────────────────────────────
    "phenylalanine": {
        "role": "Essential aromatic amino acid; precursor of tyrosine and catecholamines.",
        "disease_associations": "↑ in insulin resistance, type-2 diabetes and (extreme) in phenylketonuria.",
        "pathways": ["Phenylalanine, tyrosine and tryptophan biosynthesis", "Aromatic amino acid metabolism"],
        "category": "Amino acid — aromatic", "direction": "up",
    },
    "tyrosine": {
        "role": "Aromatic amino acid; precursor of dopamine, adrenaline and thyroid hormones.",
        "disease_associations": "↑ tyrosine associated with insulin resistance and type-2 diabetes risk.",
        "pathways": ["Phenylalanine, tyrosine and tryptophan biosynthesis", "Aromatic amino acid metabolism"],
        "category": "Amino acid — aromatic", "direction": "up",
    },
    "tryptophan": {
        "role": "Essential aromatic amino acid; precursor of serotonin, melatonin and NAD+.",
        "disease_associations": "↓ in inflammation and depression; shifts with gut-microbiome kynurenine pathway.",
        "pathways": ["Phenylalanine, tyrosine and tryptophan biosynthesis", "Tryptophan metabolism"],
        "category": "Amino acid — aromatic", "direction": "down",
    },
    "histidine": {
        "role": "Essential amino acid; precursor of histamine, antioxidant and metal-chelating roles.",
        "disease_associations": "↓ histidine linked to inflammation, oxidative stress and chronic kidney disease.",
        "pathways": ["Histidine metabolism"],
        "category": "Amino acid — aromatic", "direction": "down",
    },
    # ── Amino acids — other ──────────────────────────────────────────────
    "alanine": {
        "role": "Glucogenic amino acid; carries nitrogen from muscle to liver (glucose–alanine cycle).",
        "disease_associations": "↑ in insulin resistance and type-2 diabetes; reflects protein catabolism.",
        "pathways": ["Alanine, aspartate and glutamate metabolism", "Glucose–alanine cycle"],
        "category": "Amino acid — other", "direction": "up",
    },
    "glutamine": {
        "role": "Most abundant blood amino acid; major nitrogen and carbon carrier between tissues.",
        "disease_associations": "Glutamine/glutamate ratio shifts in insulin resistance and type-2 diabetes.",
        "pathways": ["Alanine, aspartate and glutamate metabolism", "Nitrogen metabolism"],
        "category": "Amino acid — other", "direction": "either",
    },
    "glutamate": {
        "role": "Central excitatory neurotransmitter and nitrogen hub for transamination.",
        "disease_associations": "↑ glutamate / low glutamine ratio associated with insulin resistance.",
        "pathways": ["Alanine, aspartate and glutamate metabolism", "Nitrogen metabolism"],
        "category": "Amino acid — other", "direction": "up",
    },
    "glycine": {
        "role": "Simplest amino acid; one-carbon metabolism, glutathione and heme synthesis.",
        "disease_associations": "↓ glycine associated with insulin resistance, obesity and type-2 diabetes.",
        "pathways": ["Glycine, serine and threonine metabolism", "One-carbon metabolism"],
        "category": "Choline / one-carbon", "direction": "down",
    },
    "serine": {
        "role": "Non-essential amino acid; one-carbon donor for nucleotide and glutathione synthesis.",
        "disease_associations": "↓ in type-2 diabetes and impaired one-carbon metabolism.",
        "pathways": ["Glycine, serine and threonine metabolism", "One-carbon metabolism"],
        "category": "Choline / one-carbon", "direction": "down",
    },
    "threonine": {
        "role": "Essential amino acid; mucin synthesis and one-carbon contributions.",
        "disease_associations": "Altered in protein malnutrition and gut-barrier dysfunction.",
        "pathways": ["Glycine, serine and threonine metabolism"],
        "category": "Amino acid — other", "direction": "either",
    },
    "asparagine": {
        "role": "Non-essential amino acid; amide nitrogen storage and transport.",
        "disease_associations": "Altered in metabolic stress and some cancers.",
        "pathways": ["Alanine, aspartate and glutamate metabolism"],
        "category": "Amino acid — other", "direction": "either",
    },
    "aspartate": {
        "role": "Amino acid feeding the TCA cycle (oxaloacetate) and nucleotide synthesis.",
        "disease_associations": "Shifts with mitochondrial activity and nitrogen balance.",
        "pathways": ["Alanine, aspartate and glutamate metabolism", "Citric acid cycle (TCA cycle)"],
        "category": "Amino acid — other", "direction": "either",
    },
    "lysine": {
        "role": "Essential amino acid; collagen cross-linking and carnitine synthesis.",
        "disease_associations": "Altered in protein turnover disorders and malnutrition.",
        "pathways": ["Lysine degradation"],
        "category": "Amino acid — other", "direction": "either",
    },
    "methionine": {
        "role": "Essential sulfur amino acid; methyl donor (SAM) for one-carbon metabolism.",
        "disease_associations": "Altered in oxidative stress, cardiovascular and liver disease.",
        "pathways": ["Cysteine and methionine metabolism", "One-carbon metabolism"],
        "category": "Choline / one-carbon", "direction": "either",
    },
    "proline": {
        "role": "Cyclic amino acid; collagen structure and redox balance.",
        "disease_associations": "↑ in some metabolic and fibrotic conditions.",
        "pathways": ["Arginine and proline metabolism"],
        "category": "Amino acid — other", "direction": "either",
    },
    "arginine": {
        "role": "Conditionally essential amino acid; nitric-oxide and urea-cycle precursor.",
        "disease_associations": "Altered in cardiovascular disease and endothelial dysfunction.",
        "pathways": ["Arginine and proline metabolism", "Nitrogen metabolism"],
        "category": "Amino acid — other", "direction": "either",
    },
    "ornithine": {
        "role": "Urea-cycle intermediate; polyamine synthesis precursor.",
        "disease_associations": "Altered in urea-cycle disorders and liver dysfunction.",
        "pathways": ["Arginine and proline metabolism", "Nitrogen metabolism"],
        "category": "Amino acid — other", "direction": "either",
    },
    # ── Kidney / muscle ──────────────────────────────────────────────────
    "creatinine": {
        "role": "Breakdown product of muscle creatine; excreted by the kidney at a steady rate.",
        "disease_associations": "↑ in reduced kidney function / diabetic nephropathy; used to gauge renal clearance.",
        "pathways": ["Arginine and proline metabolism", "Creatine metabolism"],
        "category": "Kidney / muscle", "direction": "up",
    },
    "creatine": {
        "role": "Energy buffer in muscle and brain (phosphocreatine system).",
        "disease_associations": "Altered in muscle wasting, renal disease and creatine-deficiency syndromes.",
        "pathways": ["Arginine and proline metabolism", "Creatine metabolism"],
        "category": "Kidney / muscle", "direction": "either",
    },
    "hypoxanthine": {
        "role": "Purine-degradation intermediate; marker of ATP turnover and tissue hypoxia.",
        "disease_associations": "↑ in hypoxia, ischemia, oxidative stress and gout-related purine turnover.",
        "pathways": ["Purine metabolism"],
        "category": "Purine metabolism", "direction": "up",
    },
    # ── Choline / one-carbon ─────────────────────────────────────────────
    "choline": {
        "role": "Essential nutrient; membrane phospholipids, acetylcholine and methyl metabolism.",
        "disease_associations": "Altered in fatty liver, cardiovascular risk and neurological conditions.",
        "pathways": ["Glycerophospholipid metabolism", "One-carbon metabolism"],
        "category": "Choline / one-carbon", "direction": "either",
    },
    "betaine": {
        "role": "Methyl donor derived from choline; osmolyte protecting cells from stress.",
        "disease_associations": "↓ associated with metabolic syndrome and cardiovascular risk.",
        "pathways": ["Glycine, serine and threonine metabolism", "One-carbon metabolism"],
        "category": "Choline / one-carbon", "direction": "down",
    },
    "dimethylglycine": {
        "role": "Intermediate of choline/betaine methyl metabolism.",
        "disease_associations": "Shifts with one-carbon metabolism and folate status.",
        "pathways": ["Glycine, serine and threonine metabolism", "One-carbon metabolism"],
        "category": "Choline / one-carbon", "direction": "either",
    },
    # ── Gut microbiome ───────────────────────────────────────────────────
    "trimethylamine n-oxide": {
        "role": "Gut-microbiome metabolite (TMAO) from dietary choline/carnitine, oxidised in liver.",
        "disease_associations": "↑ TMAO strongly linked to cardiovascular disease and atherosclerosis risk.",
        "pathways": ["Gut microbial metabolism", "Glycerophospholipid metabolism"],
        "category": "Gut microbiome", "direction": "up",
    },
    "trimethylamine": {
        "role": "Gut-microbiome product of dietary choline/carnitine; precursor of TMAO.",
        "disease_associations": "↑ with gut dysbiosis; linked to cardiovascular risk via TMAO.",
        "pathways": ["Gut microbial metabolism"],
        "category": "Gut microbiome", "direction": "up",
    },
    "dimethylamine": {
        "role": "Amine from gut-microbial and endogenous metabolism.",
        "disease_associations": "↑ in chronic kidney disease (impaired clearance).",
        "pathways": ["Gut microbial metabolism", "Nitrogen metabolism"],
        "category": "Gut microbiome", "direction": "up",
    },
    "methylamine": {
        "role": "Small amine from gut-microbial and dietary amine metabolism.",
        "disease_associations": "Altered with gut dysbiosis and renal clearance.",
        "pathways": ["Gut microbial metabolism", "Nitrogen metabolism"],
        "category": "Gut microbiome", "direction": "either",
    },
    "formate": {
        "role": "One-carbon unit; product of gut-microbial and one-carbon metabolism.",
        "disease_associations": "Shifts with gut microbiome and one-carbon flux.",
        "pathways": ["One-carbon metabolism", "Gut microbial metabolism"],
        "category": "Gut microbiome", "direction": "either",
    },
    "succinate-gut": {  # alias guard; real entry is 'succinate' above
        "role": "", "disease_associations": "", "pathways": [],
        "category": "Gut microbiome", "direction": "either",
    },
    # ── Lipid / membrane ─────────────────────────────────────────────────
    "carnitine": {
        "role": "Shuttles long-chain fatty acids into mitochondria for β-oxidation.",
        "disease_associations": "Altered in fatty-acid-oxidation disorders and insulin resistance.",
        "pathways": ["Fatty acid oxidation", "Lysine degradation"],
        "category": "Lipid / membrane", "direction": "either",
    },
    "myo-inositol": {
        "role": "Sugar alcohol; membrane signalling (phosphoinositides) and osmolyte.",
        "disease_associations": "↑ in diabetes and altered in kidney disease; insulin-signalling roles.",
        "pathways": ["Inositol phosphate metabolism"],
        "category": "Lipid / membrane", "direction": "up",
    },
    "taurine": {
        "role": "Sulfur amino acid; bile-salt conjugation, osmoregulation, antioxidant.",
        "disease_associations": "Altered in cardiovascular and metabolic disease.",
        "pathways": ["Taurine and hypotaurine metabolism"],
        "category": "Lipid / membrane", "direction": "either",
    },
    # ── Exogenous / xenobiotic ───────────────────────────────────────────
    "isopropanol": {
        "role": "Exogenous solvent / disinfectant; not endogenously produced.",
        "disease_associations": "Detected after isopropanol exposure; usually a sample-handling artifact.",
        "pathways": ["Xenobiotic / exogenous"],
        "category": "Exogenous / xenobiotic", "direction": "either",
    },
    "methanol": {
        "role": "Exogenous alcohol; trace dietary/microbial origin.",
        "disease_associations": "Mostly exogenous; elevated after methanol exposure.",
        "pathways": ["Xenobiotic / exogenous"],
        "category": "Exogenous / xenobiotic", "direction": "either",
    },
    "ethanol": {
        "role": "Exogenous alcohol; also trace gut-microbial fermentation product.",
        "disease_associations": "↑ with alcohol intake; relevant to liver disease.",
        "pathways": ["Xenobiotic / exogenous", "Gut microbial metabolism"],
        "category": "Exogenous / xenobiotic", "direction": "up",
    },
    "isopropyl alcohol": {  # alias of isopropanol
        "role": "Exogenous solvent / disinfectant; not endogenously produced.",
        "disease_associations": "Detected after isopropanol exposure; usually a sample-handling artifact.",
        "pathways": ["Xenobiotic / exogenous"],
        "category": "Exogenous / xenobiotic", "direction": "either",
    },
    "dimethyl sulfone": {
        "role": "Sulfur-containing compound from diet and gut microbial metabolism (MSM).",
        "disease_associations": "Mostly dietary; reflects sulfur metabolism and supplement intake.",
        "pathways": ["Sulfur / methionine metabolism"],
        "category": "Exogenous / xenobiotic", "direction": "either",
    },
}

# Drop alias guard placeholder (kept above only for readability).
METABOLITE_BIOLOGY.pop("succinate-gut", None)


# ── Option B: pathway → metabolite membership (for enrichment) ────────────────
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
    if not info or not info.get("role"):
        return None
    category = info.get("category", "")
    return {
        "role": info["role"],
        "disease_associations": info["disease_associations"],
        "pathways": list(info["pathways"]),
        "category": category,
        "category_color": CATEGORY_COLORS.get(category, "#3f6fb0"),
        "direction": info.get("direction", "either"),
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
    """
    bm = {normalize(b) for b in biomarkers if normalize(b)}
    if background:
        universe = {normalize(b) for b in background if normalize(b)}
        universe |= bm
    else:
        universe = set(METABOLITE_BIOLOGY.keys()) | bm

    N = len(universe)
    n = len(bm & universe)
    if N == 0 or n == 0:
        return []

    results = []
    for pathway, members in PATHWAY_INDEX.items():
        members_in_universe = members & universe
        K = len(members_in_universe)
        if K == 0:
            continue
        hits = bm & members_in_universe
        k = len(hits)
        if k < min_overlap:
            continue
        p_value = float(hypergeom.sf(k - 1, N, K, n))
        fold_enrichment = (k / n) / (K / N) if (n and K) else 0.0
        results.append({
            "pathway": pathway,
            "p_value": round(p_value, 5),
            "overlap": k,
            "pathway_size": K,
            "fold_enrichment": round(fold_enrichment, 2),
            "significant": p_value < 0.05,
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
    category_counts: Dict[str, int] = {}
    for name in biomarkers:
        bio = annotate(name)
        annotations.append({"metabolite": name, "biology": bio})
        if bio and bio.get("category"):
            category_counts[bio["category"]] = category_counts.get(bio["category"], 0) + 1

    enrichment = pathway_enrichment(biomarkers, background)
    n_annotated = sum(1 for a in annotations if a["biology"])
    n_significant = sum(1 for e in enrichment if e["significant"])

    return {
        "metabolite_biology": annotations,
        "pathway_enrichment": enrichment,
        "category_counts": category_counts,
        "coverage": {
            "annotated": n_annotated,
            "total": len(annotations),
        },
        "n_significant_pathways": n_significant,
        "top_pathway": enrichment[0]["pathway"] if enrichment else None,
        "top_pathway_significant": bool(enrichment and enrichment[0]["significant"]),
    }
