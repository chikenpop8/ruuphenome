"""
Reference-card enrichment via PubChem PUG REST.

For each metabolite, fetches real metadata so the Reference Card is fully
populated and the external-database links resolve to the correct pages:

  - IUPAC name
  - alternate names / synonyms
  - CAS registry number
  - PubChem CID  → direct PubChem link
  - InChIKey     → robust cross-database lookups

Results are cached to ``cache/pubchem_cache.json`` (keyed by compound name) so
the first build hits the network once and every later load is instant and works
offline. All network use is best-effort: if PubChem is unreachable, the card
falls back to name-based search links and the app keeps working.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path
from typing import Dict, List, Optional


def offline_mode() -> bool:
    """True when NMR_OFFLINE is set — blocks ALL outbound network calls so no
    data can leave the host (required when the data owner forbids external
    processing). Cached enrichment still works; uncached compounds use
    name-based fallback links."""
    return os.environ.get("NMR_OFFLINE", "").strip().lower() in ("1", "true", "yes")

try:
    import requests
    _HAVE_REQUESTS = True
except Exception:  # pragma: no cover
    _HAVE_REQUESTS = False

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
CACHE_PATH = Path(__file__).resolve().parent / "cache" / "pubchem_cache.json"
_CAS_RE = re.compile(r"^\d{2,7}-\d{2}-\d$")
_LOCK = threading.Lock()

# in-memory cache, loaded from disk once
_CACHE: Optional[Dict[str, Dict]] = None


def _load_cache() -> Dict[str, Dict]:
    global _CACHE
    if _CACHE is None:
        try:
            _CACHE = json.loads(CACHE_PATH.read_text())
        except Exception:
            _CACHE = {}
    return _CACHE


def _save_cache() -> None:
    try:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        CACHE_PATH.write_text(json.dumps(_CACHE, indent=2))
    except Exception:
        pass


def _get(url: str) -> Optional[dict]:
    if offline_mode() or not _HAVE_REQUESTS:
        return None                      # offline → no outbound connection at all
    try:
        r = requests.get(url, timeout=12)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None


def _fetch_pubchem(name: str) -> Dict:
    """Best-effort PubChem lookup by compound name."""
    out: Dict = {"pubchem_cid": None, "iupac_name": None, "inchikey": None,
                 "synonyms": [], "cas": None}
    props = _get(
        f"{PUG}/compound/name/{requests.utils.quote(name)}"
        f"/property/IUPACName,InChIKey/JSON"
    ) if _HAVE_REQUESTS else None
    if not props:
        return out
    try:
        p = props["PropertyTable"]["Properties"][0]
        out["pubchem_cid"] = p.get("CID")
        out["iupac_name"] = p.get("IUPACName")
        out["inchikey"] = p.get("InChIKey")
    except Exception:
        return out

    cid = out["pubchem_cid"]
    if cid:
        syn = _get(f"{PUG}/compound/cid/{cid}/synonyms/JSON")
        try:
            syns = syn["InformationList"]["Information"][0]["Synonym"]
            cas = next((s for s in syns if _CAS_RE.match(s)), None)
            # human-readable alternate names: skip pure IDs / CAS / codes
            alt = [s for s in syns
                   if not _CAS_RE.match(s)
                   and not s.upper().startswith(("CHEBI", "HMDB", "CHEMBL", "SCHEMBL"))
                   and not re.fullmatch(r"[A-Z0-9\-]{8,}", s)][:6]
            out["cas"] = cas
            out["synonyms"] = alt
        except Exception:
            pass
    return out


def enrich(name: str, chebi_id: str = "", inchi: str = "") -> Dict:
    """
    Return enriched metadata + correct external links for one metabolite,
    using the on-disk cache when available.
    """
    cache = _load_cache()
    if name in cache:
        data = cache[name]
    else:
        data = _fetch_pubchem(name)
        with _LOCK:
            cache[name] = data
            _save_cache()

    return {
        "iupac_name": data.get("iupac_name"),
        "alternate_names": data.get("synonyms", []),
        "cas_registry": data.get("cas"),
        "pubchem_cid": data.get("pubchem_cid"),
        "inchikey": data.get("inchikey"),
        "external_refs": _links(name, chebi_id, data),
    }


def _links(name: str, chebi_id: str, data: Dict) -> Dict[str, Optional[str]]:
    """Build the best resolving link for each database."""
    cid = data.get("pubchem_cid")
    inchikey = data.get("inchikey")
    chebi_num = chebi_id.split(":", 1)[1] if chebi_id.upper().startswith("CHEBI:") else ""
    q = requests.utils.quote(name) if _HAVE_REQUESTS else name.replace(" ", "+")

    return {
        # direct compound page when we have a CID, else InChIKey/name search
        "pubchem": (f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}" if cid
                    else f"https://pubchem.ncbi.nlm.nih.gov/#query={inchikey or q}"),
        # ChEBI direct page from the ID we already have
        "chebi": (f"https://www.ebi.ac.uk/chebi/searchId.do?chebiId=CHEBI:{chebi_num}"
                  if chebi_num else None),
        # HMDB: InChIKey search resolves uniquely; fall back to name
        "hmdb": f"https://hmdb.ca/unearth/q?utf8=%E2%9C%93&query={inchikey or q}"
                f"&searcher=metabolites&button=",
        # KEGG: text search by name (no KEGG id available from PubChem name lookup)
        "kegg": f"https://www.genome.jp/dbget-bin/www_bfind_sub?mode=bfind&max_hit=15"
                f"&dbkey=compound&keywords={q}",
    }
