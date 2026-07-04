"""
External open ¹H-NMR reference data for HELD-OUT identification validation (RUO).

Fetches physically-exact ¹H chemical shifts from **GISSMO** (gissmo.bmrb.io) — spin
systems fit to experimental BMRB metabolomics spectra — to build *real* test
spectra whose peak positions come from an INDEPENDENT source, not our HMDB-derived
reference library. Matching our library against these real shifts is an honest
held-out test of robustness to real experimental shift variation. Cached to disk;
open data only (never the closed dataset); network-using by design.

`GISSMO_IDS` were verified by fetching each entry's `<name>` (stereo-prefix- and
acid/ate-normalised exact match).
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

CACHE = Path(tempfile.gettempdir()) / "ruuphenome_gissmo"
_XML = "https://gissmo.bmrb.io/entry/{eid}/simulation_1/spin_simulation.xml"

# compound (canonical) → GISSMO/BMRB entry id (verified via <name>).
GISSMO_IDS: Dict[str, str] = {
    "glucose": "bmse000015",       # (+/-)-Glucose
    "alanine": "bmse000028",       # L-alanine
    "valine": "bmse000052",        # L-valine  (BCAA)
    "leucine": "bmse000042",       # L-leucine  (BCAA)
    "isoleucine": "bmse000041",    # L-isoleucine  (BCAA)
    "glutamine": "bmse000038",     # L-glutamine
    "glutamate": "bmse000037",     # L-glutamic-acid
    "citrate": "bmse000076",       # Citrate
    "tyrosine": "bmse000051",      # L-tyrosine  (aromatic AA)
    "phenylalanine": "bmse000045", # L-phenylalanine  (aromatic AA)
    "histidine": "bmse000039",     # L-histidine
    "threonine": "bmse000049",     # L-threonine
    "serine": "bmse000048",        # L-serine
    "glycine": "bmse000089",       # Glycine
    "proline": "bmse000047",       # L-proline
    "lysine": "bmse000043",        # L-lysine
    "betaine": "bmse000069",       # Betaine
}  # verified by fetching each entry's <name> (stereo-/acid-normalised exact match)


def _curl(url: str, timeout: int = 20) -> str:
    try:
        return subprocess.run(["curl", "-sSL", "--max-time", str(timeout), url],
                              capture_output=True, text=True, timeout=timeout + 3).stdout
    except Exception:
        return ""


def fetch_shifts(entry_id: str) -> Dict:
    """{'id','name','shifts'} for a GISSMO entry (cached)."""
    cache = CACHE / f"{entry_id}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    xml = _curl(_XML.format(eid=entry_id))
    name_m = re.search(r"<name>([^<\n]+)", xml)
    name = name_m.group(1).strip() if name_m else entry_id
    shifts = [round(float(p), 4) for p in re.findall(r'ppm="([-\d.]+)"', xml)]
    d = {"id": entry_id, "name": name, "shifts": shifts}
    if shifts:
        CACHE.mkdir(exist_ok=True)
        cache.write_text(json.dumps(d))
    return d


def real_shifts(compounds: Optional[List[str]] = None) -> Dict[str, List[float]]:
    """{compound: real GISSMO ¹H shifts} for the verified panel (cached). Silently
    skips any that fail to fetch, so it degrades gracefully offline."""
    out: Dict[str, List[float]] = {}
    for name, eid in GISSMO_IDS.items():
        if compounds is not None and name not in compounds:
            continue
        try:
            d = fetch_shifts(eid)
            if d.get("shifts"):
                out[name] = d["shifts"]
        except Exception:
            continue
    return out
