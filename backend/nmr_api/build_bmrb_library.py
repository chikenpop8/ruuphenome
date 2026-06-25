"""
Build a large, ACCURATE ¹H reference-shift library from the public BMRB
metabolomics database (real assigned chemical shifts — not fabricated).

Writes open_data/bmrb_reference_shifts.json: {metabolite_name: [¹H ppm, ...]}.
spectral_cohort loads + merges this at import (curated values take priority).

Usage (with internet, NOT on the locked VM):
    python -m nmr_api.build_bmrb_library --target 500
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import requests

H = {"Application": "RuuPhenome/0.2 open-data client"}
API = "https://api.bmrb.io/v2"
OUT = Path(__file__).resolve().parent / "open_data" / "bmrb_reference_shifts.json"


def list_entries() -> list[str]:
    r = requests.get(f"{API}/list_entries?database=metabolomics", timeout=30, headers=H)
    r.raise_for_status()
    return list(r.json())


def _category(eid: str, category: str) -> list:
    """Fetch one saveframe category for an entry (clean keyed format)."""
    r = requests.get(f"{API}/entry/{eid}?saveframe_category={category}",
                     timeout=25, headers=H)
    if r.status_code != 200:
        return []
    return r.json().get(eid, {}).get(category, [])


def fetch_h_shifts(eid: str) -> tuple[str | None, list[float]]:
    """Return (compound_name, sorted unique ¹H shifts) for one BMRB entry."""
    name = None
    for sf in _category(eid, "entity"):
        for k, v in sf.get("tags", []):
            if k == "Name" and v and v != ".":
                name = v
    shifts: list[float] = []
    for sf in _category(eid, "assigned_chemical_shifts"):
        for lp in sf.get("loops", []):
            tags = lp.get("tags", [])
            ti = {t: i for i, t in enumerate(tags)}
            ecol = next((i for t, i in ti.items() if t.endswith("Atom_type")), None)
            vcol = next((i for t, i in ti.items() if t.endswith(".Val") or t == "Val"), None)
            if ecol is None or vcol is None:
                continue
            for row in lp.get("data", []):
                if row[ecol] == "H":
                    try:
                        v = float(row[vcol])
                        if 0.0 <= v <= 12.0:
                            shifts.append(round(v, 3))
                    except (ValueError, TypeError):
                        pass
    # dedup shifts that are within 0.02 ppm (same resonance)
    shifts.sort()
    dedup: list[float] = []
    for s in shifts:
        if not dedup or abs(s - dedup[-1]) > 0.02:
            dedup.append(s)
    return name, dedup


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=500, help="unique compounds wanted")
    ap.add_argument("--scan", type=int, default=900, help="max entries to scan")
    args = ap.parse_args()

    entries = list_entries()[: args.scan]
    print(f"scanning up to {len(entries)} BMRB entries for {args.target} compounds…")
    library: dict[str, list[float]] = {}
    t0 = time.time()
    for i, eid in enumerate(entries):
        if len(library) >= args.target:
            break
        try:
            name, shifts = fetch_h_shifts(eid)
        except Exception:
            continue
        if name and len(shifts) >= 1:
            key = name.strip().lower()
            if key not in library:                 # first (dedup by name)
                library[key] = shifts
        if (i + 1) % 50 == 0:
            print(f"  {i+1} scanned, {len(library)} compounds, {round(time.time()-t0)}s")

    OUT.write_text(json.dumps(library, indent=0, sort_keys=True))
    print(f"DONE: {len(library)} compounds → {OUT} ({round(time.time()-t0)}s)")


if __name__ == "__main__":
    main()
