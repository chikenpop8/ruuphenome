"""
BMRB metabolomics EXPERIMENTAL ¹H peak lists — an independent, real, held-out
identification source for Track-1 validation (RUO, open data).

Unlike our HMDB-derived reference library (idealised centroid shifts) and the
GISSMO *simulated* shifts, this fetches the **experimentally measured** 1D ¹H peak
list (real peak positions AND relative intensities AND multiplet splitting, on a
real spectrometer, DSS-referenced) that BMRB deposits alongside each metabolomics
entry. Test spectra rendered from these peaks stress identification with three
things our library never encoded: real intensities, real doublet/triplet
structure, and real measurement noise/referencing.

**Honest independence caveat.** BMRB metabolomics reference shifts partly feed
HMDB, so the peak *positions* are shift-correlated with our library — this is NOT a
blind-shift source. It IS independent in *provenance, intensity, multiplet
structure and lineshape*, which is what makes it a real held-out test of the
matcher/classifier beyond centroid ppm.

**Governance:** fetched OFF-VM (`build_bundle`, network) → bundled to
`open_data/bmrb_experimental_peaks.json`; loaded with no network at test/serve
time. Open data only; the closed dataset is never used.

Source: BMRB (bmrb.io), NIH/NIGMS-funded public resource; cite BMRB + the
originating Metabolomics Consortium deposition.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence

try:
    from . import external_reference as ext
except ImportError:  # pragma: no cover - direct execution
    import external_reference as ext  # type: ignore

CACHE = Path("/tmp/ruuphenome_bmrb")
BUNDLE_PATH = Path(__file__).resolve().parent / "open_data" / "bmrb_experimental_peaks.json"
_BASE = "https://bmrb.io/ftp/pub/bmrb/metabolomics/entry_directories/{eid}/nmr/set01"
_LIST_URL = _BASE + "/transitions/1H.list"
_ACQUS_URL = _BASE + "/{d}/acqus"
_XML_URL = _BASE + "/{d}/pdata/1/peaklist.xml"
_PROCS_URL = _BASE + "/{d}/pdata/1/procs"
_1R_URL = _BASE + "/{d}/pdata/1/1r"
_WATER = (4.70, 4.90)          # residual HDO — never a real metabolite peak
# 2D / heteronuclear / edited sequences whose peak list is NOT a plain 1D ¹H pick.
_NON_1D1H = re.compile(r"tocsy|cosy|hsqc|hmbc|mlev|dipsi|roesy|jres|dept|hmqc|inept|13c", re.I)

# library-compound → BMRB metabolomics entry id (bmse), name-verified.
# The GISSMO panel entries ARE BMRB metabolomics entries, so their experimental
# ¹H peak lists live at the set-level transitions/1H.list path. `lactate` is added
# (probe-verified) since it is the canonical NCD diagnostic resonance.
BMRB_IDS: Dict[str, str] = dict(ext.GISSMO_IDS)          # 17 verified entries
BMRB_IDS.setdefault("lactate", "bmse000269")             # (R)-lactate, experimental peak.txt verified


def _curl(url: str, timeout: int = 25) -> str:
    try:
        return subprocess.run(["curl", "-sSL", "--max-time", str(timeout), url],
                              capture_output=True, text=True, timeout=timeout + 3).stdout
    except Exception:
        return ""


def _curl_bytes(url: str, timeout: int = 40) -> bytes:
    try:
        return subprocess.run(["curl", "-sSL", "--max-time", str(timeout), url],
                              capture_output=True, timeout=timeout + 3).stdout
    except Exception:
        return b""


def _parse_list(text: str, lo: float = 0.4, hi: float = 10.0) -> List[List[float]]:
    """Parse a BMRB `transitions/1H.list` peak file → [[ppm, intensity], ...].

    Data rows look like `idx address freq_Hz ppm intensity`; the ppm is the
    second-to-last numeric column and the intensity the last, which is robust to
    the 4- vs 5-column variants seen across entries. HTML (404) bodies parse to []."""
    if "<html" in text.lower() or "<!doctype" in text.lower():
        return []
    peaks: List[List[float]] = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            idx = int(parts[0])            # data rows start with an integer index
        except ValueError:
            continue
        try:
            ppm = float(parts[-2])
            inten = float(parts[-1])
        except ValueError:
            continue
        if lo <= ppm <= hi and inten > 0:
            peaks.append([round(ppm, 4), round(inten, 4)])
    return peaks


def _parse_xml(text: str, lo: float = 0.4, hi: float = 10.0) -> List[List[float]]:
    """Parse a Bruker `peaklist.xml` (`<Peak1D F1="ppm" intensity="..."/>`) →
    [[ppm, intensity], ...]. F1 is the ¹H chemical shift."""
    if "<Peak1D" not in text:
        return []
    peaks: List[List[float]] = []
    for m in re.finditer(r'<Peak1D\b[^>]*\bF1="([-\d.]+)"[^>]*\bintensity="([-\d.]+)"', text):
        ppm, inten = float(m.group(1)), float(m.group(2))
        if lo <= ppm <= hi and inten > 0:
            peaks.append([round(ppm, 4), round(inten, 4)])
    return peaks


def _find_1d1h_dir(eid: str, max_dir: int = 9) -> Optional[int]:
    """Locate the plain 1D ¹H experiment dir: NUC1==1H and a non-2D/edited pulse
    program (glucose's is a numbered dir, e.g. `1` with PULPROG zgcppr)."""
    for d in range(1, max_dir + 1):
        acqus = _curl(_ACQUS_URL.format(eid=eid, d=d), timeout=15)
        if not acqus or "##$" not in acqus:
            if d >= 3:                        # ran past the populated dirs
                break
            continue
        nuc1 = re.search(r"##\$NUC1=\s*<?([^\s>]+)", acqus)
        pul = re.search(r"##\$PULPROG=\s*<?([^\s>]+)", acqus)
        pulprog = (pul.group(1) if pul else "")
        if nuc1 and "1H" in nuc1.group(1) and not _NON_1D1H.search(pulprog):
            return d
    return None


def _procs(text: str, key: str) -> Optional[float]:
    m = re.search(r"##\$" + key + r"=\s*([-\d.]+)", text)
    return float(m.group(1)) if m else None


def reconstruct_spectrum(eid: str, d: int):
    """Rebuild the REAL processed 1D ¹H spectrum from Bruker `1r`+`procs`
    → (ppm, intensity) numpy arrays over [0.4, 10] ppm. None on failure."""
    import struct
    import numpy as np
    procs = _curl(_PROCS_URL.format(eid=eid, d=d))
    SF, OFFSET, SW_p = _procs(procs, "SF"), _procs(procs, "OFFSET"), _procs(procs, "SW_p")
    SI, NC, BORD = _procs(procs, "SI"), _procs(procs, "NC_proc"), _procs(procs, "BYTORDP")
    if None in (SF, OFFSET, SW_p, SI) or SF <= 0:
        return None
    SI = int(SI)
    raw = _curl_bytes(_1R_URL.format(eid=eid, d=d))
    if len(raw) < 4 * SI:
        return None
    fmt = ("<" if int(BORD or 0) == 0 else ">") + f"{SI}i"
    ints = struct.unpack(fmt, raw[:4 * SI])
    y = np.asarray(ints, dtype=np.float64) * (2.0 ** (NC if NC is not None else 0))
    sw_ppm = SW_p / SF
    ppm = OFFSET - np.arange(SI) * sw_ppm / (SI - 1)     # descending from OFFSET
    keep = (ppm >= 0.4) & (ppm <= 10.0)
    return ppm[keep], np.clip(y[keep], 0.0, None)


def _pick_peaks(ppm, y, *, max_peaks: int = 40, min_frac: float = 0.02) -> List[List[float]]:
    """Local maxima of a real spectrum above a noise floor, water excluded → the
    strongest `max_peaks` as [[ppm, intensity], ...] (intensity relative to max)."""
    import numpy as np
    ppm = np.asarray(ppm, float); y = np.asarray(y, float)
    if y.size < 3 or float(y.max()) <= 0:
        return []
    ymax = float(y.max())
    med = float(np.median(y)); mad = float(np.median(np.abs(y - med))) or 1e-9
    thr = max(med + 6.0 * mad, min_frac * ymax)          # noise floor
    lo, hi = _WATER
    hits = []
    for i in range(1, y.size - 1):
        if y[i] >= y[i - 1] and y[i] > y[i + 1] and y[i] >= thr and not (lo <= ppm[i] <= hi):
            hits.append((float(ppm[i]), float(y[i]) / ymax))
    hits.sort(key=lambda t: t[1], reverse=True)
    return [[round(p, 4), round(v, 4)] for p, v in sorted(hits[:max_peaks])]


def fetch_peaklist(eid: str) -> Dict:
    """{'id', 'source', 'peaks': [[ppm, intensity], ...]} for a BMRB entry (cached).
    Tries, in order: the raw 1D ¹H `peaklist.xml`, the set-level assigned
    `transitions/1H.list`, and finally peak-picking the reconstructed real `1r`
    spectrum (covers entries that ship only the binary spectrum, e.g. valine)."""
    cache = CACHE / f"{eid}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text())
        except Exception:
            pass
    peaks, source = [], ""
    d = _find_1d1h_dir(eid)
    if d is not None:
        peaks = _parse_xml(_curl(_XML_URL.format(eid=eid, d=d)))
        source = f"set01/{d}/pdata/1/peaklist.xml"
    if len(peaks) < 2:                         # fallback 1: assigned transitions
        alt = _parse_list(_curl(_LIST_URL.format(eid=eid)))
        if len(alt) >= 2:
            peaks, source = alt, "set01/transitions/1H.list"
    if len(peaks) < 2 and d is not None:       # fallback 2: pick from real 1r spectrum
        spec = reconstruct_spectrum(eid, d)
        if spec is not None:
            picked = _pick_peaks(*spec)
            if len(picked) >= 2:
                peaks, source = picked, f"set01/{d}/pdata/1/1r (peak-picked)"
    rec = {"id": eid, "source": source, "peaks": peaks}
    if peaks:
        CACHE.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(rec))
    return rec


def experimental_peaks(compounds: Optional[Sequence[str]] = None) -> Dict[str, List[List[float]]]:
    """{compound: [[ppm, intensity], ...]} of MEASURED peaks (fetched + cached).
    Silently skips any that fail to fetch, so it degrades gracefully offline."""
    out: Dict[str, List[List[float]]] = {}
    for name, eid in BMRB_IDS.items():
        if compounds is not None and name not in compounds:
            continue
        try:
            d = fetch_peaklist(eid)
            if len(d.get("peaks", [])) >= 1:      # singlets (glycine, acetate) are valid
                out[name] = d["peaks"]
        except Exception:
            continue
    return out


def build_bundle(out_path: Path = BUNDLE_PATH) -> Dict:
    """OFF-VM: fetch every BMRB experimental peak list and bundle it to JSON so the
    validation runs with no network at test/serve time. Run on a networked machine
    and commit the JSON."""
    peaks = experimental_peaks()
    recs = {}
    for name, pk in peaks.items():
        eid = BMRB_IDS[name]
        src = ""
        try:
            src = fetch_peaklist(eid).get("source", "")
        except Exception:
            pass
        recs[name] = {"id": eid, "source": src, "peaks": pk}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"n_compounds": len(recs), "compounds": recs}, indent=0))
    return {"n_compounds": len(recs), "path": str(out_path),
            "compounds": sorted(recs)}


def load_bundle(path: Path = BUNDLE_PATH) -> Dict[str, List[List[float]]]:
    """{compound: [[ppm, intensity], ...]} from the bundled JSON (no network).
    Falls back to fetching if the bundle is absent."""
    if Path(path).exists():
        d = json.loads(Path(path).read_text())
        return {name: rec["peaks"] for name, rec in d["compounds"].items()}
    return experimental_peaks()


if __name__ == "__main__":       # off-VM bundle builder
    import sys
    res = build_bundle()
    print(json.dumps(res, indent=2))
    sys.exit(0 if res["n_compounds"] >= 4 else 1)
