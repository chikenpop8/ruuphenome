"""
Build a LARGE open NMR pretraining corpus from BMRB metabolomics standards.

The bundled corpus ships only 12 reference spectra, which caps the self-supervised
encoder's diversity. This builder auto-discovers every public BMRB metabolomics
entry (~5,500 listed via the BMRB API), downloads each as a raw Bruker ZIP, runs it
through the same Domain 1 pipeline used elsewhere, interpolates onto the fixed ppm
grid, and overwrites open_data/bmrb_1h_corpus.npz with the expanded set.

train_on_h100.py then trains the encoder on this much larger corpus unchanged.

Policy: OPEN data only. Every spectrum is a public BMRB standard with its source URL
and SHA256 recorded in provenance.json. The closed competition dataset is never used.
Run where there is internet (your laptop, or a LiCO job — software/open-data downloads
are explicitly permitted by the hackathon guide). NMR_OFFLINE blocks it on purpose.

Usage:
    python -m nmr_api.build_open_corpus --max-entries 800 --experiments 1,2
    python -m nmr_api.train_on_h100 --epochs 300 --embedding-dim 128
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, List, Optional, Sequence

import numpy as np
import requests

from . import signal_processing
from .open_data import (
    CORPUS_PATH,
    DATA_DIR,
    GRID_PPM,
    PROVENANCE_PATH,
    RAW_DIR,
    _interpolate_to_grid,
    entry_url,
    page_url,
    sha256,
)

LIST_API = "https://api.bmrb.io/v2/list_entries?database=metabolomics"
HEADERS = {
    "User-Agent": "RuuPhenome/0.3 open-data research client",
    "Accept": "application/json",
}
SUMMARY_PATH = DATA_DIR / "corpus_summary.json"


def _check_online() -> None:
    if os.environ.get("NMR_OFFLINE", "").strip().lower() in ("1", "true", "yes"):
        raise RuntimeError(
            "NMR_OFFLINE is set — corpus building needs internet for OPEN data. "
            "Run it on a machine/job with network access; never on the closed VM "
            "once the competition dataset is present."
        )


def list_entries(timeout: int = 60) -> List[str]:
    """Return every BMRB metabolomics experimental entry id (bmseXXXXXX)."""
    resp = requests.get(LIST_API, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, dict):  # defensive: unwrap if a future API wraps the list
        for value in data.values():
            if isinstance(value, list):
                data = value
                break
    return [str(x) for x in data if str(x).startswith("bmse")]


def _fetch_one(
    entry_id: str, experiments: Sequence[int], timeout: int
) -> Optional[Dict]:
    """Download + process one entry to a grid spectrum. Returns a record or an error."""
    last_error = "no 1D experiment processed"
    for experiment in experiments:
        entry = {"id": entry_id, "experiment": experiment}
        dest = RAW_DIR / f"{entry_id}_{experiment}.zip"
        url = entry_url(entry)
        try:
            if not dest.exists():
                response = requests.get(url, headers=HEADERS, timeout=timeout)
                response.raise_for_status()
                if len(response.content) < 1024:
                    raise RuntimeError("empty or too-small download")
                dest.write_bytes(response.content)
            fid, acquisition = signal_processing.read_bruker_zip(dest.read_bytes())
            result = signal_processing.process_fid(
                fid,
                acquisition["sw_hz"],
                acquisition["sf_mhz"],
                carrier_ppm=acquisition["carrier_ppm"],
                assignment_backend="pattern-matcher",
            )
            spectrum = _interpolate_to_grid(
                np.asarray(result["ppm"], dtype=float),
                np.asarray(result["intensity"], dtype=float),
            )
            if not np.isfinite(spectrum).all():
                raise RuntimeError("non-finite spectrum")
            return {
                "id": entry_id,
                "experiment": experiment,
                "label": entry_id,
                "download_url": url,
                "source_page": page_url(entry),
                "sha256": sha256(dest),
                "spectrum": spectrum,
            }
        except Exception as exc:  # try the next experiment number
            last_error = str(exc)
            if dest.exists() and dest.stat().st_size < 1024:
                dest.unlink(missing_ok=True)
            continue
    return {"id": entry_id, "error": last_error}


def build(
    max_entries: int = 800,
    experiments: Sequence[int] = (1,),
    pause: float = 0.15,
    timeout: int = 60,
) -> Dict:
    """Download, process and stack as many BMRB metabolomics spectra as possible."""
    _check_online()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_ids = list_entries(timeout=timeout)
    target_ids = all_ids[:max_entries] if max_entries > 0 else all_ids
    print(
        f"BMRB metabolomics entries listed: {len(all_ids)}; "
        f"attempting: {len(target_ids)} (experiments={list(experiments)})"
    )

    spectra: List[np.ndarray] = []
    labels: List[str] = []
    metadata: List[Dict] = []
    failures: List[Dict] = []
    for index, entry_id in enumerate(target_ids, 1):
        record = _fetch_one(entry_id, experiments, timeout)
        if record and "spectrum" in record:
            spectra.append(record.pop("spectrum"))
            labels.append(record["label"])
            metadata.append(record)
        else:
            failures.append(record or {"id": entry_id, "error": "unknown"})
        if index % 25 == 0:
            print(
                f"  {index}/{len(target_ids)} processed — "
                f"{len(spectra)} ok, {len(failures)} skipped"
            )
        time.sleep(pause)

    if not spectra:
        raise RuntimeError(
            f"No spectra processed from {len(target_ids)} entries. "
            f"First errors: {failures[:3]}"
        )

    np.savez_compressed(
        CORPUS_PATH,
        spectra=np.stack(spectra),
        ppm=GRID_PPM,
        labels=np.asarray(labels),
        metadata=np.asarray([json.dumps(item) for item in metadata]),
    )
    provenance = {
        "dataset": "RuuPhenome expanded BMRB 1D 1H reference corpus",
        "source": "Biological Magnetic Resonance Bank metabolomics standards",
        "data_policy": "Publicly available free of charge; preserve attribution and DOI.",
        "policy_url": "https://bmrb.io/metabolomics/data_policy.shtml",
        "list_api": LIST_API,
        "entries": metadata,
    }
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2))
    summary = {
        "corpus_path": str(CORPUS_PATH),
        "n_spectra": len(spectra),
        "n_points": int(len(GRID_PPM)),
        "n_attempted": len(target_ids),
        "n_failed": len(failures),
        "labels": labels[:50],
        "provenance_path": str(PROVENANCE_PATH),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(
        f"=== corpus built: {len(spectra)} spectra "
        f"({len(failures)} skipped) → {CORPUS_PATH} ==="
    )
    return summary


def main() -> Dict:
    parser = argparse.ArgumentParser(description="Build expanded open BMRB corpus.")
    parser.add_argument(
        "--max-entries",
        type=int,
        default=800,
        help="cap on entries to attempt (0 = all ~5500)",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        default="1",
        help="comma list of experiment numbers to try, e.g. 1,2,3",
    )
    parser.add_argument(
        "--pause",
        type=float,
        default=0.15,
        help="seconds between downloads (be polite to BMRB)",
    )
    args = parser.parse_args()
    experiments = [int(x) for x in str(args.experiments).split(",") if x.strip()]
    return build(
        max_entries=args.max_entries, experiments=experiments, pause=args.pause
    )


if __name__ == "__main__":
    main()
