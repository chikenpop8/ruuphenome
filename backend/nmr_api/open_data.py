"""
Auditable open NMR data ingestion.

Downloads selected BMRB raw 1D 1H Bruker experiments, records checksums and
provenance, processes them through the same Domain 1 pipeline, and exports a
fixed-grid corpus suitable for self-supervised learning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import requests

from . import signal_processing


MODULE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = MODULE_DIR / "open_data_manifest.json"
DATA_DIR = MODULE_DIR / "open_data"
RAW_DIR = DATA_DIR / "bmrb_raw"
CORPUS_PATH = DATA_DIR / "bmrb_1h_corpus.npz"
PROVENANCE_PATH = DATA_DIR / "provenance.json"
GRID_PPM = np.linspace(10.0, 0.0, 4096, dtype=np.float32)


def load_manifest() -> Dict:
    return json.loads(MANIFEST_PATH.read_text())


def entry_url(entry: Dict) -> str:
    return (
        "https://bmrb.io/metabolomics/mol_summary/zip_entry_directory.php"
        f"?experiment={entry['experiment']}&id={entry['id']}"
    )


def page_url(entry: Dict) -> str:
    return (
        "https://bmrb.io/metabolomics/mol_summary/show_data.php"
        f"?id={entry['id']}"
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_corpus(force: bool = False, timeout: int = 90) -> Dict:
    manifest = load_manifest()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict] = []
    for entry in manifest["entries"]:
        filename = f"{entry['id']}_{entry['experiment']}.zip"
        destination = RAW_DIR / filename
        url = entry_url(entry)
        if force or not destination.exists():
            response = requests.get(
                url,
                timeout=timeout,
                headers={"User-Agent": "RuuPhenome/0.2 open-data research client"},
            )
            response.raise_for_status()
            destination.write_bytes(response.content)
        records.append(
            {
                **entry,
                "download_url": url,
                "source_page": page_url(entry),
                "local_file": str(destination.relative_to(MODULE_DIR)),
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
    provenance = {
        "dataset": manifest["dataset"],
        "source": manifest["source"],
        "data_policy": manifest["data_policy"],
        "policy_url": manifest["policy_url"],
        "entries": records,
    }
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROVENANCE_PATH.write_text(json.dumps(provenance, indent=2))
    return provenance


def _interpolate_to_grid(ppm: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    order = np.argsort(ppm)
    ascending_ppm = np.asarray(ppm)[order]
    ascending_y = np.asarray(intensity)[order]
    interpolated = np.interp(
        GRID_PPM[::-1], ascending_ppm, ascending_y, left=0.0, right=0.0
    )[::-1]
    scale = np.percentile(np.abs(interpolated), 99.5) or 1.0
    return np.clip(interpolated / scale, -1.5, 1.5).astype(np.float32)


def build_processed_corpus(force_download: bool = False) -> Dict:
    provenance = download_corpus(force=force_download)
    spectra, labels, metadata = [], [], []
    failures = []
    for entry in provenance["entries"]:
        path = MODULE_DIR / entry["local_file"]
        try:
            fid, acquisition = signal_processing.read_bruker_zip(path.read_bytes())
            result = signal_processing.process_fid(
                fid,
                acquisition["sw_hz"],
                acquisition["sf_mhz"],
                carrier_ppm=acquisition["carrier_ppm"],
                assignment_backend="pattern-matcher",
            )
            spectra.append(
                _interpolate_to_grid(
                    np.asarray(result["ppm"], dtype=float),
                    np.asarray(result["intensity"], dtype=float),
                )
            )
            labels.append(entry["name"])
            metadata.append(
                {
                    "id": entry["id"],
                    "doi": entry["doi"],
                    "sha256": entry["sha256"],
                    "source_page": entry["source_page"],
                    "quality_control": result["quality_control"],
                    "acquisition": acquisition,
                }
            )
        except Exception as exc:
            failures.append({"id": entry["id"], "error": str(exc)})

    if not spectra:
        raise RuntimeError(f"No BMRB spectra could be processed: {failures}")
    np.savez_compressed(
        CORPUS_PATH,
        spectra=np.stack(spectra),
        ppm=GRID_PPM,
        labels=np.asarray(labels),
        metadata=np.asarray([json.dumps(item) for item in metadata]),
    )
    summary = {
        "corpus_path": str(CORPUS_PATH),
        "n_spectra": len(spectra),
        "n_points": len(GRID_PPM),
        "labels": labels,
        "failures": failures,
        "provenance_path": str(PROVENANCE_PATH),
    }
    (DATA_DIR / "corpus_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-download", action="store_true")
    args = parser.parse_args()
    print(json.dumps(build_processed_corpus(args.force_download), indent=2))
