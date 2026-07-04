"""
F3 — on-VM fine-tune loader for the competition's PROVIDED compound annotations.

WHAT: turns the organizer-provided training annotations into a fine-tune panel and
(on demand) fine-tunes the serve-time pSCNN identifier on them, so the learned
channel is trained on REAL labels instead of simulation. This is the highest-value
Track-1 training path (real labels beat GISSMO sims).

GOVERNANCE (hard rules — enforced here, not just documented):
  * The provided annotations are CLOSED data. Any function that TOUCHES a fine-tune
    (`finetune_pscnn`) refuses to run unless NMR_OFFLINE=1 (the governed offline VM).
  * Reads a LOCAL path only; opens NO network connection; writes the checkpoint
    LOCALLY only. Closed data is never exported.

STATUS: ready-to-use path. It is NOT trained here and is NOT invoked automatically.
Parsing/validation is exercised on OPEN/synthetic data only. Run the real fine-tune
yourself, inside the VM, once the organizer annotations arrive.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Dict, List, Optional

try:
    from . import pscnn, spectral_cohort as sc
except ImportError:  # pragma: no cover - direct execution
    import pscnn  # type: ignore
    import spectral_cohort as sc  # type: ignore


def offline_ok() -> bool:
    """True only inside the governed offline VM (NMR_OFFLINE=1)."""
    return os.environ.get("NMR_OFFLINE") == "1"


def _require_offline() -> None:
    if not offline_ok():
        raise RuntimeError(
            "F3 fine-tune is gated to the governed VM: set NMR_OFFLINE=1. The provided "
            "annotations are closed data and must never be processed off-VM or exported.")


def load_annotation_panel(raw: bytes) -> Dict[str, List[float]]:
    """Parse organizer annotations into a {compound: [¹H shifts]} panel. Pure local
    parsing — no network, no training. Accepts either:
      (a) a PEAK table  {ppm, metabolite}                 → grouped by compound
      (b) a PANEL table {metabolite, shift, shift, ...}   → shifts per compound row
    Whichever the organizer ships, this yields a compound→shifts panel the pSCNN can
    fine-tune on. Non-numeric / out-of-range (0–12 ppm) tokens are ignored."""
    panel: Dict[str, List[float]] = {}
    # (a) {ppm: metabolite} peak form — reuse the vetted parser
    try:
        for ppm, name in sc.parse_identified_peaks(raw).items():
            key = str(name).strip().lower()
            if key:
                panel.setdefault(key, []).append(float(ppm))
    except Exception:
        pass
    if panel:
        return {k: sorted(set(round(v, 4) for v in vs)) for k, vs in panel.items() if vs}

    # (b) compound-per-row table with shift columns
    text = raw.decode("utf-8", errors="replace")
    sep = "\t" if text[:4096].count("\t") >= text[:4096].count(",") else ","
    for i, line in enumerate(io.StringIO(text)):
        cells = line.rstrip("\n").split(sep)
        if not cells or not cells[0].strip():
            continue
        name = cells[0].strip().lower()
        if i == 0 and not any(c.replace(".", "", 1).lstrip("-").isdigit() for c in cells[1:]):
            continue  # header row
        shifts: List[float] = []
        for cell in cells[1:]:
            for tok in cell.replace(";", " ").replace(",", " ").split():
                try:
                    v = float(tok)
                    if 0.0 <= v <= 12.0:
                        shifts.append(round(v, 4))
                except ValueError:
                    continue
        if name and shifts:
            panel.setdefault(name, []).extend(shifts)
    return {k: sorted(set(v)) for k, v in panel.items() if v}


def build_finetune_panel(annotations: Dict[str, List[float]], *,
                         include_base: bool = True) -> Dict[str, List[float]]:
    """Merge the provided-annotation compounds with the serve panel; the PROVIDED
    (real) shifts take precedence for any overlapping compound."""
    base = pscnn.default_panel() if include_base else {}
    merged = dict(base)
    merged.update({k: v for k, v in annotations.items() if v})   # provided wins
    return merged


def finetune_pscnn(annotations_path, *, out: Optional[Path] = None, include_base: bool = True,
                   epochs: int = 40, n_mixtures: int = 8000, seed: int = 0) -> Dict:
    """ON-VM ONLY (NMR_OFFLINE=1). Fine-tune the pSCNN identifier on the provided
    annotations and save a new checkpoint LOCALLY (default: the serve checkpoint, so
    the app immediately uses the fine-tuned model). Refuses off-VM; never networks;
    never exports the annotations. NOT invoked automatically — run this yourself in
    the VM once the organizer annotations are available."""
    _require_offline()
    raw = Path(annotations_path).read_bytes()
    panel = build_finetune_panel(load_annotation_panel(raw), include_base=include_base)
    if len(panel) < 2:
        raise ValueError("Parsed < 2 annotated compounds — check the annotations file format "
                         "(expected {ppm, metabolite} or {metabolite, shifts...}).")
    dest = Path(out) if out else pscnn.CHECKPOINT_PATH
    model, meta = pscnn.train(panel, n_mixtures=n_mixtures, epochs=epochs, seed=seed, save=False)
    pscnn.save_checkpoint(model, meta, path=dest)
    return {
        "n_annotated_compounds": len(panel),
        "epochs": epochs, "n_mixtures": n_mixtures,
        "checkpoint": str(dest),
        "final_loss": meta["loss_history"][-1],
        "trained_on_device": meta.get("device"),
        "governance": "NMR_OFFLINE on-VM only; local read/write; closed annotations not exported",
    }


if __name__ == "__main__":   # on-VM CLI: python -m nmr_api.finetune_loader <annotations_file>
    import json
    import sys
    if len(sys.argv) < 2:
        print("usage (inside the VM, NMR_OFFLINE=1): python -m nmr_api.finetune_loader <annotations_file> [out.pt]")
        raise SystemExit(2)
    out = sys.argv[2] if len(sys.argv) > 2 else None
    print(json.dumps(finetune_pscnn(sys.argv[1], out=out), indent=2))
