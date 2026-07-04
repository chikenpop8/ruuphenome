"""
H100 training entry point — retrain the self-supervised masked NMR encoder.

Designed to run as a LiCO "Common Job" on the KKU H100 cluster (see
H100_TRAINING.md for the exact job-form values). On the H100 you can afford many
more epochs, a larger batch and a bigger embedding than the Mac dev run, which
gives stronger, more separable spectral embeddings.

It is safe to run anywhere (falls back to CPU/MPS): the augmentation pipeline
generates unlimited synthetic mixtures from the open BMRB references, so longer
training genuinely helps even with a small reference corpus.

Usage (on the cluster, inside the project):
    python -m nmr_api.train_on_h100 --epochs 200 --batch-size 256 \
        --embedding-dim 128 --steps-per-epoch 128

Outputs (committed back via the LiCO file tree):
    models/masked_nmr_encoder.pt        retrained checkpoint (latest run, always overwritten)
    models/masked_nmr_training.json     loss history + config (latest run)
    models/h100_training_report.json    full run report incl. retrieval benchmark (latest run)

Every run also auto-saves a copy of all three tagged with the ACTUAL config,
e.g. models/masked_nmr_encoder_10ep_cuda_b256.pt — no manual `cp` step, so a
short run can never silently overwrite a longer run's mislabeled checkpoint.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from . import open_data, self_supervised

REPORT_PATH = self_supervised.MODEL_DIR / "h100_training_report.json"


def _checkpoint_suffix(args: argparse.Namespace, device: dict) -> str:
    """Deterministic tag derived from the ACTUAL run config — no hand-typed names.

    A hand-typed `cp ... _200ep_cuda_b256.pt` after a 10-epoch run silently
    overwrote the real 200-epoch checkpoint with a mislabeled 10-epoch one.
    Deriving the suffix from args/device makes that mistake impossible: every
    run's saved files are named after what actually ran, automatically.
    """
    tag = "cuda" if device.get("cuda_available") else "mps" if device.get("device_name") == "mps" else "cpu"
    return f"{args.epochs}ep_{tag}_b{args.batch_size}"


def _save_named_checkpoint(suffix: str) -> dict:
    saved = {}
    pairs = [
        (self_supervised.CHECKPOINT_PATH, f"masked_nmr_encoder_{suffix}.pt"),
        (REPORT_PATH, f"h100_training_report_{suffix}.json"),
        (self_supervised.TRAINING_REPORT_PATH, f"masked_nmr_training_{suffix}.json"),
    ]
    for src, name in pairs:
        if src.exists():
            dest = self_supervised.MODEL_DIR / name
            shutil.copy2(src, dest)
            saved[src.name] = str(dest)
    return saved


def _device_summary() -> dict:
    try:
        import torch
        cuda = torch.cuda.is_available()
        return {
            "torch": torch.__version__,
            "cuda_available": bool(cuda),
            "device_name": torch.cuda.get_device_name(0) if cuda else self_supervised._device_name(),
            "n_gpu": torch.cuda.device_count() if cuda else 0,
        }
    except Exception as exc:  # pragma: no cover
        return {"error": str(exc)}


def _train_gissmo_quant(args: argparse.Namespace, device: dict) -> dict:
    """F8 — train the GISSMO transformer quantifier on open simulated mixtures.
    Loads the bundled corpus (no download on the node); config-tags the outputs."""
    from . import gissmo_corpus, quantifier
    corpus = gissmo_corpus.load_corpus()
    src = "gissmo_corpus.json" if gissmo_corpus.CORPUS_PATH.exists() else "verified-panel fallback (run build_corpus off-VM for a bigger corpus)"
    steps = max(1, args.mixtures // args.batch_size)
    train_device = quantifier._pick_device().type            # cuda on the H100 node
    print(f"GISSMO quantifier · device {train_device.upper()} · {len(corpus)} compounds ({src}) · "
          f"field {args.field_mhz} MHz · n_bins {args.n_bins} · epochs {args.epochs} · "
          f"mixtures/epoch {args.mixtures} (steps {steps}) · batch {args.batch_size}")
    if train_device == "cpu":
        print("WARNING: training on CPU — no GPU detected. On the H100 node this should say CUDA.")
    t0 = time.time()
    model, meta = quantifier.train(
        corpus, n_bins=args.n_bins, epochs=args.epochs, steps_per_epoch=steps,
        batch_size=args.batch_size, lr=args.learning_rate, patch=args.patch,
        seed=args.seed, save=True)
    train_seconds = round(time.time() - t0, 1)
    report = {
        "model": "gissmo-quant", "device": device,
        "trained_on_device": meta.get("device"),      # the device the model ACTUALLY ran on
        "config": vars(args),
        "n_compounds": len(meta["names"]), "field_mhz": args.field_mhz,
        "train_seconds": train_seconds,
        "initial_loss": meta["loss_history"][0], "final_loss": meta["loss_history"][-1],
        "checkpoint": str(quantifier.CHECKPOINT_PATH),
        "note": ("Relative-concentration quantifier trained on open GISSMO-simulated mixtures. "
                 "RUO — benchmark identification vs deterministic + pSCNN on the held-out set; "
                 "not clinical validation."),
    }
    report_path = quantifier.CHECKPOINT_PATH.parent / "gissmo_quantifier_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    # config-tagged copies (never hand-named)
    suffix = _checkpoint_suffix(args, device)
    for src_path, name in [(quantifier.CHECKPOINT_PATH, f"gissmo_quantifier_{suffix}.pt"),
                           (report_path, f"gissmo_quantifier_report_{suffix}.json")]:
        if src_path.exists():
            shutil.copy2(src_path, src_path.parent / name)
    report["named_checkpoint_suffix"] = suffix
    print("=== GISSMO quantifier training complete ===")
    print(json.dumps(report, indent=2))
    return report


def main() -> dict:
    ap = argparse.ArgumentParser(description="Retrain the SSL NMR encoder (H100-ready).")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--steps-per-epoch", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--embedding-dim", type=int, default=128)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--rebuild-corpus", action="store_true",
                    help="Re-download + reprocess the open BMRB corpus first.")
    # F8 — supervised GISSMO quantifier mode
    ap.add_argument("--supervised", choices=["ssl", "gissmo-quant"], default="ssl",
                    help="'ssl' (default masked encoder) or 'gissmo-quant' (F8 quantifier)")
    ap.add_argument("--mixtures", type=int, default=250000,
                    help="[gissmo-quant] simulated mixtures per epoch")
    ap.add_argument("--field-mhz", type=int, default=500,
                    help="[gissmo-quant] spectrometer field for the report (informational)")
    ap.add_argument("--n-bins", type=int, default=2048, help="[gissmo-quant] spectrum length")
    ap.add_argument("--patch", type=int, default=16, help="[gissmo-quant] transformer patch size")
    args = ap.parse_args()

    device = _device_summary()
    print("Device:", json.dumps(device))

    if args.supervised == "gissmo-quant":
        return _train_gissmo_quant(args, device)

    # 1. ensure the open BMRB corpus exists (download is best-effort).
    if args.rebuild_corpus or not open_data.CORPUS_PATH.exists():
        print("Building open BMRB corpus…")
        try:
            open_data.download_corpus(force=args.rebuild_corpus)
        except Exception as exc:  # pragma: no cover - network best-effort
            print(f"Corpus build warning: {exc}")

    # 2. train.
    print(f"Training: epochs={args.epochs} batch={args.batch_size} "
          f"dim={args.embedding_dim} steps/epoch={args.steps_per_epoch}")
    t0 = time.time()
    report = self_supervised.train(
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        batch_size=args.batch_size,
        embedding_dim=args.embedding_dim,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    train_seconds = round(time.time() - t0, 1)

    # 3. benchmark retrieval on the new encoder.
    try:
        bench = self_supervised.benchmark_retrieval()
    except Exception as exc:  # pragma: no cover
        bench = {"error": str(exc)}

    full = {
        "device": device,
        "config": vars(args),
        "train_seconds": train_seconds,
        "initial_loss": report.get("initial_loss"),
        "final_loss": report.get("final_loss"),
        "n_open_reference_spectra": report.get("n_open_reference_spectra"),
        "embedding_dim": report.get("embedding_dim"),
        "retrieval_benchmark": bench,
        "checkpoint": report.get("checkpoint"),
    }
    REPORT_PATH.write_text(json.dumps(full, indent=2))

    suffix = _checkpoint_suffix(args, device)
    full["named_checkpoint_suffix"] = suffix
    full["named_checkpoints"] = _save_named_checkpoint(suffix)

    print("=== H100 training complete ===")
    print(json.dumps(full, indent=2))
    return full


if __name__ == "__main__":
    main()
