"""
pSCNN — pseudo-Siamese 1D-CNN compound identifier (Track-1, OPTIONAL evidence).

A *learned* present/absent compound classifier that COMPLEMENTS — never replaces —
the deterministic NNLS + target-decoy-FDR baseline (which is already ~0.90 F1 on
in-distribution synthetic; see `track1_benchmark`). Given a (reference-standard
spectrum, sample spectrum) pair, two weight-independent 1D-CNN towers decide
whether the reference compound is present in the sample. CNN translation-
invariance absorbs pH/temperature/referencing ppm drift — which is why it can help
resolve overlapping signals where fixed-tolerance matching fails.

Precedent: pSCNN (Wei et al. 2022, *Molecules* 27:3653, doi:10.3390/molecules27123653)
— 99.8% on augmented / 97.6% on real known mixtures.

**Governance:** trains ONLY on open/simulated superposed reference-standard
mixtures (off-VM). Provided competition annotations are used only for local on-VM
fine-tuning (`NMR_OFFLINE=1`), never exported. **Honesty:** any in-distribution
number is optimistic — the decisive test is held-out REAL data; keep it OPTIONAL
(no checkpoint → silently off, like the SSL encoder) and blend it as an evidence
channel, never quote a leaky headline. D2O guard applied: only non-exchangeable
C-H resonances build fingerprints.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from . import identification_quality as idq
except ImportError:  # pragma: no cover - direct execution
    import identification_quality as idq  # type: ignore

_MODELS = Path(__file__).resolve().parent / "models"
CHECKPOINT_PATH = _MODELS / "pscnn_identifier.pt"


def available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def status() -> Dict:
    have_ckpt = CHECKPOINT_PATH.exists()
    have_torch = available()
    # Make degradation NON-silent: if the hybrid channel is off, say exactly why so a
    # clean clone / minimal install doesn't quietly fall back to deterministic-only.
    if have_torch and have_ckpt:
        note = "pSCNN hybrid active."
    elif not have_torch:
        note = ("torch not installed — running deterministic-only (NNLS + target–decoy FDR). "
                "Install torch to enable the pSCNN hybrid channel.")
    else:
        note = (f"pSCNN checkpoint missing at {CHECKPOINT_PATH.name} — running deterministic-only. "
                "Commit/restore models/pscnn_identifier.pt (or run train_on_h100.py --supervised pscnn) "
                "to enable the hybrid.")
    return {
        "available": have_torch,
        "trained": have_ckpt,
        "hybrid_active": bool(have_torch and have_ckpt),
        "checkpoint": str(CHECKPOINT_PATH) if have_ckpt else None,
        "note": note,
        "role": "optional learned evidence channel; blended with deterministic NNLS+FDR, not a replacement",
    }


# ── grid + fingerprints (D2O-guarded: non-exchangeable protons only) ──────────
def make_grid(n: int = 512, lo: float = 0.5, hi: float = 9.5) -> np.ndarray:
    return np.linspace(lo, hi, n)


def fingerprint(shifts: Sequence[float], grid: np.ndarray, sigma: float = 0.03) -> np.ndarray:
    spec = np.zeros(len(grid), dtype=np.float32)
    for sh in shifts:
        if idq.classify_shift(float(sh)) != "non_exchangeable":
            continue
        spec += np.exp(-((grid - float(sh)) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)
    m = float(spec.max())
    return (spec / m) if m > 0 else spec


def resample(bin_ppm, intensity, grid) -> np.ndarray:
    y = np.interp(grid, np.asarray(bin_ppm, float), np.clip(np.asarray(intensity, float), 0, None))
    m = float(y.max())
    return (y / m).astype(np.float32) if m > 0 else y.astype(np.float32)


# ── model (torch, built lazily) ──────────────────────────────────────────────
def _build_model():
    import torch
    import torch.nn as nn

    def _tower():                       # 3-conv tower → fixed 256-d embedding
        # BatchNorm stabilises training on sparse, max-normalised spectra
        # (without it the ReLUs die and the net never learns).
        return nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.BatchNorm1d(16), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(), nn.AdaptiveMaxPool1d(8),
        )

    class PSCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.ref = _tower()          # pseudo-Siamese: SAME architecture,
            self.samp = _tower()         # INDEPENDENT weights
            # head sees both embeddings AND explicit matching features
            # (elementwise product + |difference|) — the "is the reference present
            # in the sample?" signal — which makes the matching learnable fast.
            self.head = nn.Sequential(
                nn.Linear(4 * 32 * 8, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, 1))

        def forward(self, r, s):
            re = self.ref(r.unsqueeze(1)).flatten(1)
            se = self.samp(s.unsqueeze(1)).flatten(1)
            feat = torch.cat([re, se, re * se, (re - se).abs()], dim=1)
            return self.head(feat).squeeze(1)   # logits

    return PSCNN()


# ── training data: superposed open reference-standard mixtures ───────────────
def _simulate_pairs(panel: Dict[str, List[float]], grid, n_mixtures, seed,
                    sigma=0.03, noise=0.02, drift=0.012):
    """Positive/negative (ref, sample) pairs. The SAMPLE peaks are randomly
    ppm-drifted (augmentation) while the REF fingerprint stays canonical, so the
    CNN learns to match despite the pH/temperature/referencing drift that defeats
    fixed-tolerance matching."""
    rng = np.random.default_rng(seed)
    names = list(panel)
    fps = {n: fingerprint(panel[n], grid, sigma) for n in names}   # canonical refs
    R, S, Y = [], [], []
    for _ in range(n_mixtures):
        k = int(rng.integers(2, max(3, len(names) // 2 + 1)))
        present = list(rng.choice(names, size=min(k, len(names)), replace=False))
        sample = np.zeros(len(grid), dtype=np.float32)
        for n in present:
            d = float(rng.normal(0.0, drift))                      # per-compound ppm drift
            drifted = fingerprint([s + d for s in panel[n]], grid, sigma)
            sample += np.float32(rng.lognormal(0.0, 0.4)) * drifted
        sample += np.abs(rng.normal(noise, noise * 0.3, size=len(grid))).astype(np.float32)
        m = float(sample.max())
        sample = sample / m if m > 0 else sample
        absent = [n for n in names if n not in present]
        neg = list(rng.choice(absent, size=min(len(absent), len(present)), replace=False)) if absent else []
        for n in present:                                   # positive pairs
            R.append(fps[n]); S.append(sample); Y.append(1.0)
        for n in neg:                                       # balanced negatives
            R.append(fps[n]); S.append(sample); Y.append(0.0)
    return (np.asarray(R, np.float32), np.asarray(S, np.float32),
            np.asarray(Y, np.float32), fps)


def _pick_device():
    """CUDA (H100) if present, else Apple MPS, else CPU — so the GPU is actually used."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Curated common blood/urine metabolite panel the SERVE-TIME pSCNN is trained on
# (the NCD-relevant + routine ¹H-NMR set). Kept small so the checkpoint is fast to
# train and the learned channel stays interpretable.
_APP_PANEL = [
    "glucose", "lactate", "alanine", "valine", "leucine", "isoleucine", "citrate",
    "creatinine", "glycine", "glutamine", "glutamate", "tyrosine", "phenylalanine",
    "histidine", "threonine", "serine", "acetate", "pyruvate", "succinate", "betaine",
    "taurine", "methionine", "lysine", "proline", "choline", "formate",
    "2-oxoisovalerate", "3-hydroxybutyrate", "myo-inositol", "creatine",
]


def default_panel() -> Dict[str, List[float]]:
    """The serve-time panel {name: shifts}, drawn from the reference library."""
    try:
        from . import spectral_cohort as sc
    except ImportError:  # pragma: no cover
        import spectral_cohort as sc  # type: ignore
    return {n: sc.REFERENCE_SHIFTS[n] for n in _APP_PANEL if n in sc.REFERENCE_SHIFTS}


def train(panel: Dict[str, List[float]], *, grid=None, n_mixtures=300, epochs=20,
          lr=1e-3, batch_size=256, seed=0, save=True):
    """Train the pSCNN on synthetic superposed mixtures of `panel`. Returns
    (model, meta). Open-data only — safe to run off-VM."""
    import torch
    torch.manual_seed(seed)                              # reproducible weight init/dropout
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    grid = make_grid() if grid is None else np.asarray(grid, float)
    R, S, Y, fps = _simulate_pairs(panel, grid, n_mixtures, seed)
    device = _pick_device()
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = _build_model().to(device)
    Rt = torch.as_tensor(R, device=device)             # full dataset onto the GPU once
    St = torch.as_tensor(S, device=device)
    Yt = torch.as_tensor(Y, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    rng = np.random.default_rng(seed)
    n = len(Yt)
    hist: List[float] = []
    model.train()
    for _ in range(epochs):
        perm = rng.permutation(n)
        tot = 0.0
        for i in range(0, n, batch_size):
            b = torch.as_tensor(perm[i:i + batch_size], device=device)   # index on-device
            opt.zero_grad()
            loss = loss_fn(model(Rt[b], St[b]), Yt[b])
            loss.backward(); opt.step()
            tot += float(loss.detach()) * len(b)
        hist.append(round(tot / n, 5))
    if device.type == "cuda":
        torch.cuda.synchronize()
    meta = {"grid": grid, "panel": list(panel), "fingerprints": fps,
            "panel_shifts": {n: list(map(float, panel[n])) for n in panel},
            "device": device.type, "loss_history": hist}
    if save:
        save_checkpoint(model, meta)
    return model, meta


def identify(bundle, bin_ppm, intensity, threshold: float = 0.5) -> Dict:
    """Per-compound present probability for one sample spectrum."""
    import torch
    model, meta = bundle
    grid, panel, fps = meta["grid"], meta["panel"], meta["fingerprints"]
    dev = next(model.parameters()).device
    samp = resample(bin_ppm, intensity, grid)
    Rt = torch.as_tensor(np.asarray([fps[n] for n in panel], np.float32), device=dev)
    St = torch.as_tensor(np.repeat(samp[None, :], len(panel), axis=0), device=dev)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Rt, St)).float().cpu().numpy()
    return {panel[i]: round(float(probs[i]), 4) for i in range(len(panel))}


def identify_cohort(bundle, X, bin_ppm, *, present_threshold: float = 0.5) -> Dict:
    """Run identify() on every sample of a cohort (samples × bins) and aggregate:
    a panel compound is 'present' if its MEAN present-probability across samples
    exceeds present_threshold. Returns {'present': [...], 'mean_probability': {...},
    'detection_rate': {...}} — the serve-time learned identification channel."""
    import numpy as np
    bins = np.asarray(bin_ppm, dtype=float)
    panel = bundle[1]["panel"]
    psum = {n: 0.0 for n in panel}
    dcount = {n: 0 for n in panel}
    n_samples = int(X.shape[0])
    for i in range(n_samples):
        probs = identify(bundle, bins, np.asarray(X.iloc[i].values, dtype=float))
        for k, v in probs.items():
            psum[k] += v
            if v >= present_threshold:
                dcount[k] += 1
    denom = max(1, n_samples)
    mean = {k: round(psum[k] / denom, 4) for k in panel}
    rate = {k: round(dcount[k] / denom, 3) for k in panel}
    present = sorted([k for k in panel if mean[k] >= present_threshold],
                     key=lambda k: -mean[k])
    return {"present": present, "mean_probability": mean, "detection_rate": rate,
            "panel_size": len(panel), "n_samples": n_samples}


def save_checkpoint(model, meta, path=None) -> None:
    import torch
    p = Path(path) if path else CHECKPOINT_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "panel": meta["panel"],
                "grid": np.asarray(meta["grid"]).tolist(),
                "panel_shifts": meta["panel_shifts"],
                "loss_history": meta.get("loss_history", [])}, p)


def load_checkpoint(path: Optional[Path] = None):
    import torch
    p = path or CHECKPOINT_PATH
    if not Path(p).exists():
        raise FileNotFoundError("pSCNN checkpoint is not trained.")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    grid = np.asarray(ck["grid"], float)
    panel, shifts = ck["panel"], ck["panel_shifts"]
    model = _build_model()
    model.load_state_dict(ck["state_dict"])
    fps = {n: fingerprint(shifts[n], grid) for n in panel}
    return model, {"grid": grid, "panel": panel, "fingerprints": fps,
                   "loss_history": ck.get("loss_history", [])}


def main(argv=None) -> int:
    """Train the pSCNN on open synthetic mixtures (off-VM) and save a checkpoint.
    On the H100/LiCO node, upload the code + reference library and run this — it
    downloads nothing (open reference shifts are bundled)."""
    import argparse
    ap = argparse.ArgumentParser(description="Train the pSCNN compound identifier (open-data, RUO).")
    ap.add_argument("--mixtures", type=int, default=20000, help="synthetic training mixtures")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--grid", type=int, default=2048, help="spectrum length (bins)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--panel-size", type=int, default=0, help="0 = full reference library")
    args = ap.parse_args(argv)
    try:
        from . import spectral_cohort as sc
    except ImportError:  # pragma: no cover
        import spectral_cohort as sc  # type: ignore
    refs = sc.REFERENCE_SHIFTS
    names = list(refs)[:args.panel_size] if args.panel_size else list(refs)
    panel = {n: refs[n] for n in names}
    grid = make_grid(args.grid)
    print(f"pSCNN training (open-data): {len(panel)} compounds · {args.mixtures} mixtures · "
          f"{args.epochs} epochs · grid {args.grid}")
    model, meta = train(panel, grid=grid, n_mixtures=args.mixtures, epochs=args.epochs,
                        lr=args.lr, batch_size=args.batch_size, save=True)
    h = meta["loss_history"]
    print(f"done · loss {h[0]:.4f} → {h[-1]:.4f} · checkpoint {CHECKPOINT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
