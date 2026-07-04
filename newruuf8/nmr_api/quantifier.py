"""
F8 — GISSMO transformer quantifier (Track-1 H100 hero; OPTIONAL evidence).

A transformer that maps a ¹H spectrum → a per-compound **relative concentration**
vector (identity = concentration above a threshold), trained on GISSMO-simulated
mixtures with known ground truth (`gissmo_corpus`). Architecture: a Conv1d patch
embedding (ViT-hybrid — stabilises training on sparse spectra) → learnable
positional embedding → Transformer encoder → mean-pool → MLP → softplus
concentrations.

**Why it exists:** the held-out validation showed deterministic matching + the
library collapse on *real* GISSMO shifts (F1 ~0.16); training a model on the
**real GISSMO peak patterns** (not the simplified library) is the way to close that
sim-to-real gap. It stays an evidence channel blended with the deterministic
baseline — never a sole replacement — and every number is caveated as
research-use-only, not clinical.

**Governance:** trains on open GISSMO/simulated data (off-VM / on the GPU node with
the bundled corpus, no download); provided competition annotations only for on-VM
fine-tuning (`NMR_OFFLINE=1`), never exported. **OPTIONAL:** no checkpoint → off.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

try:
    from . import gissmo_corpus, pscnn
except ImportError:  # pragma: no cover
    import gissmo_corpus, pscnn  # type: ignore

_MODELS = Path(__file__).resolve().parent / "models"
CHECKPOINT_PATH = _MODELS / "gissmo_quantifier.pt"


def _pick_device():
    """The compute device: CUDA (H100) if present, else Apple MPS, else CPU.
    Without this the model trained on CPU even on a GPU node — the H100 sat idle."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


def status() -> Dict:
    return {
        "available": available(),
        "trained": CHECKPOINT_PATH.exists(),
        "checkpoint": str(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else None,
        "role": "optional learned quantifier (identity + relative concentration); blended, not a replacement",
    }


def _build_model(n_bins: int, n_compounds: int, *, d_model: int = 64, nhead: int = 4,
                 nlayers: int = 2, patch: int = 16):
    import torch
    import torch.nn as nn

    class Quantifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = nn.Conv1d(1, d_model, kernel_size=patch, stride=patch)  # ViT patch embed
            n_patch = n_bins // patch
            self.pos = nn.Parameter(torch.zeros(1, n_patch, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model, nhead, dim_feedforward=4 * d_model, dropout=0.1, batch_first=True)
            self.encoder = nn.TransformerEncoder(layer, nlayers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Sequential(
                nn.Linear(d_model, 128), nn.ReLU(), nn.Dropout(0.1), nn.Linear(128, n_compounds))

        def forward(self, x):                       # x: (B, n_bins)
            h = self.patch(x.unsqueeze(1)).transpose(1, 2)   # (B, n_patch, d_model)
            h = self.encoder(h + self.pos)
            h = self.norm(h.mean(dim=1))                     # mean-pool
            return torch.nn.functional.softplus(self.head(h))  # (B, K) ≥ 0 relative conc

    return Quantifier()


def train(corpus: Optional[Dict[str, List[float]]] = None, *, grid=None, n_bins: int = 2048,
          epochs: int = 100, steps_per_epoch: int = 64, batch_size: int = 128, lr: float = 3e-4,
          d_model: int = 64, nhead: int = 4, nlayers: int = 2, patch: int = 16,
          drift: float = 0.012, noise: float = 0.02, seed: int = 0, save: bool = True):
    """Train the quantifier on GISSMO-simulated mixtures (open data). Returns
    (model, meta). Generates mixtures on the fly — unlimited training data."""
    import torch
    corpus = corpus or gissmo_corpus.load_corpus()
    names = list(corpus)
    grid = pscnn.make_grid(n_bins) if grid is None else np.asarray(grid, float)
    fps = {n: pscnn.fingerprint(corpus[n], grid) for n in names}
    device = _pick_device()
    if device.type == "cuda":                         # Hopper/Ampere: fast TF32 matmuls
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    amp = device.type == "cuda"                        # bf16 autocast — big H100 speedup
    model = _build_model(len(grid), len(names), d_model=d_model, nhead=nhead,
                         nlayers=nlayers, patch=patch).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    rng = np.random.default_rng(seed)
    model.train()
    hist: List[float] = []
    for _ in range(epochs):
        # rebuild the drift bank each epoch → fresh offsets, but O(1) per-step sampling
        bank = (gissmo_corpus.build_drift_bank(fps, names, grid, drift, rng=rng)
                if drift else None)
        tot = 0.0
        for _s in range(steps_per_epoch):
            X, C = gissmo_corpus.simulate_batch(fps, names, grid, batch_size, rng,
                                                drift_bank=bank, drift=drift, noise=noise)
            xb = torch.as_tensor(X, device=device)     # move batch onto the GPU
            cb = torch.as_tensor(C, device=device)
            opt.zero_grad()
            if amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = loss_fn(model(xb), cb)
            else:
                loss = loss_fn(model(xb), cb)
            loss.backward(); opt.step()
            tot += float(loss.detach())
        hist.append(round(tot / steps_per_epoch, 6))
    if device.type == "cuda":
        torch.cuda.synchronize()
    meta = {"grid": grid, "names": names, "corpus": corpus, "loss_history": hist,
            "device": device.type,
            "config": {"n_bins": len(grid), "d_model": d_model, "nhead": nhead,
                       "nlayers": nlayers, "patch": patch}}
    if save:
        save_checkpoint(model, meta)
    return model, meta


def predict(bundle, bin_ppm, intensity) -> Dict[str, float]:
    """Per-compound predicted relative concentration for one spectrum."""
    import torch
    model, meta = bundle
    grid, names = meta["grid"], meta["names"]
    dev = next(model.parameters()).device            # run inference on the model's device
    x = pscnn.resample(bin_ppm, intensity, grid)
    model.eval()
    with torch.no_grad():
        conc = model(torch.as_tensor(x[None, :], device=dev)).float().cpu().numpy()[0]
    return {names[i]: round(float(conc[i]), 4) for i in range(len(names))}


def identify(bundle, bin_ppm, intensity, threshold: float = 0.05) -> Dict[str, float]:
    """Compounds whose predicted relative concentration exceeds `threshold`."""
    conc = predict(bundle, bin_ppm, intensity)
    return {n: c for n, c in conc.items() if c >= threshold}


def save_checkpoint(model, meta) -> None:
    import torch
    _MODELS.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "names": meta["names"],
                "grid": np.asarray(meta["grid"]).tolist(),
                "corpus": meta["corpus"], "config": meta["config"],
                "loss_history": meta.get("loss_history", [])}, CHECKPOINT_PATH)


def load_checkpoint(path: Optional[Path] = None):
    import torch
    p = path or CHECKPOINT_PATH
    if not Path(p).exists():
        raise FileNotFoundError("GISSMO quantifier checkpoint is not trained.")
    ck = torch.load(p, map_location="cpu", weights_only=False)
    grid = np.asarray(ck["grid"], float)
    cfg = ck["config"]
    model = _build_model(len(grid), len(ck["names"]), d_model=cfg["d_model"], nhead=cfg["nhead"],
                         nlayers=cfg["nlayers"], patch=cfg["patch"])
    model.load_state_dict(ck["state_dict"])
    return model, {"grid": grid, "names": ck["names"], "corpus": ck["corpus"],
                   "loss_history": ck.get("loss_history", [])}
