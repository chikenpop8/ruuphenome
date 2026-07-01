"""
Self-supervised 1D NMR representation learning.

A compact masked convolutional autoencoder learns from unlabeled augmented
mixtures of open BMRB spectra. Compound labels are not used during pretraining;
they are used only afterward to build a transparent reference-embedding index.
"""

from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .open_data import CORPUS_PATH, GRID_PPM


MODULE_DIR = Path(__file__).resolve().parent
MODEL_DIR = MODULE_DIR / "models"
CHECKPOINT_PATH = MODEL_DIR / "masked_nmr_encoder.pt"
TRAINING_REPORT_PATH = MODEL_DIR / "masked_nmr_training.json"


def _torch():
    import torch
    from torch import nn

    return torch, nn


class MaskedSpectrumAutoencoder:
    """Factory wrapper that keeps torch optional at module import time."""

    @staticmethod
    def build(embedding_dim: int = 64):
        torch, nn = _torch()

        class Network(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv1d(1, 16, 9, stride=2, padding=4),
                    nn.GELU(),
                    nn.Conv1d(16, 32, 9, stride=2, padding=4),
                    nn.GELU(),
                    nn.Conv1d(32, embedding_dim, 9, stride=2, padding=4),
                    nn.GELU(),
                    nn.Conv1d(embedding_dim, embedding_dim, 9, stride=2, padding=4),
                    nn.GELU(),
                )
                self.decoder = nn.Sequential(
                    nn.ConvTranspose1d(
                        embedding_dim, embedding_dim, 8, stride=2, padding=3
                    ),
                    nn.GELU(),
                    nn.ConvTranspose1d(
                        embedding_dim, 32, 8, stride=2, padding=3
                    ),
                    nn.GELU(),
                    nn.ConvTranspose1d(32, 16, 8, stride=2, padding=3),
                    nn.GELU(),
                    nn.ConvTranspose1d(16, 1, 8, stride=2, padding=3),
                )

            def forward(self, x):
                encoded = self.encoder(x)
                reconstruction = self.decoder(encoded)
                embedding = encoded.mean(dim=-1)
                embedding = torch.nn.functional.normalize(embedding, dim=-1)
                return reconstruction[..., : x.shape[-1]], embedding

            def embed(self, x):
                encoded = self.encoder(x)
                return torch.nn.functional.normalize(encoded.mean(dim=-1), dim=-1)

        return Network()


def _device_name() -> str:
    torch, _nn = _torch()
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_corpus() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not CORPUS_PATH.exists():
        raise FileNotFoundError(
            f"Open corpus not found at {CORPUS_PATH}. Run "
            "`python -m backend.nmr_api.open_data` first."
        )
    data = np.load(CORPUS_PATH)
    return (
        np.asarray(data["spectra"], dtype=np.float32),
        np.asarray(data["labels"]).astype(str),
        np.asarray(data["ppm"], dtype=np.float32),
    )


def _augment_batch(
    spectra: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    length = spectra.shape[1]
    targets = np.zeros((batch_size, length), dtype=np.float32)
    masked = np.zeros_like(targets)
    masks = np.zeros_like(targets)
    for row in range(batch_size):
        count = int(rng.integers(1, min(5, len(spectra)) + 1))
        indices = rng.choice(len(spectra), size=count, replace=False)
        weights = rng.lognormal(mean=0.0, sigma=0.7, size=count)
        mixture = np.sum(spectra[indices] * weights[:, None], axis=0)
        mixture /= np.percentile(np.abs(mixture), 99.5) or 1.0
        mixture = np.roll(mixture, int(rng.integers(-10, 11)))
        mixture *= float(rng.uniform(0.75, 1.25))
        mixture += rng.normal(0, rng.uniform(0.002, 0.02), length)
        mixture = np.clip(mixture, -1.5, 1.5).astype(np.float32)

        mask = np.zeros(length, dtype=np.float32)
        target_points = int(length * rng.uniform(0.12, 0.28))
        covered = 0
        while covered < target_points:
            width = int(rng.integers(12, 96))
            start = int(rng.integers(0, max(1, length - width)))
            mask[start : start + width] = 1.0
            covered = int(mask.sum())
        corrupted = mixture.copy()
        corrupted[mask.astype(bool)] = rng.normal(
            0, 0.005, int(mask.sum())
        ).astype(np.float32)
        targets[row], masked[row], masks[row] = mixture, corrupted, mask
    return masked, targets, masks


def _reference_embeddings(model, spectra: np.ndarray, device: str) -> np.ndarray:
    torch, _nn = _torch()
    with torch.no_grad():
        tensor = torch.from_numpy(spectra[:, None, :]).to(device)
        embedding = model.embed(tensor).cpu().numpy()
    return embedding.astype(np.float32)


def train(
    *,
    epochs: int = 20,
    steps_per_epoch: int = 32,
    batch_size: int = 16,
    embedding_dim: int = 64,
    learning_rate: float = 1e-3,
    seed: int = 2026,
) -> Dict:
    torch, _nn = _torch()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    spectra, labels, ppm = _load_corpus()
    device = _device_name()
    model = MaskedSpectrumAutoencoder.build(embedding_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    rng = np.random.default_rng(seed)
    history: List[float] = []
    model.train()
    for _epoch in range(max(1, epochs)):
        losses = []
        for _step in range(max(1, steps_per_epoch)):
            masked, target, mask = _augment_batch(spectra, batch_size, rng)
            x = torch.from_numpy(masked[:, None, :]).to(device)
            y = torch.from_numpy(target[:, None, :]).to(device)
            m = torch.from_numpy(mask[:, None, :]).to(device)
            reconstruction, _embedding = model(x)
            masked_loss = ((reconstruction - y) ** 2 * m).sum() / (
                m.sum() + 1e-8
            )
            full_loss = torch.mean((reconstruction - y) ** 2)
            loss = masked_loss + 0.1 * full_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        history.append(float(np.mean(losses)))

    model.eval()
    references = _reference_embeddings(model, spectra, device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "embedding_dim": embedding_dim,
            "labels": labels.tolist(),
            "reference_embeddings": references,
            "ppm": ppm,
            "corpus_path": str(CORPUS_PATH),
            "training": {
                "objective": "masked spectrum reconstruction",
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "batch_size": batch_size,
                "seed": seed,
                "labels_used_during_pretraining": False,
            },
        },
        CHECKPOINT_PATH,
    )
    report = {
        "checkpoint": str(CHECKPOINT_PATH),
        "device": device,
        "n_open_reference_spectra": int(len(spectra)),
        "embedding_dim": embedding_dim,
        "initial_loss": round(history[0], 7),
        "final_loss": round(history[-1], 7),
        "loss_history": [round(value, 7) for value in history],
        "self_supervised": True,
        "labels_used_during_pretraining": False,
        "reference_labels": labels.tolist(),
    }
    TRAINING_REPORT_PATH.write_text(json.dumps(report, indent=2))
    return report


def status() -> Dict:
    try:
        import torch

        torch_version = torch.__version__
    except Exception:
        return {
            "available": False,
            "trained": False,
            "reason": "PyTorch is not installed.",
        }
    report = {}
    if TRAINING_REPORT_PATH.exists():
        try:
            report = json.loads(TRAINING_REPORT_PATH.read_text())
        except Exception:
            report = {}
    return {
        "available": True,
        "trained": CHECKPOINT_PATH.exists(),
        "torch": torch_version,
        "checkpoint": str(CHECKPOINT_PATH) if CHECKPOINT_PATH.exists() else None,
        "corpus_available": CORPUS_PATH.exists(),
        **report,
    }


@lru_cache(maxsize=1)
def _load_model():
    torch, _nn = _torch()
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError("Self-supervised checkpoint is not trained.")
    device = _device_name()
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    model = MaskedSpectrumAutoencoder.build(
        int(checkpoint["embedding_dim"])
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint, device


def _resample(ppm: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    order = np.argsort(ppm)
    x = np.asarray(ppm, dtype=float)[order]
    y = np.asarray(intensity, dtype=float)[order]
    spectrum = np.interp(GRID_PPM[::-1], x, y, left=0.0, right=0.0)[::-1]
    spectrum /= np.percentile(np.abs(spectrum), 99.5) or 1.0
    return np.clip(spectrum, -1.5, 1.5).astype(np.float32)


def identify(
    ppm: np.ndarray,
    intensity: np.ndarray,
    *,
    top_k: int = 5,
) -> List[Dict]:
    torch, _nn = _torch()
    model, checkpoint, device = _load_model()
    spectrum = _resample(ppm, intensity)
    with torch.no_grad():
        embedding = (
            model.embed(torch.from_numpy(spectrum[None, None, :]).to(device))
            .cpu()
            .numpy()[0]
        )
    references = np.asarray(checkpoint["reference_embeddings"], dtype=np.float32)
    similarities = references @ embedding
    order = np.argsort(similarities)[::-1][: max(1, top_k)]
    return [
        {
            "metabolite": checkpoint["labels"][int(index)],
            "cosine_similarity": round(float(similarities[index]), 4),
            "source": "self-supervised BMRB embedding",
        }
        for index in order
    ]


def benchmark_retrieval(augmentations_per_spectrum: int = 10, seed: int = 91) -> Dict:
    """
    Measure augmented-query retrieval against the open pure-compound references.

    This checks representation robustness, not performance on independent
    biological mixtures.
    """
    spectra, labels, ppm = _load_corpus()
    rng = np.random.default_rng(seed)
    top1 = 0
    top5 = 0
    reciprocal_ranks = []
    total = 0
    for index, label in enumerate(labels):
        for _ in range(max(1, augmentations_per_spectrum)):
            query = np.roll(spectra[index], int(rng.integers(-5, 6)))
            query = query * float(rng.uniform(0.8, 1.2))
            query = query + rng.normal(0, 0.005, len(query))
            matches = identify(ppm, query, top_k=5)
            names = [item["metabolite"] for item in matches]
            total += 1
            top1 += bool(names and names[0] == label)
            if label in names:
                rank = names.index(label) + 1
                top5 += 1
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
    return {
        "queries": total,
        "top1_accuracy": round(top1 / total, 4),
        "top5_accuracy": round(top5 / total, 4),
        "mean_reciprocal_rank": round(float(np.mean(reciprocal_ranks)), 4),
        "warning": (
            "Augmented retrieval against the same open reference collection; "
            "not an external mixture-identification estimate."
        ),
    }


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
