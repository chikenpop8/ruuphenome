"""
Reproducible synthetic stress benchmark for the Domain 1 processor.

This compares the original prototype algorithm with the upgraded processor on
spectra containing phase error, receiver delay, DC offset and complex noise.
It measures peak precision/recall/F1 against known injected resonances.

Run:
    python -m backend.nmr_api.benchmark_domain1 --spectra 30
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d
from scipy.signal import find_peaks

from . import signal_processing


TRUTH_SHIFTS = np.array(
    [0.94, 0.99, 1.33, 1.48, 1.92, 2.37, 2.54, 3.14, 3.36, 3.55, 6.89, 7.18, 7.31, 7.42, 8.12]
)
SF_MHZ = 600.0
SW_HZ = 7200.0


def simulate(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    size = 8192
    time = np.arange(size) / SW_HZ
    fid = np.zeros(size, dtype=complex)
    for shift in TRUTH_SHIFTS:
        amplitude = rng.lognormal(0, 0.55)
        frequency = (4.7 - shift) * SF_MHZ
        fid += (
            amplitude
            * np.exp(2j * np.pi * frequency * time)
            * np.exp(-time * rng.uniform(1.8, 4.0))
        )
    fid *= np.exp(1j * np.deg2rad(rng.uniform(-100, 100)))
    fid = np.roll(fid, rng.integers(0, 5))  # receiver-delay / first-order phase
    scale = np.max(np.abs(fid))
    fid += (rng.normal(0, 0.018, size) + 1j * rng.normal(0, 0.018, size)) * scale
    fid += (rng.normal(0, 0.002) + 1j * rng.normal(0, 0.002)) * scale
    return fid


def legacy_peak_positions(fid: np.ndarray) -> np.ndarray:
    """The original prototype's processing, retained only for comparison."""
    fid = signal_processing.apodize(fid, 0.3, SW_HZ)
    spectrum = np.fft.fftshift(np.fft.fft(fid, len(fid) * 2))
    best, best_score = spectrum, -np.inf
    for phase in np.linspace(-np.pi, np.pi, 72):
        candidate = spectrum * np.exp(1j * phase)
        score = np.real(candidate).sum()
        if score > best_score:
            best, best_score = candidate, score
    intensity = np.real(best)
    baseline = minimum_filter1d(intensity, 101, mode="nearest")
    baseline = uniform_filter1d(baseline, 101, mode="nearest")
    intensity -= baseline
    ppm = np.linspace(10.7, -1.3, len(intensity))
    mask = (ppm >= 0) & (ppm <= 10)
    ppm, intensity = ppm[mask], intensity[mask]
    intensity /= np.max(np.abs(intensity)) or 1.0
    noise = np.std(intensity[-max(1, len(intensity) // 20) :]) or 1e-9
    indices, _properties = find_peaks(intensity, height=5 * noise, distance=3)
    return ppm[indices]


def peak_metrics(observed: np.ndarray, tolerance_ppm: float = 0.01) -> np.ndarray:
    matched_truth = (
        sum(np.min(np.abs(observed - shift)) <= tolerance_ppm for shift in TRUTH_SHIFTS)
        if len(observed)
        else 0
    )
    matched_observed = (
        sum(np.min(np.abs(TRUTH_SHIFTS - shift)) <= tolerance_ppm for shift in observed)
        if len(observed)
        else 0
    )
    recall = matched_truth / len(TRUTH_SHIFTS)
    precision = matched_observed / len(observed) if len(observed) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return np.array([precision, recall, f1, len(observed)], dtype=float)


def benchmark(spectra: int = 30) -> dict:
    legacy, upgraded = [], []
    for seed in range(spectra):
        fid = simulate(seed)
        legacy.append(peak_metrics(legacy_peak_positions(fid)))
        result = signal_processing.process_fid(fid, SW_HZ, SF_MHZ, carrier_ppm=4.7)
        observed = np.array(
            [peak["ppm"] for peak in result["peaks"] if not peak.get("artifact")]
        )
        upgraded.append(peak_metrics(observed))
    old = np.mean(legacy, axis=0)
    new = np.mean(upgraded, axis=0)
    return {
        "spectra": spectra,
        "legacy": {
            "precision": round(float(old[0]), 3),
            "recall": round(float(old[1]), 3),
            "f1": round(float(old[2]), 3),
            "mean_peak_calls": round(float(old[3]), 1),
        },
        "upgraded": {
            "precision": round(float(new[0]), 3),
            "recall": round(float(new[1]), 3),
            "f1": round(float(new[2]), 3),
            "mean_peak_calls": round(float(new[3]), 1),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--spectra", type=int, default=30)
    args = parser.parse_args()
    print(benchmark(max(1, args.spectra)))
