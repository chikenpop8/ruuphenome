"""
Optional NMRformer integration contract.

The published NMRformer implementation/weights are not bundled with this
project. To avoid fabricating support, a local adapter must be supplied through
``NMRFORMER_ADAPTER_MODULE``. That module must expose:

    predict_assignments(ppm, intensity, peaks) -> list[dict]

Each result must contain ``metabolite`` and a confidence/probability in 0..1 or
0..100. This file validates the output and combines it with the transparent
chemical-shift pattern matcher.
"""

from __future__ import annotations

import importlib
import os
from typing import Dict, List, Sequence

import numpy as np


ADAPTER_ENV = "NMRFORMER_ADAPTER_MODULE"


def status() -> Dict:
    module_name = os.environ.get(ADAPTER_ENV, "").strip()
    if not module_name:
        return {
            "available": False,
            "adapter_module": None,
            "reason": f"Set {ADAPTER_ENV} to a validated local inference adapter.",
        }
    try:
        module = importlib.import_module(module_name)
        callable_found = callable(getattr(module, "predict_assignments", None))
        return {
            "available": callable_found,
            "adapter_module": module_name,
            "reason": None if callable_found else "Adapter lacks predict_assignments().",
        }
    except Exception as exc:
        return {
            "available": False,
            "adapter_module": module_name,
            "reason": f"Adapter import failed: {exc}",
        }


def predict(
    ppm: np.ndarray, intensity: np.ndarray, peaks: Sequence[Dict]
) -> List[Dict]:
    info = status()
    if not info["available"]:
        return []
    module = importlib.import_module(info["adapter_module"])
    raw = module.predict_assignments(
        ppm=np.asarray(ppm, dtype=float),
        intensity=np.asarray(intensity, dtype=float),
        peaks=list(peaks),
    )
    if not isinstance(raw, list):
        raise ValueError("NMRformer adapter must return a list of assignments.")
    normalized = []
    for item in raw:
        if not isinstance(item, dict) or not item.get("metabolite"):
            continue
        confidence = item.get("confidence", item.get("probability", 0))
        confidence = float(confidence)
        if confidence <= 1:
            confidence *= 100
        normalized.append(
            {
                "metabolite": str(item["metabolite"]),
                "confidence": round(float(np.clip(confidence, 0, 100)), 1),
                "peak_indices": item.get("peak_indices", []),
                "source": "nmrformer",
            }
        )
    return normalized


def hybridize(pattern_assignments: Sequence[Dict], neural_assignments: Sequence[Dict]) -> List[Dict]:
    """
    Combine explainable pattern evidence with NMRformer support.

    Pattern evidence remains dominant until the neural backend is independently
    validated on the target sample type.
    """
    neural = {item["metabolite"].casefold(): item for item in neural_assignments}
    combined = []
    used = set()
    for pattern in pattern_assignments:
        item = dict(pattern)
        key = item["metabolite"].casefold()
        support = neural.get(key)
        if support:
            used.add(key)
            confidence = 0.65 * float(item["confidence"]) + 0.35 * float(
                support["confidence"]
            )
            item["pattern_confidence"] = item["confidence"]
            item["nmrformer_confidence"] = support["confidence"]
            item["confidence"] = round(confidence, 1)
            item["neural_support"] = True
        else:
            item["nmrformer_confidence"] = None
            item["neural_support"] = False
        item["confidence_label"] = (
            "high"
            if item["confidence"] >= 75
            else "medium"
            if item["confidence"] >= 50
            else "tentative"
        )
        combined.append(item)

    # Neural-only calls remain tentative and require strong model confidence.
    for key, support in neural.items():
        if key in used or support["confidence"] < 80:
            continue
        combined.append(
            {
                "metabolite": support["metabolite"],
                "matched_peaks": [],
                "matched_count": 0,
                "expected_count": 0,
                "coverage": None,
                "mean_ppm_error": None,
                "median_snr": None,
                "ambiguity": None,
                "pattern_confidence": None,
                "nmrformer_confidence": support["confidence"],
                "neural_support": True,
                "confidence": round(0.35 * support["confidence"], 1),
                "confidence_label": "tentative",
                "source": "nmrformer-only",
            }
        )
    combined.sort(key=lambda item: -float(item["confidence"]))
    return combined
