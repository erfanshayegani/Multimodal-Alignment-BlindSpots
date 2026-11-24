"""Direction vectors and layer-wise geometry.

All vectors are arrays of shape ``(num_layers, hidden_dim)``; activation sets are
``(num_samples, num_layers, hidden_dim)``.

* refusal direction  = mean(harmful activations) - mean(harmless activations)
* attack direction   = mean(perturbed, successful prompts) - mean(same prompts, unperturbed)
"""

from __future__ import annotations

import numpy as np


def mean_difference(acts_a: np.ndarray, acts_b: np.ndarray) -> np.ndarray:
    return acts_a.mean(0) - acts_b.mean(0)


def layerwise_cosine(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    num = (v1 * v2).sum(-1)
    den = np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1)
    return num / den


def layerwise_angle(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Angle in degrees per layer."""
    return np.degrees(np.arccos(np.clip(layerwise_cosine(v1, v2), -1.0, 1.0)))


def layerwise_magnitude_gap(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    return np.abs(np.linalg.norm(v2, axis=-1) - np.linalg.norm(v1, axis=-1))


def mean_similarity_to_refusal(refusal: np.ndarray, vec: np.ndarray, layers: range | list[int] = range(11, 15)) -> float:
    """Mean cosine similarity between ``-refusal`` and ``vec`` over ``layers`` (0-indexed).

    Used by the compositionality check: a perturbation that pushes activations *against* the
    refusal direction has a positive similarity with ``-refusal``.
    """
    sims = layerwise_cosine(-refusal[list(layers)], vec[list(layers)])
    return float(np.mean(sims))


def similarity_report(vectors: dict[str, np.ndarray], target: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Cosine similarity, angle and magnitude gap of each vector against ``target``, per layer."""
    return {
        name: {
            "cosine": layerwise_cosine(v, target),
            "angle": layerwise_angle(v, target),
            "magnitude_gap": layerwise_magnitude_gap(v, target),
        }
        for name, v in vectors.items()
    }
