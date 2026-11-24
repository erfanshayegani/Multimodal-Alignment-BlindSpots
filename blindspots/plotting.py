"""Figures: layer-wise PCA of activations and cosine-similarity curves."""

from __future__ import annotations

import math
import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

SIMILARITY_COLORS = ["red", "darkviolet", "lime", "darkgreen", "silver", "orange", "blue", "maroon", "black"]


def layerwise_pca(harmless: np.ndarray, harmful: np.ndarray, others: dict[str, np.ndarray], n_components: int = 2):
    """Fit PCA per layer on harmless+harmful activations and project everything.

    Returns ``{name: array (num_layers, num_samples, n_components)}`` incl. ``harmless`` / ``harmful``.
    """
    sets = {"harmless": harmless, "harmful": harmful, **others}
    out = {k: [] for k in sets}
    for layer in range(harmless.shape[1]):
        pca = PCA(n_components=n_components).fit(np.concatenate([harmless[:, layer], harmful[:, layer]]))
        for k, acts in sets.items():
            out[k].append(pca.transform(acts[:, layer]))
    return {k: np.stack(v) for k, v in out.items()}


def plot_pca_grid(projected: dict[str, np.ndarray], save_path: str | None = None, cols: int = 8):
    """All layers in a grid; harmless/harmful baselines plus each perturbed set."""
    num_layers = projected["harmless"].shape[0]
    rows = math.ceil(num_layers / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 7.5, rows * 7.5), constrained_layout=True, squeeze=False)
    extra = [k for k in projected if k not in ("harmless", "harmful")]
    palette = plt.cm.tab10.colors
    for layer in range(rows * cols):
        ax = axes[divmod(layer, cols)]
        if layer >= num_layers:
            ax.axis("off")
            continue
        ax.scatter(*projected["harmless"][layer, :, :2].T, label="Harmless", s=80, color="darkgreen")
        ax.scatter(*projected["harmful"][layer, :, :2].T, label="Harmful", s=80, color="maroon")
        for i, k in enumerate(extra):
            ax.scatter(*projected[k][layer, :, :2].T, label=k, s=80, marker="*", color=palette[(i + 1) % 10])
        ax.set_title(f"Layer {layer + 1}", fontsize=32)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=len(labels), markerscale=5, prop={"size": 40})
    _finish(fig, save_path)


def plot_pca_single_layer(projected: dict[str, np.ndarray], layer: int, save_path: str | None = None):
    """One panel per perturbed set at a single layer (e.g. layer 16 for Qwen in the paper)."""
    extra = [k for k in projected if k not in ("harmless", "harmful")]
    cols = 4
    rows = max(1, math.ceil(len(extra) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(24, 5 * rows), constrained_layout=True, squeeze=False)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i >= len(extra):
            ax.axis("off")
            continue
        h1 = ax.scatter(*projected["harmless"][layer, :, :2].T, s=80, color="darkgreen", marker="+")
        h2 = ax.scatter(*projected["harmful"][layer, :, :2].T, s=20, color="maroon", marker="o")
        h3 = ax.scatter(*projected[extra[i]][layer, :, :2].T, s=80, color="blue", marker="*")
        ax.set_title(extra[i], fontsize=19)
    fig.legend([h1, h2, h3], ["Harmless", "Harmful", "Setting"], loc="upper center",
               bbox_to_anchor=(0.5, 1.09), ncol=3, markerscale=3, prop={"size": 20})
    _finish(fig, save_path)


def plot_similarity_curves(similarities: dict[str, np.ndarray], save_path: str | None = None,
                           ylabel: str = "Cosine Similarity", title: str | None = None):
    """One line per vector: layer-wise similarity (or angle / magnitude) against a target."""
    fig = plt.figure(figsize=(10, 8))
    for i, (name, values) in enumerate(similarities.items()):
        plt.plot(range(1, len(values) + 1), values, color=SIMILARITY_COLORS[i % len(SIMILARITY_COLORS)], label=name)
    plt.tick_params(axis="both", labelsize=20)
    plt.xlabel("Layer", fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(range(1, len(next(iter(similarities.values()))) + 1, 5))
    if title:
        plt.title(title)
    plt.legend(fontsize=12)
    _finish(fig, save_path)


def _finish(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=100, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
