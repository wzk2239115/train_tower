from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from PIL import Image

from tower.schema import UnifiedSample
from tower.viz.data_stats import DatasetStats, ModalityBreakdown, StageDataSummary
from tower.viz.training_metrics import TrainingRun


def _setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (10, 5),
            "axes.grid": True,
            "grid.alpha": 0.3,
            "font.size": 11,
        }
    )


def plot_modality_breakdown(
    breakdown: ModalityBreakdown,
    *,
    title: str = "Modality breakdown",
    ax: plt.Axes | None = None,
) -> Figure:
    _setup_style()
    data = {k: v for k, v in breakdown.as_dict().items() if v > 0}
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    if not data:
        ax.text(0.5, 0.5, "No samples", ha="center", va="center")
        ax.set_title(title)
        return fig

    labels = list(data.keys())
    values = list(data.values())
    colors = plt.cm.Set2(np.linspace(0, 1, len(labels)))
    ax.barh(labels, values, color=colors)
    ax.set_xlabel("Sample count")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_role_distribution(
    summary: StageDataSummary,
    *,
    title: str = "Role distribution",
    ax: plt.Axes | None = None,
) -> Figure:
    _setup_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 4))
    else:
        fig = ax.figure

    roles = summary.role_counts
    if not roles:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.set_title(title)
        return fig

    labels = list(roles.keys())
    values = [roles[r] for r in labels]
    ax.bar(labels, values, color=plt.cm.Pastel1(np.linspace(0, 1, len(labels))))
    ax.set_ylabel("Samples")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    return fig


def plot_resolution_histogram(
    datasets: list[DatasetStats],
    *,
    title: str = "Image resolution",
    ax: plt.Axes | None = None,
) -> Figure:
    _setup_style()
    widths: list[int] = []
    heights: list[int] = []
    for ds in datasets:
        widths.extend(ds.widths)
        heights.extend(ds.heights)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4))
    else:
        fig = ax.figure

    if not widths:
        ax.text(0.5, 0.5, "No resolution metadata", ha="center", va="center")
        ax.set_title(title)
        return fig

    max_side = max(max(widths), max(heights))
    bins = min(30, max(5, int(math.sqrt(len(widths)))))
    ax.hist2d(widths, heights, bins=bins, cmap="Blues")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    ax.set_title(title)
    ax.set_xlim(0, max_side * 1.05)
    ax.set_ylim(0, max_side * 1.05)
    fig.tight_layout()
    return fig


def plot_stage_comparison(
    summaries: dict[str, StageDataSummary],
    *,
    metric: str = "total",
    title: str = "Samples per stage",
) -> Figure:
    _setup_style()
    fig, ax = plt.subplots(figsize=(10, 4))

    stages = list(summaries.keys())
    if metric == "valid":
        values = [s.valid_samples for s in summaries.values()]
        ylabel = "Valid samples"
    else:
        values = [s.total_samples for s in summaries.values()]
        ylabel = "Total samples"

    ax.bar(stages, values, color=plt.cm.tab10(np.linspace(0, 1, len(stages))))
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return fig


def plot_training_curves(
    runs: list[TrainingRun],
    *,
    metric: str = "loss",
    title: str | None = None,
    ax: plt.Axes | None = None,
) -> Figure:
    _setup_style()
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    if not runs:
        ax.text(0.5, 0.5, "No training runs found", ha="center", va="center")
        return fig

    for run in runs:
        if metric == "grad_norm":
            curve = run.grad_norm_curve
            ylabel = "Grad norm"
        elif metric == "lr":
            curve = run.lr_curve
            ylabel = "Learning rate"
        else:
            curve = run.loss_curve
            ylabel = "Loss"

        if not curve:
            continue
        steps, values = zip(*curve)
        label = run.stage or run.name
        ax.plot(steps, values, marker="o", markersize=3, label=label)

    ax.set_xlabel("Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"Training {metric}")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def _load_image(path: str, max_size: int = 256) -> np.ndarray | None:
    p = Path(path)
    if not p.is_file():
        return None
    img = Image.open(p).convert("RGB")
    img.thumbnail((max_size, max_size))
    return np.asarray(img)


def _audio_spectrogram_from_patches(patches: list[list[float]]) -> np.ndarray | None:
    if not patches:
        return None
    arr = np.asarray(patches, dtype=np.float32)
    if arr.ndim != 2:
        return None
    return arr.T


def plot_sample_grid(
    samples: list[UnifiedSample],
    *,
    n_cols: int = 2,
    max_samples: int = 4,
    title: str = "Sample preview",
) -> Figure:
    _setup_style()
    chosen = samples[:max_samples]
    if not chosen:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No samples to display", ha="center", va="center")
        ax.axis("off")
        return fig

    n = len(chosen)
    n_rows = math.ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = np.array([axes])
    elif n_cols == 1:
        axes = np.array([[ax] for ax in axes])

    for idx, sample in enumerate(chosen):
        r, c = divmod(idx, n_cols)
        ax = axes[r][c]
        ax.axis("off")

        images = [sample.image] if isinstance(sample.image, str) else list(sample.image)
        img_arr = _load_image(images[0]) if images else None

        if img_arr is not None:
            ax.imshow(img_arr)
        elif sample.audio_values:
            spec = _audio_spectrogram_from_patches(sample.audio_values)
            if spec is not None:
                ax.imshow(spec, aspect="auto", origin="lower", cmap="magma")
                ax.set_title("Audio patches", fontsize=9)
            else:
                ax.text(0.5, 0.5, "Audio (no viz)", ha="center", va="center")
        else:
            ax.text(0.5, 0.5, "No image/audio", ha="center", va="center")

        human = next((t.get("value", "") for t in sample.conversations if t.get("from") == "human"), "")
        gpt = next((t.get("value", "") for t in sample.conversations if t.get("from") == "gpt"), "")
        caption = gpt or human
        if len(caption) > 120:
            caption = caption[:117] + "..."
        meta = sample.meta.get("role") or sample.meta.get("dataset") or sample.id
        ax.set_title(f"{meta}\n{caption}", fontsize=9)

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        r, c = divmod(idx, n_cols)
        axes[r][c].axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    return fig


def plot_tower_exit_weights(
    exit_weights: dict[str, float],
    *,
    title: str = "Tower exit loss weights",
) -> Figure:
    _setup_style()
    fig, ax = plt.subplots(figsize=(9, 4))
    if not exit_weights:
        ax.text(0.5, 0.5, "No tower exits for this stage", ha="center", va="center")
        ax.set_title(title)
        return fig

    labels = list(exit_weights.keys())
    values = [exit_weights[k] for k in labels]
    colors = ["#4C72B0" if v > 0 else "#DDDDDD" for v in values]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Weight")
    ax.set_title(title)
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    fig.tight_layout()
    return fig
