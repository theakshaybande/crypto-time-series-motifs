from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PRIMARY = "#0f4c5c"
SECONDARY = "#e36414"
ACCENT = "#6a994e"
HIGHLIGHT = "#f4d35e"
ALERT = "#bc4749"


def plot_series(
    series: np.ndarray,
    title: str = "Time series",
    ax: plt.Axes | None = None,
    label: str | None = None,
    color: str = PRIMARY,
) -> plt.Figure:
    """Plot a univariate time series."""
    values = np.asarray(series, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3.5))
    else:
        fig = ax.figure

    ax.plot(np.arange(len(values)), values, color=color, linewidth=2.0, label=label)
    ax.set_title(title)
    ax.set_xlabel("time")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    if label:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_series_with_windows(
    series: np.ndarray,
    windows: list[tuple[int, int]],
    title: str = "Series with highlighted windows",
    labels: list[str] | None = None,
    colors: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Figure:
    """Plot a series and shade selected subsequences."""
    values = np.asarray(series, dtype=float)
    fig = plot_series(values, title=title, ax=ax)
    ax = fig.axes[0]

    if labels is None:
        labels = [f"window {idx + 1}" for idx in range(len(windows))]
    if colors is None:
        colors = [HIGHLIGHT, SECONDARY, ACCENT, ALERT]

    for idx, (window, label) in enumerate(zip(windows, labels)):
        start, end = window
        color = colors[idx % len(colors)]
        ax.axvspan(start, end, color=color, alpha=0.22, label=label)

    handles, legend_labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_motif_alignment(
    series: np.ndarray,
    start_a: int,
    start_b: int,
    window: int,
    title: str = "Motif alignment",
) -> plt.Figure:
    """Overlay two subsequences to compare their shapes directly."""
    values = np.asarray(series, dtype=float)
    subseq_a = values[start_a : start_a + window]
    subseq_b = values[start_b : start_b + window]

    fig, ax = plt.subplots(figsize=(10, 3.5))
    x = np.arange(window)
    ax.plot(x, subseq_a, linewidth=2.0, label=f"start={start_a}", color=PRIMARY)
    ax.plot(x, subseq_b, linewidth=2.0, label=f"start={start_b}", color=SECONDARY)
    ax.set_title(title)
    ax.set_xlabel("offset within subsequence")
    ax.set_ylabel("value")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def plot_distance_profile(
    distance_profile: np.ndarray,
    title: str = "Distance profile",
    ax: plt.Axes | None = None,
    color: str = SECONDARY,
) -> plt.Figure:
    """Plot a distance profile over candidate subsequence starts."""
    values = np.asarray(distance_profile, dtype=float)
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3.0))
    else:
        fig = ax.figure

    ax.plot(np.arange(len(values)), values, color=color, linewidth=2.0)
    ax.set_title(title)
    ax.set_xlabel("candidate subsequence start")
    ax.set_ylabel("distance")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_matrix_profile(
    series: np.ndarray,
    profile: np.ndarray,
    title: str = "Matrix profile",
) -> plt.Figure:
    """Show a series and its matrix profile together."""
    values = np.asarray(series, dtype=float)
    mp = np.asarray(profile, dtype=float)

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 6),
        gridspec_kw={"height_ratios": [2.0, 1.2]},
    )
    axes[0].plot(np.arange(len(values)), values, color=PRIMARY, linewidth=2.0)
    axes[0].set_title(title)
    axes[0].set_ylabel("value")
    axes[0].grid(alpha=0.25)

    axes[1].plot(np.arange(len(mp)), mp, color=SECONDARY, linewidth=2.0)
    axes[1].set_xlabel("subsequence start")
    axes[1].set_ylabel("profile")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    return fig


def plot_multivariate(
    channels: np.ndarray,
    labels: list[str] | None = None,
    title: str = "Multivariate time series",
) -> plt.Figure:
    """Plot each channel of a small multivariate series in its own axis."""
    values = np.asarray(channels, dtype=float)
    if values.ndim != 2:
        raise ValueError("channels must have shape (n_channels, n_timepoints)")

    n_channels = values.shape[0]
    labels = labels or [f"channel {idx + 1}" for idx in range(n_channels)]
    fig, axes = plt.subplots(n_channels, 1, figsize=(12, 2.8 * n_channels), sharex=True)

    if n_channels == 1:
        axes = [axes]

    colors = [PRIMARY, SECONDARY, ACCENT, ALERT]
    for idx, ax in enumerate(axes):
        ax.plot(values[idx], color=colors[idx % len(colors)], linewidth=2.0)
        ax.set_ylabel(labels[idx])
        ax.grid(alpha=0.25)

    axes[0].set_title(title)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    return fig


def plot_feature_grid(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    title: str = "Feature views",
) -> plt.Figure:
    """Plot several feature series stacked vertically."""
    selected_columns = columns or list(frame.columns)
    n_rows = len(selected_columns)
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.8 * n_rows), sharex=True)

    if n_rows == 1:
        axes = [axes]

    colors = [PRIMARY, SECONDARY, ACCENT, ALERT]
    for idx, (ax, column) in enumerate(zip(axes, selected_columns)):
        ax.plot(frame[column].to_numpy(), color=colors[idx % len(colors)], linewidth=2.0)
        ax.set_ylabel(column)
        ax.grid(alpha=0.25)

    axes[0].set_title(title)
    axes[-1].set_xlabel("time")
    fig.tight_layout()
    return fig
