"""Shared helpers for daily motif-discovery notebooks.

The functions in this module keep notebook logic focused on methodology and
interpretation while centralizing data discovery, daily aggregation, feature
engineering, motif extraction, and plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TIME_COLUMN_CANDIDATES = ("timestamp", "date", "datetime", "open_time", "close_time")
SUPPORTED_EXTENSIONS = {".parquet", ".csv"}
DEFAULT_MOTIF_COLUMNS = [
    "rank",
    "profile_value",
    "query_index",
    "match_index",
    "motif_start",
    "motif_end",
    "nearest_neighbor_start",
    "nearest_neighbor_end",
    "window_size",
]
DEFAULT_DISCORD_COLUMNS = [
    "discord_rank",
    "discord_index",
    "discord_score",
    "discord_start",
    "discord_end",
    "window_size",
]


def resolve_project_root(start: Path | None = None) -> Path:
    """Walk upward until the thesis project root is found."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        if (candidate / "src").exists() and (candidate / "data").exists():
            return candidate
    raise FileNotFoundError("Could not locate the project root containing both src/ and data/.")


def _rank_candidate(path: Path) -> tuple[int, int, str]:
    path_string = str(path).lower()
    score = 0
    if "processed" in path_string:
        score += 100
    if "daily" in path_string or "1d" in path_string:
        score += 80
    if "1min" in path_string or "1m" in path_string:
        score += 40
    if path.suffix.lower() == ".parquet":
        score += 10
    return (-score, len(path_string), path_string)


def discover_input_candidates(project_root: Path, symbol: str) -> list[Path]:
    """Return ordered candidate paths for a symbol."""
    explicit_paths = [
        project_root / "data" / "processed" / "crypto" / "daily" / f"{symbol}_daily_processed.parquet",
        project_root / "data" / "processed" / "crypto" / "daily" / f"{symbol}_1d_processed.parquet",
        project_root / "data" / "processed" / "crypto" / "1d" / f"{symbol}_1d_processed.parquet",
        project_root / "data" / "processed" / "crypto" / "1min" / f"{symbol}_1m_processed.parquet",
        project_root / "data" / "raw" / "crypto" / "daily" / f"{symbol}_daily_raw.parquet",
        project_root / "data" / "raw" / "crypto" / "1d" / f"{symbol}_1d_raw.parquet",
        project_root / "data" / "raw" / "crypto" / "1min" / f"{symbol}_1m_raw.parquet",
        project_root / "data" / "processed" / "crypto" / "daily" / f"{symbol}_daily_processed.csv",
        project_root / "data" / "processed" / "crypto" / "daily" / f"{symbol}_1d_processed.csv",
        project_root / "data" / "processed" / "crypto" / "1d" / f"{symbol}_1d_processed.csv",
        project_root / "data" / "processed" / "crypto" / "1min" / f"{symbol}_1m_processed.csv",
        project_root / "data" / "raw" / "crypto" / "daily" / f"{symbol}_daily_raw.csv",
        project_root / "data" / "raw" / "crypto" / "1d" / f"{symbol}_1d_raw.csv",
        project_root / "data" / "raw" / "crypto" / "1min" / f"{symbol}_1m_raw.csv",
    ]
    candidates: dict[Path, None] = {path: None for path in explicit_paths if path.exists()}

    for search_root in (project_root / "data" / "processed", project_root / "data" / "raw"):
        if not search_root.exists():
            continue
        for path in search_root.rglob(f"*{symbol}*"):
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                candidates[path] = None

    return sorted(candidates.keys(), key=_rank_candidate)


def resolve_input_data(
    project_root: Path,
    preferred_symbol: str = "BTCUSDT",
    fallback_symbols: Sequence[str] | None = None,
) -> dict[str, object]:
    """Resolve the best available input dataset for daily motif discovery."""
    symbols = [preferred_symbol, *(fallback_symbols or ())]
    search_log: list[dict[str, object]] = []

    for symbol in symbols:
        candidates = discover_input_candidates(project_root, symbol=symbol)
        search_log.append(
            {
                "symbol": symbol,
                "candidate_count": len(candidates),
                "top_candidates": [str(path.relative_to(project_root)) for path in candidates[:5]],
            }
        )
        if candidates:
            return {
                "symbol": symbol,
                "path": candidates[0],
                "candidate_paths": candidates,
                "search_log": search_log,
            }

    raise FileNotFoundError(
        "No candidate market dataset was found for the requested symbols under data/processed or data/raw."
    )


def _coerce_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if non_null.empty:
            return pd.to_datetime(series, utc=True, errors="coerce")
        magnitude = float(non_null.abs().median())
        unit = "ms" if magnitude >= 1e11 else "s"
        return pd.to_datetime(series, utc=True, errors="coerce", unit=unit)
    return pd.to_datetime(series, utc=True, errors="coerce")


def read_market_data(path: Path) -> pd.DataFrame:
    """Load one market dataset and standardize columns."""
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

    frame = frame.rename(columns={column: str(column).strip().lower() for column in frame.columns}).copy()

    timestamp_column = next((column for column in TIME_COLUMN_CANDIDATES if column in frame.columns), None)
    if timestamp_column is None:
        raise KeyError(
            "No timestamp-like column was found. Expected one of "
            f"{TIME_COLUMN_CANDIDATES}, received {sorted(frame.columns)}."
        )
    if timestamp_column != "timestamp":
        frame = frame.rename(columns={timestamp_column: "timestamp"})

    frame["timestamp"] = _coerce_timestamp(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    required_columns = {"open", "high", "low", "close", "volume"}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise KeyError(f"Input data is missing OHLCV columns required for daily aggregation: {sorted(missing_columns)}")

    for column in ["open", "high", "low", "close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame


def infer_timestamp_frequency(frame: pd.DataFrame) -> dict[str, object]:
    """Infer the median time step of a dataset."""
    deltas = frame["timestamp"].sort_values().diff().dropna()
    if deltas.empty:
        return {
            "median_delta": pd.NaT,
            "frequency_label": "unknown",
            "is_daily_or_slower": False,
            "observed_intervals": 0,
        }

    median_delta = deltas.median()
    if median_delta >= pd.Timedelta(hours=20):
        label = "daily_or_slower"
    elif median_delta >= pd.Timedelta(minutes=55):
        label = "hourly_to_subdaily"
    elif median_delta >= pd.Timedelta(minutes=1):
        label = "intraday"
    else:
        label = "subminute"

    return {
        "median_delta": median_delta,
        "frequency_label": label,
        "is_daily_or_slower": label == "daily_or_slower",
        "observed_intervals": int(len(deltas)),
    }


def build_daily_ohlcv(source_frame: pd.DataFrame) -> pd.DataFrame:
    """Aggregate arbitrary OHLCV bars into daily OHLCV."""
    working = (
        source_frame[["timestamp", "open", "high", "low", "close", "volume"]]
        .dropna(subset=["timestamp"])
        .sort_values("timestamp")
        .set_index("timestamp")
    )
    daily = working.resample("1D").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    daily = daily.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return daily


def compute_daily_features(
    daily_ohlcv: pd.DataFrame,
    volatility_window: int = 21,
    volume_window: int = 21,
) -> pd.DataFrame:
    """Compute daily motif-discovery features from daily OHLCV."""
    if volatility_window < 2:
        raise ValueError("volatility_window must be at least 2.")
    if volume_window < 2:
        raise ValueError("volume_window must be at least 2.")

    frame = daily_ohlcv.copy().sort_values("timestamp").reset_index(drop=True)
    close_safe = frame["close"].replace(0, np.nan)
    volume_safe = frame["volume"].replace(0, np.nan)

    frame["log_return"] = np.log(close_safe).diff()
    frame["pct_return"] = close_safe.pct_change()
    frame["hl_range"] = (frame["high"] - frame["low"]) / close_safe
    frame["rolling_volatility"] = frame["log_return"].rolling(
        window=volatility_window,
        min_periods=volatility_window,
    ).std()

    volume_mean = volume_safe.rolling(window=volume_window, min_periods=volume_window).mean()
    volume_std = volume_safe.rolling(window=volume_window, min_periods=volume_window).std()
    frame["volume_zscore"] = (volume_safe - volume_mean) / volume_std.replace(0, np.nan)

    return frame


def apply_run_mode(
    frame: pd.DataFrame,
    run_mode: str = "quick_test",
    quick_test_days: int = 540,
) -> pd.DataFrame:
    """Apply notebook run-mode slicing to a time series frame."""
    if run_mode not in {"quick_test", "full_run"}:
        raise ValueError("run_mode must be either 'quick_test' or 'full_run'.")
    if run_mode == "full_run" or frame.empty:
        return frame.copy().reset_index(drop=True)

    cutoff = frame["timestamp"].max() - pd.Timedelta(days=quick_test_days - 1)
    return frame.loc[frame["timestamp"] >= cutoff].reset_index(drop=True)


def prepare_univariate_series(frame: pd.DataFrame, target_column: str) -> pd.DataFrame:
    """Create a clean univariate panel for matrix-profile computation."""
    required_columns = {"timestamp", "close", target_column}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise KeyError(f"Frame is missing columns required for univariate analysis: {sorted(missing_columns)}")

    prepared = (
        frame[["timestamp", "close", target_column]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[target_column])
        .reset_index(drop=True)
        .copy()
    )
    return prepared


def global_standardize(values: np.ndarray) -> np.ndarray:
    """Standardize one feature channel over the current analysis slice."""
    array = np.asarray(values, dtype=np.float64)
    mean_value = np.nanmean(array)
    std_value = np.nanstd(array)
    if not np.isfinite(std_value) or std_value == 0:
        return np.zeros_like(array, dtype=np.float64)
    return (array - mean_value) / std_value


def prepare_multivariate_panel(frame: pd.DataFrame, channels: Sequence[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Align multivariate channels and build the MSTUMP input matrix."""
    required_columns = {"timestamp", "close", *channels}
    missing_columns = required_columns.difference(frame.columns)
    if missing_columns:
        raise KeyError(f"Frame is missing columns required for multivariate analysis: {sorted(missing_columns)}")

    panel = (
        frame[["timestamp", "close", *channels]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=list(channels))
        .reset_index(drop=True)
        .copy()
    )
    matrix = np.vstack([global_standardize(panel[channel].to_numpy(dtype=np.float64)) for channel in channels])
    return panel, matrix


def _intervals_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    return max(start_a, start_b) < min(end_a, end_b)


def extract_top_non_overlapping_motifs(
    profile: np.ndarray,
    indices: np.ndarray,
    timestamps: pd.Series,
    window_size: int,
    top_k: int = 5,
) -> pd.DataFrame:
    """Extract the strongest non-overlapping motif pairs."""
    profile = np.asarray(profile, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    timestamps = pd.to_datetime(pd.Series(timestamps), utc=True)

    motifs: list[dict[str, object]] = []
    occupied_intervals: list[tuple[int, int]] = []

    for query_index in np.argsort(profile):
        profile_value = float(profile[query_index])
        if not np.isfinite(profile_value):
            continue

        match_index = int(indices[query_index])
        if match_index < 0:
            continue

        query_interval = (int(query_index), int(query_index) + int(window_size))
        match_interval = (match_index, match_index + int(window_size))
        if query_interval[1] > len(timestamps) or match_interval[1] > len(timestamps):
            continue
        if _intervals_overlap(*query_interval, *match_interval):
            continue
        if any(
            _intervals_overlap(*query_interval, *used_interval) or _intervals_overlap(*match_interval, *used_interval)
            for used_interval in occupied_intervals
        ):
            continue

        occupied_intervals.extend([query_interval, match_interval])
        motifs.append(
            {
                "rank": len(motifs) + 1,
                "profile_value": profile_value,
                "query_index": int(query_index),
                "match_index": int(match_index),
                "motif_start": timestamps.iloc[query_interval[0]],
                "motif_end": timestamps.iloc[query_interval[1] - 1],
                "nearest_neighbor_start": timestamps.iloc[match_interval[0]],
                "nearest_neighbor_end": timestamps.iloc[match_interval[1] - 1],
                "window_size": int(window_size),
            }
        )
        if len(motifs) >= top_k:
            break

    return pd.DataFrame(motifs, columns=DEFAULT_MOTIF_COLUMNS)


def extract_top_non_overlapping_discords(
    profile: np.ndarray,
    timestamps: pd.Series,
    window_size: int,
    top_k: int = 5,
) -> pd.DataFrame:
    """Extract the strongest non-overlapping discords."""
    profile = np.asarray(profile, dtype=np.float64)
    timestamps = pd.to_datetime(pd.Series(timestamps), utc=True)

    discords: list[dict[str, object]] = []
    occupied_intervals: list[tuple[int, int]] = []

    for discord_index in np.argsort(profile)[::-1]:
        discord_score = float(profile[discord_index])
        if not np.isfinite(discord_score):
            continue

        interval = (int(discord_index), int(discord_index) + int(window_size))
        if interval[1] > len(timestamps):
            continue
        if any(_intervals_overlap(*interval, *used_interval) for used_interval in occupied_intervals):
            continue

        occupied_intervals.append(interval)
        discords.append(
            {
                "discord_rank": len(discords) + 1,
                "discord_index": int(discord_index),
                "discord_score": discord_score,
                "discord_start": timestamps.iloc[interval[0]],
                "discord_end": timestamps.iloc[interval[1] - 1],
                "window_size": int(window_size),
            }
        )
        if len(discords) >= top_k:
            break

    return pd.DataFrame(discords, columns=DEFAULT_DISCORD_COLUMNS)


def summarize_motifs(motif_df: pd.DataFrame) -> pd.DataFrame:
    """Return the compact motif summary table used in the notebooks."""
    summary_columns = [
        "rank",
        "motif_start",
        "motif_end",
        "nearest_neighbor_start",
        "nearest_neighbor_end",
        "profile_value",
        "window_size",
    ]
    return motif_df.loc[:, summary_columns].copy()


def plot_series_over_time(
    frame: pd.DataFrame,
    value_column: str,
    title: str,
    ylabel: str,
    color: str = "#1d3557",
    figsize: tuple[float, float] = (14, 4.2),
):
    """Plot one daily series over time."""
    fig, axis = plt.subplots(figsize=figsize)
    axis.plot(frame["timestamp"], frame[value_column], color=color, linewidth=1.2)
    axis.set_title(title)
    axis.set_xlabel("Date")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.25)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig, axis


def plot_matrix_profile_with_motifs(
    profile: np.ndarray,
    timestamps: pd.Series,
    motif_df: pd.DataFrame,
    discord_df: pd.DataFrame | None = None,
    title: str = "Matrix Profile",
):
    """Plot a matrix profile with motif and optional discord markers."""
    profile = np.asarray(profile, dtype=np.float64)
    timestamps = pd.to_datetime(pd.Series(timestamps), utc=True)
    subsequence_timestamps = timestamps.iloc[: len(profile)]

    fig, axis = plt.subplots(figsize=(14, 4.4))
    axis.plot(subsequence_timestamps, profile, color="#1d3557", linewidth=1.1, label="Matrix profile")

    if not motif_df.empty:
        motif_times = subsequence_timestamps.iloc[motif_df["query_index"].to_numpy(dtype=int)]
        motif_values = profile[motif_df["query_index"].to_numpy(dtype=int)]
        axis.scatter(motif_times, motif_values, color="#2a9d8f", s=44, zorder=3, label="Selected motifs")

    if discord_df is not None and not discord_df.empty:
        discord_times = subsequence_timestamps.iloc[discord_df["discord_index"].to_numpy(dtype=int)]
        discord_values = profile[discord_df["discord_index"].to_numpy(dtype=int)]
        axis.scatter(discord_times, discord_values, color="#e63946", marker="x", s=58, zorder=4, label="Selected discords")

    axis.set_title(title)
    axis.set_xlabel("Subsequence start date")
    axis.set_ylabel("Profile value")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig, axis


def plot_univariate_windows(
    frame: pd.DataFrame,
    value_column: str,
    motif_df: pd.DataFrame,
    max_motifs: int = 3,
):
    """Highlight motif windows on the full univariate signal."""
    fig, axis = plt.subplots(figsize=(14, 4.6))
    axis.plot(frame["timestamp"], frame[value_column], color="#1d3557", linewidth=1.0, label=value_column)

    color_map = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, min(max_motifs, len(motif_df)))))
    for color, (_, motif_row) in zip(color_map, motif_df.head(max_motifs).iterrows()):
        axis.axvspan(motif_row["motif_start"], motif_row["motif_end"], color=color, alpha=0.22)
        axis.axvspan(motif_row["nearest_neighbor_start"], motif_row["nearest_neighbor_end"], color=color, alpha=0.10)

    axis.set_title(f"{value_column} with highlighted motif windows")
    axis.set_xlabel("Date")
    axis.set_ylabel(value_column)
    axis.grid(alpha=0.25)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig, axis


def plot_motif_pair_comparison(
    frame: pd.DataFrame,
    value_column: str,
    motif_row: pd.Series,
    window_size: int,
    title_prefix: str = "",
):
    """Plot one motif pair side by side on the original scale."""
    query_window = frame.iloc[int(motif_row["query_index"]) : int(motif_row["query_index"]) + int(window_size)]
    match_window = frame.iloc[int(motif_row["match_index"]) : int(motif_row["match_index"]) + int(window_size)]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.4), sharey=True)
    axes[0].plot(query_window["timestamp"], query_window[value_column], color="#1d3557", linewidth=1.3)
    axes[1].plot(match_window["timestamp"], match_window[value_column], color="#f4a261", linewidth=1.3)
    axes[0].set_title(f"{title_prefix}Motif window")
    axes[1].set_title(f"{title_prefix}Nearest-neighbor window")

    for axis in axes:
        axis.set_xlabel("Date")
        axis.set_ylabel(value_column)
        axis.grid(alpha=0.25)

    fig.autofmt_xdate()
    plt.tight_layout()
    return fig, axes


def plot_normalized_motif_overlay(
    frame: pd.DataFrame,
    value_column: str,
    motif_row: pd.Series,
    window_size: int,
    title_prefix: str = "",
):
    """Overlay one motif pair after within-window standardization."""
    query_values = frame.iloc[int(motif_row["query_index"]) : int(motif_row["query_index"]) + int(window_size)][value_column].to_numpy()
    match_values = frame.iloc[int(motif_row["match_index"]) : int(motif_row["match_index"]) + int(window_size)][value_column].to_numpy()

    def _zscore(values: np.ndarray) -> np.ndarray:
        std_value = np.nanstd(values)
        if not np.isfinite(std_value) or std_value == 0:
            return np.zeros_like(values, dtype=np.float64)
        return (values - np.nanmean(values)) / std_value

    relative_index = np.arange(int(window_size))
    fig, axis = plt.subplots(figsize=(12.5, 4.1))
    axis.plot(relative_index, _zscore(query_values), color="#1d3557", linewidth=1.5, label="Motif window")
    axis.plot(relative_index, _zscore(match_values), color="#f4a261", linewidth=1.5, label="Nearest-neighbor window")
    axis.set_title(f"{title_prefix}Normalized overlay")
    axis.set_xlabel("Relative day")
    axis.set_ylabel("Within-window z-score")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    plt.tight_layout()
    return fig, axis


def plot_discord_windows(
    frame: pd.DataFrame,
    value_column: str,
    discord_df: pd.DataFrame,
    max_discords: int = 1,
):
    """Highlight discord windows on the full univariate signal."""
    fig, axis = plt.subplots(figsize=(14, 4.6))
    axis.plot(frame["timestamp"], frame[value_column], color="#1d3557", linewidth=1.0, label=value_column)

    for _, discord_row in discord_df.head(max_discords).iterrows():
        axis.axvspan(discord_row["discord_start"], discord_row["discord_end"], color="#e63946", alpha=0.18)

    axis.set_title(f"{value_column} with highlighted discord windows")
    axis.set_xlabel("Date")
    axis.set_ylabel(value_column)
    axis.grid(alpha=0.25)
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig, axis


def plot_multichannel_overview(frame: pd.DataFrame, channels: Sequence[str]):
    """Plot each selected multivariate channel over time."""
    fig, axes = plt.subplots(len(channels), 1, figsize=(14, max(3.0 * len(channels), 6.0)), sharex=True)
    axes = np.atleast_1d(axes)
    colors = plt.cm.tab10(np.linspace(0.0, 1.0, len(channels)))

    for axis, color, channel in zip(axes, colors, channels):
        axis.plot(frame["timestamp"], frame[channel], color=color, linewidth=1.1)
        axis.set_title(channel)
        axis.set_ylabel(channel)
        axis.grid(alpha=0.22)

    axes[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig, axes


def plot_multichannel_windows(
    frame: pd.DataFrame,
    channels: Sequence[str],
    motif_df: pd.DataFrame,
    max_motifs: int = 3,
):
    """Highlight motif windows across all selected channels."""
    fig, axes = plt.subplots(len(channels), 1, figsize=(14, max(3.0 * len(channels), 6.0)), sharex=True)
    axes = np.atleast_1d(axes)
    color_map = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, min(max_motifs, len(motif_df)))))

    for axis, channel in zip(axes, channels):
        axis.plot(frame["timestamp"], frame[channel], color="#1d3557", linewidth=1.0)
        for color, (_, motif_row) in zip(color_map, motif_df.head(max_motifs).iterrows()):
            axis.axvspan(motif_row["motif_start"], motif_row["motif_end"], color=color, alpha=0.22)
            axis.axvspan(motif_row["nearest_neighbor_start"], motif_row["nearest_neighbor_end"], color=color, alpha=0.10)
        axis.set_ylabel(channel)
        axis.set_title(channel)
        axis.grid(alpha=0.22)

    axes[-1].set_xlabel("Date")
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig, axes


def plot_multichannel_motif_comparison(
    frame: pd.DataFrame,
    channels: Sequence[str],
    motif_row: pd.Series,
    window_size: int,
    normalized: bool = False,
):
    """Compare one multivariate motif pair across all channels."""
    n_channels = len(channels)
    if normalized:
        fig, axes = plt.subplots(n_channels, 1, figsize=(13.5, max(3.0 * n_channels, 6.0)), sharex=False)
        axes = np.atleast_1d(axes)
        relative_index = np.arange(int(window_size))

        for axis, channel in zip(axes, channels):
            query_values = frame.iloc[int(motif_row["query_index"]) : int(motif_row["query_index"]) + int(window_size)][channel].to_numpy()
            match_values = frame.iloc[int(motif_row["match_index"]) : int(motif_row["match_index"]) + int(window_size)][channel].to_numpy()
            axis.plot(relative_index, global_standardize(query_values), color="#1d3557", linewidth=1.4, label="Motif window")
            axis.plot(relative_index, global_standardize(match_values), color="#f4a261", linewidth=1.4, label="Nearest-neighbor window")
            axis.set_title(f"{channel} normalized overlay")
            axis.set_ylabel("z-score")
            axis.grid(alpha=0.22)

        axes[-1].set_xlabel("Relative day")
        axes[0].legend(loc="best")
        plt.tight_layout()
        return fig, axes

    fig, axes = plt.subplots(n_channels, 2, figsize=(15.5, max(3.0 * n_channels, 6.0)), sharex=False)
    axes = np.atleast_2d(axes)

    query_window = frame.iloc[int(motif_row["query_index"]) : int(motif_row["query_index"]) + int(window_size)]
    match_window = frame.iloc[int(motif_row["match_index"]) : int(motif_row["match_index"]) + int(window_size)]

    for row_index, channel in enumerate(channels):
        axes[row_index, 0].plot(query_window["timestamp"], query_window[channel], color="#1d3557", linewidth=1.25)
        axes[row_index, 1].plot(match_window["timestamp"], match_window[channel], color="#f4a261", linewidth=1.25)
        axes[row_index, 0].set_ylabel(channel)
        axes[row_index, 0].set_title(f"{channel} motif window")
        axes[row_index, 1].set_title(f"{channel} nearest-neighbor window")
        axes[row_index, 0].grid(alpha=0.22)
        axes[row_index, 1].grid(alpha=0.22)

    axes[-1, 0].set_xlabel("Date")
    axes[-1, 1].set_xlabel("Date")
    fig.autofmt_xdate()
    plt.tight_layout()
    return fig, axes
