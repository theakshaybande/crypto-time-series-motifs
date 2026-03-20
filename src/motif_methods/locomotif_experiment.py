"""Reusable LoCoMotif pipeline for multivariate financial motif experiments.

If LoCoMotif is missing in the active environment, install it with:
    pip install dtai-locomotif
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from sklearn.preprocessing import StandardScaler

from src.utils.paths import INTERIM_DIR, PROJECT_ROOT, ensure_directories, processed_ohlcv_path

DEFAULT_LOCOMOTIF_PARAMS: dict[str, Any] = {
    "l_min": 30,
    "l_max": 120,
    "rho": 0.7,
    "overlap": 0.1,
    "warping": True,
    "nb": 5,
}

FEATURE_ALIASES: dict[str, list[str]] = {
    "log_return": ["log_return"],
    "vol_30m": ["vol_30m", "volatility_30m"],
    "vol_60m": ["vol_60m", "volatility_60m"],
    "vol_240m": ["vol_240m", "volatility_240m"],
}

DEFAULT_CONFIG: dict[str, Any] = {
    "dataset_path": processed_ohlcv_path(symbol="BTCUSDT", interval="1m"),
    "dataset_name": "BTCUSDT_1m_processed",
    "sample_name": "BTCUSDT_1m",
    "debug_mode": True,
    "debug_rows": 5_000,
    "include_log_volume": True,
    "locomotif_params": DEFAULT_LOCOMOTIF_PARAMS.copy(),
    "max_highlighted_sets": 3,
    "overlay_motif_set_id": 0,
    "overlay_feature": "log_return",
}


def build_argument_parser() -> argparse.ArgumentParser:
    """Build a minimal CLI around the reusable LoCoMotif helpers."""
    parser = argparse.ArgumentParser(description="Run the first thesis LoCoMotif experiment on processed BTC data.")
    parser.add_argument(
        "--dataset-path",
        default=str(DEFAULT_CONFIG["dataset_path"]),
        help="Processed parquet dataset to analyze.",
    )
    parser.add_argument(
        "--dataset-name",
        default=DEFAULT_CONFIG["dataset_name"],
        help="Stable dataset label stored in outputs.",
    )
    parser.add_argument(
        "--sample-name",
        default=DEFAULT_CONFIG["sample_name"],
        help="Sample label used in filenames and saved tables.",
    )
    parser.add_argument(
        "--debug-rows",
        type=int,
        default=DEFAULT_CONFIG["debug_rows"],
        help="Analyze only the first N cleaned rows. Ignored when --full is used.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run on all cleaned rows instead of a debug slice.",
    )
    parser.add_argument(
        "--skip-log-volume",
        action="store_true",
        help="Do not add log_volume even if a volume column is available.",
    )
    parser.add_argument("--l-min", type=int, default=DEFAULT_LOCOMOTIF_PARAMS["l_min"])
    parser.add_argument("--l-max", type=int, default=DEFAULT_LOCOMOTIF_PARAMS["l_max"])
    parser.add_argument("--rho", type=float, default=DEFAULT_LOCOMOTIF_PARAMS["rho"])
    parser.add_argument("--overlap", type=float, default=DEFAULT_LOCOMOTIF_PARAMS["overlap"])
    parser.add_argument(
        "--warping",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_LOCOMOTIF_PARAMS["warping"],
        help="Enable or disable time warping in LoCoMotif.",
    )
    parser.add_argument("--nb", type=int, default=DEFAULT_LOCOMOTIF_PARAMS["nb"])
    parser.add_argument(
        "--max-highlighted-sets",
        type=int,
        default=DEFAULT_CONFIG["max_highlighted_sets"],
        help="Maximum number of motif sets to highlight in the interval plot.",
    )
    return parser


def get_installed_version(package_name: str) -> str | None:
    """Return the installed package version, if available."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def format_decimal_tag(value: float) -> str:
    """Convert a decimal hyperparameter to a filename-safe compact tag."""
    return str(value).replace(".", "")


def sanitize_token(value: str) -> str:
    """Normalize a free-text label into a filesystem-friendly token."""
    return value.lower().replace(" ", "_")


def format_debug_mode_tag(debug_mode: bool, debug_rows: int | None) -> str:
    """Create a concise mode tag for output filenames."""
    if not debug_mode or not debug_rows:
        return "full"
    if debug_rows >= 1000 and debug_rows % 1000 == 0:
        return f"debug{debug_rows // 1000}k"
    return f"debug{debug_rows}"


def build_experiment_name(sample_name: str, params: dict[str, Any], debug_mode: bool, debug_rows: int | None) -> str:
    """Create a stable, descriptive experiment name for saved outputs."""
    sample_tag = sanitize_token(sample_name)
    rho_tag = format_decimal_tag(params["rho"])
    mode_tag = format_debug_mode_tag(debug_mode=debug_mode, debug_rows=debug_rows)
    return (
        f"{sample_tag}_locomotif_"
        f"l{params['l_min']}_{params['l_max']}_"
        f"rho{rho_tag}_{mode_tag}"
    )


def to_serializable(value: Any) -> Any:
    """Recursively convert numpy-heavy objects into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    return value


def truncate_repr(value: Any, limit: int = 300) -> str:
    """Create a short preview string for diagnostics and metadata."""
    text = repr(value)
    return text if len(text) <= limit else f"{text[:limit]}..."


def select_feature_columns(
    df: pd.DataFrame,
    feature_aliases: dict[str, list[str]] | None = None,
    include_log_volume: bool = True,
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Resolve canonical thesis feature names against actual dataset columns."""
    aliases = feature_aliases or FEATURE_ALIASES
    working_df = df.copy()
    resolved_sources: dict[str, str] = {}

    for canonical_name, candidate_sources in aliases.items():
        source_name = next((name for name in candidate_sources if name in working_df.columns), None)
        if source_name is None:
            raise ValueError(
                f"Required feature '{canonical_name}' is missing. "
                f"Checked columns: {candidate_sources}. Available columns: {list(working_df.columns)}"
            )
        resolved_sources[canonical_name] = source_name
        if canonical_name != source_name:
            working_df[canonical_name] = working_df[source_name]

    selected_features = list(aliases.keys())

    if include_log_volume:
        if "log_volume" in working_df.columns:
            resolved_sources["log_volume"] = "log_volume"
            selected_features.append("log_volume")
        elif "volume" in working_df.columns:
            working_df["log_volume"] = np.log1p(working_df["volume"])
            resolved_sources["log_volume"] = "volume"
            selected_features.append("log_volume")

    return working_df, selected_features, resolved_sources


def standardize_features(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    """Standardize each feature channel independently for multivariate motif search."""
    scaler = StandardScaler()
    Xz = scaler.fit_transform(X)
    return np.asarray(Xz, dtype=np.float64, order="C"), scaler


def load_and_prepare_data(
    dataset_path: str | Path,
    debug_rows: int | None = None,
    include_log_volume: bool = True,
    feature_aliases: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str], dict[str, Any]]:
    """Load processed parquet data and return cleaned matrices ready for LoCoMotif."""
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Processed parquet dataset not found: {path}")

    df = pd.read_parquet(path)
    rows_before_cleaning = len(df)

    if "timestamp" not in df.columns:
        raise ValueError("Expected a 'timestamp' column in the processed parquet dataset.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.sort_values("timestamp").reset_index(drop=True)

    working_df, feature_columns, feature_source_map = select_feature_columns(
        df=df,
        feature_aliases=feature_aliases,
        include_log_volume=include_log_volume,
    )

    selected_df = working_df[["timestamp", *feature_columns]].copy()
    cleaned_df = selected_df.dropna().reset_index(drop=True)
    rows_after_cleaning = len(cleaned_df)

    if rows_after_cleaning == 0:
        raise ValueError("No rows remain after dropping NaNs from the selected feature block.")

    debug_mode = debug_rows is not None and debug_rows > 0
    if debug_mode:
        analyzed_df = cleaned_df.iloc[:debug_rows].copy().reset_index(drop=True)
    else:
        analyzed_df = cleaned_df

    X = np.asarray(analyzed_df[feature_columns].to_numpy(copy=True), dtype=np.float64, order="C")
    Xz, scaler = standardize_features(X)

    diagnostics = {
        "dataset_path": str(path.resolve()),
        "rows_before_cleaning": rows_before_cleaning,
        "rows_after_cleaning": rows_after_cleaning,
        "rows_analyzed": len(analyzed_df),
        "debug_mode": debug_mode,
        "debug_rows_requested": debug_rows,
        "feature_columns": feature_columns,
        "feature_source_map": feature_source_map,
        "scaler_means": scaler.mean_.tolist(),
        "scaler_scales": scaler.scale_.tolist(),
    }

    print("Data preparation diagnostics")
    print(f"  dataset_path: {diagnostics['dataset_path']}")
    print(f"  rows_before_cleaning: {rows_before_cleaning:,}")
    print(f"  rows_after_cleaning: {rows_after_cleaning:,}")
    print(f"  rows_analyzed: {len(analyzed_df):,}")
    print(f"  feature_columns: {feature_columns}")
    print(f"  feature_source_map: {feature_source_map}")

    return analyzed_df, X, Xz, feature_columns, diagnostics


def inspect_motif_structure(motif_sets: Any) -> dict[str, Any]:
    """Capture the runtime structure of the LoCoMotif result for debugging."""
    if not isinstance(motif_sets, list):
        return {
            "container_type": type(motif_sets).__name__,
            "length": None,
            "first_item_type": None,
            "first_item_preview": truncate_repr(motif_sets),
        }

    if not motif_sets:
        return {
            "container_type": "list",
            "length": 0,
            "first_item_type": None,
            "first_item_preview": None,
        }

    return {
        "container_type": "list",
        "length": len(motif_sets),
        "first_item_type": type(motif_sets[0]).__name__,
        "first_item_preview": truncate_repr(motif_sets[0]),
    }


def run_locomotif(Xz: np.ndarray, params: dict[str, Any]) -> tuple[list[Any], dict[str, Any]]:
    """Run LoCoMotif with clear import and parameter diagnostics."""
    try:
        import locomotif.locomotif as locomotif
    except ImportError as exc:
        raise ImportError(
            "LoCoMotif is not installed in the active environment. "
            "Install it with `pip install dtai-locomotif`."
        ) from exc

    if Xz.ndim != 2:
        raise ValueError(f"Expected a 2D array of shape (n_timestamps, n_features). Received shape {Xz.shape}.")
    if Xz.dtype != np.float64:
        Xz = np.asarray(Xz, dtype=np.float64, order="C")

    required_params = {"l_min", "l_max", "rho", "overlap", "warping", "nb"}
    missing_params = required_params.difference(params)
    if missing_params:
        raise ValueError(f"Missing LoCoMotif parameters: {sorted(missing_params)}")
    if params["l_min"] <= 0:
        raise ValueError(f"Expected l_min to be positive but received {params['l_min']}.")
    if params["l_min"] >= params["l_max"]:
        raise ValueError(f"Expected l_min < l_max but received {params['l_min']} and {params['l_max']}.")
    if params["l_max"] > Xz.shape[0]:
        raise ValueError(
            f"Expected l_max <= n_timestamps but received l_max={params['l_max']} "
            f"for n_timestamps={Xz.shape[0]}."
        )
    if not 0.0 <= params["overlap"] <= 0.5:
        raise ValueError(f"Expected overlap to lie in [0.0, 0.5] but received {params['overlap']}.")
    if not 0.0 < params["rho"] <= 1.0:
        raise ValueError(f"Expected rho to lie in (0.0, 1.0] but received {params['rho']}.")
    if params["nb"] is not None and int(params["nb"]) <= 0:
        raise ValueError(f"Expected nb to be a positive integer or None but received {params['nb']}.")

    print("Running LoCoMotif")
    print(f"  array_shape: {Xz.shape}")
    print(f"  params: {params}")

    motif_sets = locomotif.apply_locomotif(
        Xz,
        l_min=params["l_min"],
        l_max=params["l_max"],
        rho=params["rho"],
        nb=params["nb"],
        overlap=params["overlap"],
        warping=params["warping"],
    )

    structure_info = inspect_motif_structure(motif_sets)
    print("LoCoMotif output inspection")
    print(f"  container_type: {structure_info['container_type']}")
    print(f"  length: {structure_info['length']}")
    print(f"  first_item_type: {structure_info['first_item_type']}")
    print(f"  first_item_preview: {structure_info['first_item_preview']}")

    return motif_sets, structure_info


def extract_segment_bounds(segment: Any, n_rows: int) -> tuple[int, int]:
    """Normalize a segment-like object into validated integer start/end bounds."""
    if isinstance(segment, dict):
        if {"start", "end"} <= set(segment):
            start_idx, end_idx = segment["start"], segment["end"]
        elif {"start_idx", "end_idx"} <= set(segment):
            start_idx, end_idx = segment["start_idx"], segment["end_idx"]
        elif {"b", "e"} <= set(segment):
            start_idx, end_idx = segment["b"], segment["e"]
        else:
            raise TypeError(
                "Unsupported segment dictionary structure. "
                f"Keys received: {sorted(segment.keys())}. Expected one of "
                "['start', 'end'], ['start_idx', 'end_idx'], or ['b', 'e']."
            )
    elif isinstance(segment, (tuple, list)) and len(segment) == 2:
        start_idx, end_idx = segment
    else:
        raise TypeError(
            "Unsupported segment object received from LoCoMotif. "
            f"Type: {type(segment).__name__}; value preview: {truncate_repr(segment)}"
        )

    start_idx = int(start_idx)
    end_idx = int(end_idx)

    if start_idx < 0 or end_idx < 0:
        raise ValueError(f"Segment indices must be non-negative. Received ({start_idx}, {end_idx}).")
    if end_idx <= start_idx:
        raise ValueError(
            "LoCoMotif segment end must be greater than start. "
            f"Received ({start_idx}, {end_idx})."
        )
    if end_idx > n_rows:
        raise ValueError(
            "LoCoMotif segment end exceeds the analyzed dataframe length. "
            f"Received end={end_idx} with n_rows={n_rows}."
        )

    return start_idx, end_idx


def normalize_motif_set_entry(entry: Any, n_rows: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    """Normalize one LoCoMotif motif-set entry into a candidate and list of occurrences.

    LoCoMotif 0.2.0 returns entries shaped like:
        ((candidate_start, candidate_end), [(start_1, end_1), (start_2, end_2), ...])

    If a future package version changes that structure, adjust this function first.
    """
    if isinstance(entry, dict):
        candidate = entry.get("candidate") or entry.get("representative")
        motif_set = entry.get("motif_set") or entry.get("occurrences") or entry.get("segments")
        if candidate is None or motif_set is None:
            raise TypeError(
                "Unsupported dictionary motif-set structure. "
                f"Keys received: {sorted(entry.keys())}."
            )
    elif isinstance(entry, (tuple, list)) and len(entry) == 2:
        candidate, motif_set = entry
    else:
        raise TypeError(
            "Unsupported motif-set entry returned by LoCoMotif. "
            f"Type: {type(entry).__name__}; value preview: {truncate_repr(entry)}"
        )

    candidate_bounds = extract_segment_bounds(candidate, n_rows=n_rows)
    if not isinstance(motif_set, (list, tuple)):
        raise TypeError(
            "Expected motif_set to be a list or tuple of segments. "
            f"Received {type(motif_set).__name__}."
        )

    occurrence_bounds = [extract_segment_bounds(segment, n_rows=n_rows) for segment in motif_set]
    return candidate_bounds, occurrence_bounds


def parse_locomotif_output(
    motif_sets: list[Any],
    analyzed_df: pd.DataFrame,
    feature_columns: list[str],
    params: dict[str, Any],
    dataset_name: str,
    sample_name: str,
) -> pd.DataFrame:
    """Flatten LoCoMotif motif sets into a thesis-ready occurrence table."""
    n_rows = len(analyzed_df)
    rows: list[dict[str, Any]] = []

    for motif_set_id, entry in enumerate(motif_sets):
        candidate_bounds, occurrence_bounds = normalize_motif_set_entry(entry, n_rows=n_rows)
        candidate_start_idx, candidate_end_idx = candidate_bounds
        candidate_start_time = analyzed_df.iloc[candidate_start_idx]["timestamp"]
        candidate_end_time = analyzed_df.iloc[candidate_end_idx - 1]["timestamp"]

        for occurrence_id, (start_idx, end_idx) in enumerate(occurrence_bounds):
            rows.append(
                {
                    "motif_set_id": motif_set_id,
                    "occurrence_id": occurrence_id,
                    "candidate_start_idx": candidate_start_idx,
                    "candidate_end_idx": candidate_end_idx,
                    "candidate_start_time": candidate_start_time,
                    "candidate_end_time": candidate_end_time,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "length": end_idx - start_idx,
                    "start_time": analyzed_df.iloc[start_idx]["timestamp"],
                    "end_time": analyzed_df.iloc[end_idx - 1]["timestamp"],
                    "l_min": params["l_min"],
                    "l_max": params["l_max"],
                    "rho": params["rho"],
                    "overlap": params["overlap"],
                    "warping": params["warping"],
                    "nb": params["nb"],
                    "feature_columns": json.dumps(feature_columns),
                    "sample_name": sample_name,
                    "dataset_name": dataset_name,
                }
            )

    occurrence_df = pd.DataFrame(rows)
    if occurrence_df.empty:
        return pd.DataFrame(
            columns=[
                "motif_set_id",
                "occurrence_id",
                "candidate_start_idx",
                "candidate_end_idx",
                "candidate_start_time",
                "candidate_end_time",
                "start_idx",
                "end_idx",
                "length",
                "start_time",
                "end_time",
                "l_min",
                "l_max",
                "rho",
                "overlap",
                "warping",
                "nb",
                "feature_columns",
                "sample_name",
                "dataset_name",
            ]
        )

    return occurrence_df.sort_values(["motif_set_id", "occurrence_id"]).reset_index(drop=True)


def summarize_motif_sets(
    occurrence_df: pd.DataFrame,
    n_rows_analyzed: int,
    n_channels: int,
) -> pd.DataFrame:
    """Create a compact run-level summary table."""
    if occurrence_df.empty:
        summary = {
            "total_motif_sets_found": 0,
            "total_motif_occurrences": 0,
            "average_motif_length": np.nan,
            "min_motif_length": np.nan,
            "max_motif_length": np.nan,
            "number_of_rows_analyzed": n_rows_analyzed,
            "number_of_channels_used": n_channels,
        }
    else:
        summary = {
            "total_motif_sets_found": int(occurrence_df["motif_set_id"].nunique()),
            "total_motif_occurrences": int(len(occurrence_df)),
            "average_motif_length": float(occurrence_df["length"].mean()),
            "min_motif_length": int(occurrence_df["length"].min()),
            "max_motif_length": int(occurrence_df["length"].max()),
            "number_of_rows_analyzed": n_rows_analyzed,
            "number_of_channels_used": n_channels,
        }
    return pd.DataFrame([summary])


def save_motif_results(
    motif_sets: list[Any],
    occurrence_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    metadata: dict[str, Any],
    experiment_name: str,
) -> dict[str, Path]:
    """Persist LoCoMotif outputs for later thesis analysis."""
    interim_dir = INTERIM_DIR / "locomotif"
    figure_dir = PROJECT_ROOT / "outputs" / "figures" / "locomotif"
    table_dir = PROJECT_ROOT / "outputs" / "tables" / "locomotif"
    ensure_directories(interim_dir, figure_dir, table_dir)

    raw_motif_path = interim_dir / f"{experiment_name}_motif_sets_raw.json"
    metadata_path = interim_dir / f"{experiment_name}_run_metadata.json"
    occurrences_csv_path = table_dir / f"{experiment_name}_occurrences.csv"
    occurrences_parquet_path = table_dir / f"{experiment_name}_occurrences.parquet"
    summary_csv_path = table_dir / f"{experiment_name}_summary.csv"

    raw_motif_path.write_text(json.dumps(to_serializable(motif_sets), indent=2), encoding="utf-8")
    metadata_path.write_text(json.dumps(to_serializable(metadata), indent=2), encoding="utf-8")
    occurrence_df.to_csv(occurrences_csv_path, index=False)
    occurrence_df.to_parquet(occurrences_parquet_path, index=False)
    summary_df.to_csv(summary_csv_path, index=False)

    return {
        "interim_dir": interim_dir,
        "figure_dir": figure_dir,
        "table_dir": table_dir,
        "raw_motif_path": raw_motif_path,
        "metadata_path": metadata_path,
        "occurrences_csv_path": occurrences_csv_path,
        "occurrences_parquet_path": occurrences_parquet_path,
        "summary_csv_path": summary_csv_path,
    }


def plot_motif_occurrences(
    analyzed_df: pd.DataFrame,
    occurrence_df: pd.DataFrame,
    feature_columns: list[str],
    figure_dir: str | Path,
    experiment_name: str,
    max_highlighted_sets: int = 3,
    overlay_motif_set_id: int | None = 0,
    overlay_feature: str | None = None,
) -> dict[str, Path]:
    """Create simple interval and aligned-occurrence diagnostic plots."""
    figure_dir = Path(figure_dir)
    ensure_directories(figure_dir)

    first_feature = feature_columns[0]
    interval_plot_path = figure_dir / f"{experiment_name}_intervals.png"
    overlay_plot_path = figure_dir / f"{experiment_name}_aligned_occurrences.png"

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(analyzed_df["timestamp"], analyzed_df[first_feature], linewidth=0.9, color="black")
    ax.set_title(f"{first_feature} with highlighted motif occurrences")
    ax.set_xlabel("Timestamp")
    ax.set_ylabel(first_feature)

    if not occurrence_df.empty:
        motif_set_ids = sorted(occurrence_df["motif_set_id"].unique())[:max_highlighted_sets]
        colors = plt.cm.tab10(np.linspace(0.0, 1.0, max(1, len(motif_set_ids))))
        legend_handles: list[Patch] = []
        for color, motif_set_id in zip(colors, motif_set_ids):
            subset = occurrence_df.loc[occurrence_df["motif_set_id"] == motif_set_id]
            for _, row in subset.iterrows():
                ax.axvspan(row["start_time"], row["end_time"], color=color, alpha=0.18)
            legend_handles.append(Patch(facecolor=color, alpha=0.25, label=f"motif_set_{motif_set_id}"))
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper right")
    else:
        ax.text(
            0.5,
            0.5,
            "No motif occurrences were found for this parameterization.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )

    fig.tight_layout()
    fig.savefig(interval_plot_path, dpi=150)
    plt.close(fig)

    overlay_feature_name = overlay_feature if overlay_feature in feature_columns else first_feature
    target_motif_set_id = overlay_motif_set_id
    if not occurrence_df.empty and target_motif_set_id not in set(occurrence_df["motif_set_id"]):
        target_motif_set_id = int(occurrence_df["motif_set_id"].min())

    fig, ax = plt.subplots(figsize=(12, 6))
    if occurrence_df.empty or target_motif_set_id is None:
        ax.text(
            0.5,
            0.5,
            "No motif set available for aligned-occurrence plotting.",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=11,
        )
    else:
        subset = occurrence_df.loc[occurrence_df["motif_set_id"] == target_motif_set_id].sort_values("occurrence_id")
        for _, row in subset.iterrows():
            segment = analyzed_df.iloc[row["start_idx"] : row["end_idx"]][overlay_feature_name].to_numpy()
            ax.plot(np.arange(len(segment)), segment, alpha=0.8, linewidth=1.1)
        ax.set_title(
            f"Aligned occurrences for motif_set_{target_motif_set_id} on {overlay_feature_name}"
        )
        ax.set_xlabel("Relative time step")
        ax.set_ylabel(overlay_feature_name)

    fig.tight_layout()
    fig.savefig(overlay_plot_path, dpi=150)
    plt.close(fig)

    return {
        "interval_plot_path": interval_plot_path,
        "overlay_plot_path": overlay_plot_path,
    }


def run_experiment(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Run the complete LoCoMotif experiment pipeline and return in-memory outputs."""
    run_config = DEFAULT_CONFIG.copy()
    if config:
        run_config.update(config)
    run_config["locomotif_params"] = {**DEFAULT_LOCOMOTIF_PARAMS, **run_config.get("locomotif_params", {})}

    dataset_path = Path(run_config["dataset_path"])
    debug_rows = None if not run_config["debug_mode"] else run_config["debug_rows"]
    params = run_config["locomotif_params"]
    experiment_name = build_experiment_name(
        sample_name=run_config["sample_name"],
        params=params,
        debug_mode=run_config["debug_mode"],
        debug_rows=debug_rows,
    )

    analyzed_df, X, Xz, feature_columns, diagnostics = load_and_prepare_data(
        dataset_path=dataset_path,
        debug_rows=debug_rows,
        include_log_volume=run_config["include_log_volume"],
    )
    motif_sets, structure_info = run_locomotif(Xz, params=params)
    occurrence_df = parse_locomotif_output(
        motif_sets=motif_sets,
        analyzed_df=analyzed_df,
        feature_columns=feature_columns,
        params=params,
        dataset_name=run_config["dataset_name"],
        sample_name=run_config["sample_name"],
    )
    summary_df = summarize_motif_sets(
        occurrence_df=occurrence_df,
        n_rows_analyzed=len(analyzed_df),
        n_channels=len(feature_columns),
    )

    metadata = {
        "experiment_name": experiment_name,
        "run_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": diagnostics["dataset_path"],
        "dataset_name": run_config["dataset_name"],
        "sample_name": run_config["sample_name"],
        "row_counts": {
            "before_cleaning": diagnostics["rows_before_cleaning"],
            "after_cleaning": diagnostics["rows_after_cleaning"],
            "analyzed": diagnostics["rows_analyzed"],
        },
        "selected_features": feature_columns,
        "feature_source_map": diagnostics["feature_source_map"],
        "parameters": params,
        "debug_slice_info": {
            "debug_mode": diagnostics["debug_mode"],
            "debug_rows_requested": diagnostics["debug_rows_requested"],
        },
        "motif_output_structure": structure_info,
        "index_convention": "start_idx is inclusive and end_idx is exclusive, matching LoCoMotif segment tuples.",
        "package_versions": {
            "dtai-locomotif": get_installed_version("dtai-locomotif"),
            "scikit-learn": get_installed_version("scikit-learn"),
        },
        "feature_matrix_shape": list(X.shape),
        "standardized_matrix_shape": list(Xz.shape),
        "summary": summary_df.iloc[0].to_dict(),
    }

    saved_paths = save_motif_results(
        motif_sets=motif_sets,
        occurrence_df=occurrence_df,
        summary_df=summary_df,
        metadata=metadata,
        experiment_name=experiment_name,
    )
    plot_paths = plot_motif_occurrences(
        analyzed_df=analyzed_df,
        occurrence_df=occurrence_df,
        feature_columns=feature_columns,
        figure_dir=saved_paths["figure_dir"],
        experiment_name=experiment_name,
        max_highlighted_sets=run_config["max_highlighted_sets"],
        overlay_motif_set_id=run_config["overlay_motif_set_id"],
        overlay_feature=run_config["overlay_feature"],
    )

    print("Run summary")
    print(summary_df.to_string(index=False))
    print("Saved outputs")
    for key, value in {**saved_paths, **plot_paths}.items():
        print(f"  {key}: {value}")

    return {
        "config": run_config,
        "experiment_name": experiment_name,
        "analyzed_df": analyzed_df,
        "feature_columns": feature_columns,
        "motif_sets": motif_sets,
        "occurrence_df": occurrence_df,
        "summary_df": summary_df,
        "metadata": metadata,
        "saved_paths": {**saved_paths, **plot_paths},
    }


def config_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Translate CLI arguments into the reusable run_experiment config format."""
    debug_mode = (not args.full) and (args.debug_rows is not None) and (args.debug_rows > 0)
    debug_rows = args.debug_rows if debug_mode else None
    return {
        "dataset_path": Path(args.dataset_path),
        "dataset_name": args.dataset_name,
        "sample_name": args.sample_name,
        "debug_mode": debug_mode,
        "debug_rows": debug_rows,
        "include_log_volume": not args.skip_log_volume,
        "locomotif_params": {
            "l_min": args.l_min,
            "l_max": args.l_max,
            "rho": args.rho,
            "overlap": args.overlap,
            "warping": args.warping,
            "nb": args.nb,
        },
        "max_highlighted_sets": args.max_highlighted_sets,
        "overlay_motif_set_id": 0,
        "overlay_feature": "log_return",
    }


def main() -> None:
    """CLI entry point."""
    parser = build_argument_parser()
    args = parser.parse_args()
    run_experiment(config_from_args(args))


if __name__ == "__main__":
    main()
