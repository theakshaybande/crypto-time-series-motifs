"""Minimal utilities for crypto data engineering notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

EXPECTED_FREQUENCY = pd.Timedelta(minutes=1)


def resolve_project_root(start: Path | None = None) -> Path:
    """Resolve the project root by walking upward until `src/` is found."""
    current = (start or Path.cwd()).resolve()
    while current != current.parent and not (current / "src").exists():
        current = current.parent
    if not (current / "src").exists():
        raise FileNotFoundError("Could not locate project root containing src/.")
    return current


def load_asset_bundle(project_root: Path, symbol: str) -> dict[str, Any]:
    """Load processed data and metadata diagnostics for one symbol."""
    processed_path = project_root / "data" / "processed" / "crypto" / "1min" / f"{symbol}_1m_processed.parquet"
    missing_csv_path = project_root / "data" / "metadata" / f"{symbol}_1m_missing_timestamps.csv"
    report_json_path = project_root / "data" / "metadata" / f"{symbol}_1m_data_report.json"

    df = pd.read_parquet(processed_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    missing_df = pd.read_csv(missing_csv_path)
    if not missing_df.empty:
        missing_df["timestamp"] = pd.to_datetime(missing_df["timestamp"], utc=True)

    with report_json_path.open("r", encoding="utf-8") as fp:
        report = json.load(fp)

    return {
        "symbol": symbol,
        "df": df,
        "missing_df": missing_df,
        "report": report,
        "processed_path": processed_path,
        "missing_csv_path": missing_csv_path,
        "report_json_path": report_json_path,
    }


def leading_nan_count(series: pd.Series) -> int:
    """Count leading NaNs before the first valid value."""
    valid = series.notna().to_numpy()
    if valid.size == 0 or not valid.any():
        return int(len(series))
    return int(np.argmax(valid))


def build_gap_run_table(missing_df: pd.DataFrame) -> pd.DataFrame:
    """Compress minute-level missing timestamps into contiguous gap runs."""
    if missing_df.empty:
        return pd.DataFrame(columns=["gap_start", "gap_end", "missing_minutes"])

    gap_df = missing_df.copy()
    gap_df["timestamp"] = pd.to_datetime(gap_df["timestamp"], utc=True)
    gap_df = gap_df.sort_values("timestamp").reset_index(drop=True)
    gap_df["new_gap"] = gap_df["timestamp"].diff().ne(EXPECTED_FREQUENCY).fillna(True)
    gap_df["gap_id"] = gap_df["new_gap"].cumsum()
    return (
        gap_df.groupby("gap_id")
        .agg(gap_start=("timestamp", "min"), gap_end=("timestamp", "max"), missing_minutes=("timestamp", "size"))
        .reset_index(drop=True)
    )


def build_contiguous_block_table(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Summarize contiguous observed blocks for a motif candidate feature."""
    candidate = df.loc[df[value_column].notna(), ["timestamp", value_column]].copy()
    if candidate.empty:
        return pd.DataFrame(columns=["block_id", "start", "end", "rows"])

    candidate["timestamp"] = pd.to_datetime(candidate["timestamp"], utc=True)
    candidate = candidate.sort_values("timestamp").reset_index(drop=True)
    candidate["new_block"] = candidate["timestamp"].diff().ne(EXPECTED_FREQUENCY).fillna(True)
    candidate["block_id"] = candidate["new_block"].cumsum()
    return (
        candidate.groupby("block_id")
        .agg(start=("timestamp", "min"), end=("timestamp", "max"), rows=("timestamp", "size"))
        .reset_index()
        .sort_values(["rows", "start"], ascending=[False, True])
        .reset_index(drop=True)
    )


def build_summary_table(asset_bundles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a compact cross-asset summary table."""
    rows: list[dict[str, Any]] = []
    for symbol, bundle in asset_bundles.items():
        df = bundle["df"]
        report = bundle["report"]
        expected_rows = int(((df["timestamp"].max() - df["timestamp"].min()) / EXPECTED_FREQUENCY) + 1)
        rows.append(
            {
                "symbol": symbol,
                "rows": len(df),
                "start_utc": df["timestamp"].min(),
                "end_utc": df["timestamp"].max(),
                "expected_rows_if_continuous": expected_rows,
                "coverage_ratio": len(df) / expected_rows,
                "duplicate_timestamps": report["duplicate_timestamps"],
                "missing_intervals": report["missing_intervals_count"],
                "irregular_intervals": report["irregular_intervals_count"],
            }
        )
    return pd.DataFrame(rows).sort_values("symbol").reset_index(drop=True)


def build_dtype_table(asset_bundles: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Build a column-by-column dtype comparison table."""
    dtype_rows: list[dict[str, str]] = []
    for symbol, bundle in asset_bundles.items():
        for column, dtype in bundle["df"].dtypes.items():
            dtype_rows.append({"symbol": symbol, "column": column, "dtype": str(dtype)})
    return pd.DataFrame(dtype_rows).pivot(index="column", columns="symbol", values="dtype")


def build_missingness_table(asset_bundles: dict[str, dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    """Build total, leading, and internal NaN diagnostics for selected columns."""
    rows: list[dict[str, Any]] = []
    for symbol, bundle in asset_bundles.items():
        df = bundle["df"]
        for column in columns:
            total_nan = int(df[column].isna().sum())
            leading_nan = leading_nan_count(df[column])
            rows.append(
                {
                    "symbol": symbol,
                    "column": column,
                    "total_nan": total_nan,
                    "leading_nan": leading_nan,
                    "internal_nan": total_nan - leading_nan,
                }
            )
    return pd.DataFrame(rows)


def make_recent_window(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Return the trailing observation window in days."""
    cutoff = df["timestamp"].max() - pd.Timedelta(days=days)
    return df.loc[df["timestamp"] >= cutoff].copy()


def make_reindexed_diagnostic_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex to a full minute grid for diagnostics only."""
    full_index = pd.date_range(df["timestamp"].min(), df["timestamp"].max(), freq="1min", tz="UTC")
    diagnostic = df.set_index("timestamp").reindex(full_index)
    diagnostic.index.name = "timestamp"
    diagnostic = diagnostic.reset_index()
    diagnostic["observed_bar"] = diagnostic["close"].notna()
    diagnostic["synthetic_gap"] = ~diagnostic["observed_bar"]
    diagnostic["close_ffill_for_plot"] = diagnostic["close"].ffill()
    return diagnostic

