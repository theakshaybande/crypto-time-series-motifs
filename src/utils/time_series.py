"""Time-series schema checks and diagnostics for OHLCV data."""

from __future__ import annotations

from typing import Any

import pandas as pd

CANONICAL_OHLCV_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]
NUMERIC_COLUMNS = ["open", "high", "low", "close", "volume"]


def enforce_canonical_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure required OHLCV columns exist and return canonical column order."""
    missing = [col for col in CANONICAL_OHLCV_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.loc[:, CANONICAL_OHLCV_COLUMNS].copy()


def standardize_ohlcv_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce schema to canonical dtypes and enforce monotonic UTC timestamps."""
    standardized = enforce_canonical_columns(df)

    standardized["timestamp"] = pd.to_datetime(standardized["timestamp"], utc=True, errors="coerce")
    if standardized["timestamp"].isna().any():
        invalid_count = int(standardized["timestamp"].isna().sum())
        raise ValueError(f"Found {invalid_count} invalid timestamps after UTC coercion.")

    for col in NUMERIC_COLUMNS:
        standardized[col] = pd.to_numeric(standardized[col], errors="coerce")
        if standardized[col].isna().any():
            invalid_count = int(standardized[col].isna().sum())
            raise ValueError(f"Found {invalid_count} invalid numeric values in column '{col}'.")

    standardized = (
        standardized.sort_values("timestamp")
        .drop_duplicates(subset=["timestamp"], keep="last")
        .reset_index(drop=True)
    )

    return standardized


def infer_missing_timestamps(timestamp_series: pd.Series, freq: str = "1min") -> pd.DataFrame:
    """Infer missing timestamps from a regular time grid between min and max timestamps."""
    ts = pd.Series(timestamp_series).dropna().sort_values().drop_duplicates().reset_index(drop=True)
    if ts.empty:
        return pd.DataFrame({"timestamp": pd.Series(dtype="datetime64[ns, UTC]")})

    expected = pd.date_range(start=ts.iloc[0], end=ts.iloc[-1], freq=freq, tz="UTC")
    missing = expected.difference(pd.DatetimeIndex(ts))
    return pd.DataFrame({"timestamp": missing})


def compute_gap_diagnostics(timestamp_series: pd.Series, expected_freq: str = "1min") -> dict[str, Any]:
    """Compute interval diagnostics such as most common gap and irregular intervals."""
    ts = pd.Series(timestamp_series).dropna().sort_values().drop_duplicates().reset_index(drop=True)
    if ts.empty:
        return {
            "most_common_gap_seconds": None,
            "irregular_intervals_count": 0,
            "irregular_intervals_sample": [],
        }

    deltas = ts.diff().dropna()
    if deltas.empty:
        return {
            "most_common_gap_seconds": None,
            "irregular_intervals_count": 0,
            "irregular_intervals_sample": [],
        }

    gap_seconds = deltas.dt.total_seconds()
    most_common_gap_seconds = float(gap_seconds.mode().iloc[0])

    expected_gap = pd.to_timedelta(expected_freq)
    irregular_mask = deltas != expected_gap
    irregular_count = int(irregular_mask.sum())

    irregular_samples = []
    if irregular_count > 0:
        irregular_idx = deltas[irregular_mask].index[:10]
        for idx in irregular_idx:
            irregular_samples.append(
                {
                    "from_timestamp": ts.iloc[idx - 1].isoformat(),
                    "to_timestamp": ts.iloc[idx].isoformat(),
                    "gap_seconds": float((ts.iloc[idx] - ts.iloc[idx - 1]).total_seconds()),
                }
            )

    return {
        "most_common_gap_seconds": most_common_gap_seconds,
        "irregular_intervals_count": irregular_count,
        "irregular_intervals_sample": irregular_samples,
    }
