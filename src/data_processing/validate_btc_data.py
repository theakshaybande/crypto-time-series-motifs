"""Data quality diagnostics for symbol-specific 1-minute OHLCV data."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import (
    ensure_directories,
    missing_timestamps_csv_path,
    raw_ohlcv_path,
    report_json_path,
)
from src.utils.time_series import (
    compute_gap_diagnostics,
    infer_missing_timestamps,
    standardize_ohlcv_dataframe,
)


def evaluate_time_series_quality(
    df: pd.DataFrame,
    *,
    symbol: str,
    frequency: str,
    duplicate_timestamps: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build quality report and missing timestamp frame from an OHLCV dataframe."""
    missing_df = infer_missing_timestamps(df["timestamp"], freq=frequency)
    gap_stats = compute_gap_diagnostics(df["timestamp"], expected_freq=frequency)

    report: dict[str, Any] = {
        "symbol": symbol,
        "frequency": frequency,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(len(df)),
        "start_timestamp_utc": df["timestamp"].iloc[0].isoformat() if not df.empty else None,
        "end_timestamp_utc": df["timestamp"].iloc[-1].isoformat() if not df.empty else None,
        "duplicate_timestamps": int(duplicate_timestamps),
        "missing_intervals_count": int(len(missing_df)),
        "most_common_gap_seconds": gap_stats["most_common_gap_seconds"],
        "irregular_intervals_count": gap_stats["irregular_intervals_count"],
        "irregular_intervals_sample": gap_stats["irregular_intervals_sample"],
    }
    return report, missing_df


def write_quality_artifacts(
    report: dict[str, Any],
    missing_df: pd.DataFrame,
    *,
    report_output: Path,
    missing_output: Path,
) -> None:
    """Persist quality report JSON and missing timestamp CSV diagnostics."""
    ensure_directories(report_output.parent, missing_output.parent)

    with report_output.open("w", encoding="utf-8") as fp:
        json.dump(report, fp, indent=2)

    missing_df.to_csv(missing_output, index=False)


def generate_quality_artifacts_from_file(
    input_path: Path,
    *,
    report_output: Path,
    missing_output: Path,
    symbol: str = "BTCUSDT",
    frequency: str = "1min",
) -> dict[str, Any]:
    """Load OHLCV parquet, evaluate quality, and write report artifacts."""
    raw_df = pd.read_parquet(input_path)
    if "timestamp" not in raw_df.columns:
        raise ValueError("Input file does not contain a 'timestamp' column.")

    duplicate_timestamps = int(pd.to_datetime(raw_df["timestamp"], utc=True, errors="coerce").duplicated().sum())
    standardized = standardize_ohlcv_dataframe(raw_df)

    report, missing_df = evaluate_time_series_quality(
        standardized,
        symbol=symbol,
        frequency=frequency,
        duplicate_timestamps=duplicate_timestamps,
    )
    write_quality_artifacts(report, missing_df, report_output=report_output, missing_output=missing_output)
    return report


def build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI parser for raw data diagnostics."""
    parser = argparse.ArgumentParser(description="Validate raw 1-minute OHLCV data and emit diagnostics.")
    parser.add_argument("--input", default=None, help="Input raw parquet path.")
    parser.add_argument("--report-output", default=None, help="Output JSON report path.")
    parser.add_argument(
        "--missing-output",
        default=None,
        help="Output CSV path for missing timestamps diagnostics.",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol label for metadata report.")
    parser.add_argument("--interval", default="1m", help="Trading interval used for file naming (default: 1m).")
    parser.add_argument("--frequency", default="1min", help="Expected frequency for gap checks.")
    return parser


def main() -> None:
    """Run quality checks and persist report artifacts to metadata directory."""
    parser = build_argument_parser()
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else raw_ohlcv_path(symbol=args.symbol, interval=args.interval)
    report_output = (
        Path(args.report_output)
        if args.report_output
        else report_json_path(symbol=args.symbol, interval=args.interval)
    )
    missing_output = (
        Path(args.missing_output)
        if args.missing_output
        else missing_timestamps_csv_path(symbol=args.symbol, interval=args.interval)
    )

    report = generate_quality_artifacts_from_file(
        input_path=input_path,
        report_output=report_output,
        missing_output=missing_output,
        symbol=args.symbol,
        frequency=args.frequency,
    )

    print(f"Rows: {report['row_count']:,}")
    print(f"Missing intervals: {report['missing_intervals_count']:,}")
    print(f"Most common gap (seconds): {report['most_common_gap_seconds']}")
    print(f"Report saved to: {report_output}")


if __name__ == "__main__":
    main()
