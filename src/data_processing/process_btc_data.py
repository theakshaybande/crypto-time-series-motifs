"""Processing pipeline for symbol-specific 1-minute OHLCV raw data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data_processing.validate_btc_data import generate_quality_artifacts_from_file
from src.utils.paths import (
    ensure_directories,
    missing_timestamps_csv_path,
    processed_ohlcv_path,
    raw_ohlcv_path,
    report_json_path,
)
from src.utils.time_series import standardize_ohlcv_dataframe


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create returns and volatility features without future-data leakage."""
    processed = df.copy()

    if (processed["close"] <= 0).any():
        raise ValueError("Close prices must be strictly positive to compute log returns.")

    processed["log_return"] = np.log(processed["close"]).diff()
    processed["pct_return"] = processed["close"].pct_change()

    # Rolling window statistics use only current and historical values.
    processed["volatility_30m"] = processed["log_return"].rolling(window=30, min_periods=30).std()
    processed["volatility_60m"] = processed["log_return"].rolling(window=60, min_periods=60).std()
    processed["volatility_240m"] = processed["log_return"].rolling(window=240, min_periods=240).std()

    # Realized volatility style feature from sum of squared intraminute returns.
    processed["realized_volatility_60m"] = np.sqrt(
        (processed["log_return"] ** 2).rolling(window=60, min_periods=60).sum()
    )

    return processed


def build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI parser for raw-to-processed data pipeline."""
    parser = argparse.ArgumentParser(description="Process raw OHLCV data into thesis-ready parquet.")
    parser.add_argument("--input", default=None, help="Input raw parquet path.")
    parser.add_argument("--output", default=None, help="Output processed parquet path.")
    parser.add_argument(
        "--report-output",
        default=None,
        help="Output metadata JSON report path.",
    )
    parser.add_argument(
        "--missing-output",
        default=None,
        help="Output metadata CSV path for missing timestamps.",
    )
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol label used for default file paths.")
    parser.add_argument("--interval", default="1m", help="Trading interval used for default file paths.")
    parser.add_argument(
        "--skip-quality-report",
        action="store_true",
        help="Skip generating metadata quality report and missing timestamp CSV.",
    )
    return parser


def main() -> None:
    """Run validation + feature engineering pipeline and save processed parquet."""
    parser = build_argument_parser()
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else raw_ohlcv_path(symbol=args.symbol, interval=args.interval)
    output_path = (
        Path(args.output) if args.output else processed_ohlcv_path(symbol=args.symbol, interval=args.interval)
    )
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

    if not input_path.exists():
        raise FileNotFoundError(f"Input raw parquet not found: {input_path}")

    raw_df = pd.read_parquet(input_path)
    standardized = standardize_ohlcv_dataframe(raw_df)

    engineered = create_features(standardized)

    ensure_directories(output_path.parent)
    engineered.to_parquet(output_path, index=False, engine="pyarrow")

    if not args.skip_quality_report:
        generate_quality_artifacts_from_file(
            input_path=input_path,
            report_output=report_output,
            missing_output=missing_output,
            symbol=args.symbol,
            frequency="1min",
        )

    print(f"Processed rows written: {len(engineered):,}")
    print(f"Output file: {output_path}")
    if not args.skip_quality_report:
        print(f"Quality report JSON: {report_output}")
        print(f"Missing timestamps CSV: {missing_output}")


if __name__ == "__main__":
    main()
