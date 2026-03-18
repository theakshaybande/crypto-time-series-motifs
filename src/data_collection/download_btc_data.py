"""CLI entrypoint for downloading BTCUSDT 1-minute OHLCV data."""

from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path

import pandas as pd
from dateutil import parser as dt_parser

from src.data_collection.binance_api import DownloadConfig, download_ohlcv_dataframe
from src.utils.paths import ensure_directories, raw_ohlcv_path
from src.utils.time_series import standardize_ohlcv_dataframe


def parse_utc_timestamp(value: str, *, date_only_end_exclusive: bool = False) -> pd.Timestamp:
    """Parse a date/datetime string as UTC timestamp.

    If `date_only_end_exclusive` is True and the input is date-only
    (YYYY-MM-DD), one day is added to produce an exclusive upper bound.
    """
    parsed = dt_parser.isoparse(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)

    ts = pd.Timestamp(parsed)
    if date_only_end_exclusive and "T" not in value and len(value) <= 10:
        ts = ts + pd.Timedelta(days=1)
    return ts


def build_argument_parser() -> argparse.ArgumentParser:
    """Build CLI parser for BTC download script."""
    parser = argparse.ArgumentParser(description="Download BTCUSDT 1m OHLCV data from Binance API.")
    parser.add_argument("--start", required=True, help="Start timestamp (UTC), e.g. 2023-01-01")
    parser.add_argument("--end", required=True, help="End timestamp (UTC). Date-only is treated as inclusive day.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading symbol (default: BTCUSDT)")
    parser.add_argument("--interval", default="1m", help="Kline interval (default: 1m)")
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output parquet path. Defaults to canonical raw path under data/raw/crypto/1min.",
    )
    return parser


def main() -> None:
    """Run BTC data downloader and persist canonical raw parquet file."""
    parser = build_argument_parser()
    args = parser.parse_args()

    start_utc = parse_utc_timestamp(args.start)
    end_utc_exclusive = parse_utc_timestamp(args.end, date_only_end_exclusive=True)

    config = DownloadConfig(
        symbol=args.symbol,
        interval=args.interval,
        start_utc=start_utc,
        end_utc_exclusive=end_utc_exclusive,
    )

    df = download_ohlcv_dataframe(config)
    if df.empty:
        raise RuntimeError("No rows were downloaded for the requested time window.")

    df = standardize_ohlcv_dataframe(df)

    output_path = Path(args.output) if args.output else raw_ohlcv_path(symbol=args.symbol, interval=args.interval)
    ensure_directories(output_path.parent)
    df.to_parquet(output_path, index=False, engine="pyarrow")

    print(f"Saved raw OHLCV rows: {len(df):,}")
    print(f"Time range UTC: {df['timestamp'].iloc[0].isoformat()} -> {df['timestamp'].iloc[-1].isoformat()}")
    print(f"Output file: {output_path}")


if __name__ == "__main__":
    main()
