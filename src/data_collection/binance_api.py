"""Binance REST API downloader for OHLCV market data."""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import requests

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
INTERVAL_TO_MS = {"1m": 60_000}

BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


@dataclass(frozen=True)
class DownloadConfig:
    """Configuration object for Binance kline downloads."""

    symbol: str
    interval: str
    start_utc: pd.Timestamp
    end_utc_exclusive: pd.Timestamp
    limit: int = 1000
    timeout_seconds: int = 30
    max_retries: int = 5
    pause_seconds: float = 0.05

    def validate(self) -> None:
        """Validate semantic correctness of the configuration."""
        if self.interval not in INTERVAL_TO_MS:
            raise ValueError(f"Unsupported interval '{self.interval}'. Supported: {sorted(INTERVAL_TO_MS)}")
        if self.start_utc >= self.end_utc_exclusive:
            raise ValueError("start_utc must be strictly earlier than end_utc_exclusive")
        if self.limit < 1 or self.limit > 1000:
            raise ValueError("Binance kline limit must be in [1, 1000].")


def _request_klines_batch(
    session: requests.Session,
    config: DownloadConfig,
    start_ms: int,
    end_ms: int,
) -> list[list]:
    """Request one paginated kline batch with retry logic for transient failures."""
    params = {
        "symbol": config.symbol,
        "interval": config.interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": config.limit,
    }

    last_error: Exception | None = None
    for attempt in range(1, config.max_retries + 1):
        try:
            response = session.get(BINANCE_KLINES_URL, params=params, timeout=config.timeout_seconds)
            if response.status_code == 429:
                retry_after = int(response.headers.get("Retry-After", "1"))
                time.sleep(max(retry_after, 1))
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            if attempt == config.max_retries:
                break
            backoff_seconds = min(2**attempt, 30)
            time.sleep(backoff_seconds)

    raise RuntimeError(f"Failed to fetch kline batch after {config.max_retries} attempts.") from last_error


def fetch_klines(config: DownloadConfig) -> list[list]:
    """Download all klines for the configured time window using API pagination."""
    config.validate()

    interval_ms = INTERVAL_TO_MS[config.interval]
    start_ms = int(config.start_utc.timestamp() * 1000)
    end_ms = int(config.end_utc_exclusive.timestamp() * 1000)

    all_rows: list[list] = []
    cursor_ms = start_ms

    with requests.Session() as session:
        while cursor_ms < end_ms:
            batch = _request_klines_batch(session=session, config=config, start_ms=cursor_ms, end_ms=end_ms)
            if not batch:
                break

            all_rows.extend(batch)
            last_open_ms = int(batch[-1][0])
            next_cursor_ms = last_open_ms + interval_ms

            if next_cursor_ms <= cursor_ms:
                raise RuntimeError(
                    "Pagination cursor did not advance. Aborting to avoid infinite loop."
                )

            cursor_ms = next_cursor_ms
            if config.pause_seconds > 0:
                time.sleep(config.pause_seconds)

    return all_rows


def klines_to_dataframe(
    rows: list[list],
    start_utc: pd.Timestamp,
    end_utc_exclusive: pd.Timestamp,
) -> pd.DataFrame:
    """Convert raw Binance kline rows into canonical OHLCV dataframe."""
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    df = pd.DataFrame(rows, columns=BINANCE_KLINE_COLUMNS)

    canonical = df.rename(columns={"open_time": "timestamp"})[
        ["timestamp", "open", "high", "low", "close", "volume"]
    ].copy()

    canonical["timestamp"] = pd.to_datetime(canonical["timestamp"], unit="ms", utc=True)
    numeric_cols = ["open", "high", "low", "close", "volume"]
    for col in numeric_cols:
        canonical[col] = pd.to_numeric(canonical[col], errors="coerce")

    if canonical[numeric_cols].isna().any().any():
        raise ValueError("Encountered non-numeric OHLCV values while parsing Binance data.")

    canonical = canonical[(canonical["timestamp"] >= start_utc) & (canonical["timestamp"] < end_utc_exclusive)]
    canonical = canonical.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return canonical.reset_index(drop=True)


def download_ohlcv_dataframe(config: DownloadConfig) -> pd.DataFrame:
    """Download and parse Binance kline data into canonical OHLCV format."""
    rows = fetch_klines(config)
    return klines_to_dataframe(rows=rows, start_utc=config.start_utc, end_utc_exclusive=config.end_utc_exclusive)
