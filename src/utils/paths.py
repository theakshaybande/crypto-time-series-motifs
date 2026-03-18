"""Path helpers for reproducible project file access."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"


def ensure_directories(*paths: Path) -> None:
    """Create directories if they do not already exist."""
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def interval_to_folder(interval: str) -> str:
    """Map trading interval notation to folder naming convention."""
    mapping = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1h",
        "1d": "daily",
    }
    try:
        return mapping[interval]
    except KeyError as exc:
        raise ValueError(f"Unsupported interval for folder mapping: {interval}") from exc


def raw_ohlcv_path(symbol: str = "BTCUSDT", interval: str = "1m") -> Path:
    """Return canonical raw data path for a symbol/interval pair."""
    freq_folder = interval_to_folder(interval)
    return RAW_DIR / "crypto" / freq_folder / f"{symbol}_{interval}_raw.parquet"


def processed_ohlcv_path(symbol: str = "BTCUSDT", interval: str = "1m") -> Path:
    """Return canonical processed data path for a symbol/interval pair."""
    freq_folder = interval_to_folder(interval)
    return PROCESSED_DIR / "crypto" / freq_folder / f"{symbol}_{interval}_processed.parquet"


def report_json_path(symbol: str = "BTCUSDT", interval: str = "1m") -> Path:
    """Return metadata path for quality report JSON."""
    return METADATA_DIR / f"{symbol}_{interval}_data_report.json"


def missing_timestamps_csv_path(symbol: str = "BTCUSDT", interval: str = "1m") -> Path:
    """Return metadata path for missing timestamp diagnostics CSV."""
    return METADATA_DIR / f"{symbol}_{interval}_missing_timestamps.csv"
