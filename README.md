# MastersThesis

Research workspace for benchmarking motif discovery methods in financial time series under regime nonstationarity.

## BTC Data Collection Pipeline

This repository includes a first-stage BTCUSDT 1-minute ingestion and processing pipeline.

### What the downloader does

- Pulls Binance public BTCUSDT 1-minute klines in configurable UTC date windows.
- Standardizes to canonical OHLCV schema:
  `timestamp, open, high, low, close, volume`
- Enforces UTC timestamps, sorting, duplicate removal, and numeric dtype checks.
- Stores raw output as parquet.

### Files produced

- Raw parquet:
  `data/raw/crypto/1min/BTCUSDT_1m_raw.parquet`
- Processed parquet with returns and volatility features:
  `data/processed/crypto/1min/BTCUSDT_1m_processed.parquet`
- Data quality report:
  `data/metadata/BTCUSDT_1m_data_report.json`
- Missing timestamp diagnostics:
  `data/metadata/BTCUSDT_1m_missing_timestamps.csv`

### How to run

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download BTC data:

```bash
python -m src.data_collection.download_btc_data --start 2023-01-01 --end 2024-12-31
```

4. Process raw data and generate metadata diagnostics:

```bash
python -m src.data_processing.process_btc_data
```

5. Optional standalone validation pass:

```bash
python -m src.data_processing.validate_btc_data
```

### Notebook

Use the notebook below for exploratory data checks and thesis-ready sanity validation:

- `notebooks/data_exploration/01_btcusdt_1m_data_check.ipynb`
