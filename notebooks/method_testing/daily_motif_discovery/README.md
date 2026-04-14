# Daily Motif Discovery Notebook Pipeline

This folder contains a thesis-ready daily motif discovery workflow for financial time series, with an emphasis on Matrix Profile methods, reproducibility, and compatibility with future regime-conditioned analysis.

## Notebook Order

1. `01_daily_univariate_matrix_profile.ipynb`
   - Builds daily OHLCV from the resolved input dataset.
   - Computes daily features.
   - Runs univariate STUMPY motif discovery on one selected daily signal, defaulting to `log_return`.
   - Extracts top non-overlapping motifs and discords and visualizes them.
2. `02_daily_multivariate_matrix_profile.ipynb`
   - Reuses the same daily data construction logic.
   - Aligns a daily multivariate feature panel.
   - Runs `stumpy.mstump` and ranks motifs using the full-dimensional profile row.
   - Produces multichannel motif comparison plots and a compact summary table.

## Data Expectations

- The notebooks first search the repository for BTCUSDT data and then fall back to ETHUSDT if BTCUSDT is unavailable.
- Candidate files are searched under `data/processed` and `data/raw`, with processed parquet files preferred over raw files or CSV exports.
- If only intraday data is available, the notebooks resample it transparently to daily OHLCV before feature engineering.

## Daily Aggregation Logic

When the source data is intraday OHLCV, daily bars are constructed with:

- `open = first`
- `high = max`
- `low = min`
- `close = last`
- `volume = sum`

Daily features are then recomputed from the daily OHLCV table. The default engineered features are:

- `close`
- `log_return`
- `pct_return`
- `hl_range`
- `rolling_volatility`
- `volume`
- `volume_zscore`

## Univariate vs Multivariate Daily Motifs

The univariate notebook isolates one daily signal and asks whether that signal alone contains recurring subsequences or unusual discords. The multivariate notebook instead treats the selected channels as a joint daily state, so a motif only ranks highly when the combined return, volatility, range, and volume pattern recurs together.

This separation keeps the workflow methodologically clear: the thesis can first establish whether daily motifs are visible in a single channel and then test whether those patterns remain meaningful once recurrence is defined in a richer multivariate feature space.
