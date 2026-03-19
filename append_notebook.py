import nbformat
import sys

notebook_path = r"c:\Users\learn\OneDrive\Desktop\Masters Thesis\MastersThesis\notebooks\data_exploration\Explore 01.ipynb"

# Load notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

cells = [
    nbformat.v4.new_markdown_cell("## 1. Setup and Imports\nAdding a comprehensive data inspection and summary section."),
    nbformat.v4.new_code_cell('''import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Setup paths relative to project root
project_root = Path("../../").resolve()
data_dir = project_root / "data"

btc_processed_path = data_dir / "processed/crypto/1min/BTCUSDT_1m_processed.parquet"
eth_processed_path = data_dir / "processed/crypto/1min/ETHUSDT_1m_processed.parquet"
btc_missing_path = data_dir / "metadata/BTCUSDT_1m_missing_timestamps.csv"
eth_missing_path = data_dir / "metadata/ETHUSDT_1m_missing_timestamps.csv"'''),
    nbformat.v4.new_markdown_cell("## 2. Load Data"),
    nbformat.v4.new_code_cell('''# Load Parquet files
df_btc = pd.read_parquet(btc_processed_path)
df_eth = pd.read_parquet(eth_processed_path)

# Convert timestamp to datetime UTC
df_btc['timestamp'] = pd.to_datetime(df_btc['timestamp'], utc=True)
df_eth['timestamp'] = pd.to_datetime(df_eth['timestamp'], utc=True)

# Sort by timestamp to ensure chronological order
df_btc = df_btc.sort_values('timestamp').reset_index(drop=True)
df_eth = df_eth.sort_values('timestamp').reset_index(drop=True)'''),
    nbformat.v4.new_markdown_cell("## 3. Basic Summary"),
    nbformat.v4.new_code_cell('''def summarize_data(name, df):
    print(f"--- {name} Summary ---")
    print(f"Shape: {df.shape}")
    print(f"Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print("\\nTotal missing values per column:")
    print(df.isnull().sum())
    
    print(f"\\nDuplicate timestamps check: {df['timestamp'].duplicated().sum()}\\n")

summarize_data("BTCUSDT", df_btc)
summarize_data("ETHUSDT", df_eth)'''),
    nbformat.v4.new_markdown_cell("## 4. Feature Inspection"),
    nbformat.v4.new_code_cell('''cols_to_inspect = ['log_return', 'pct_return', 'volatility_30m', 'volatility_60m', 'realized_volatility_60m']

for name, df in [("BTC", df_btc), ("ETH", df_eth)]:
    print(f"================ {name} Feature Inspection ================")
    
    existing_cols = [c for c in cols_to_inspect if c in df.columns]
    
    print(f"{name} Describe:")
    display(df[existing_cols].describe())
    
    print(f"\\n{name} Skewness:")
    display(df[existing_cols].skew())
    
    print(f"\\n{name} Kurtosis:")
    display(df[existing_cols].kurtosis())
    print("\\n")'''),
    nbformat.v4.new_markdown_cell("## 5. Missing Timestamp Analysis"),
    nbformat.v4.new_code_cell('''import os

for name, path in [("BTC", btc_missing_path), ("ETH", eth_missing_path)]:
    print(f"--- {name} Missing Timestamps ---")
    if os.path.exists(path):
        df_missing = pd.read_csv(path)
        print(f"Number of missing timestamps: {len(df_missing)}")
        print("First 10 missing timestamps:")
        display(df_missing.head(10))
        
        # Approximate missing percentage
        if len(df_missing) > 0 and len(df) > 0:
            total_expected_minutes = len(df) + len(df_missing)
            pct_missing = (len(df_missing) / total_expected_minutes) * 100
            print(f"Percentage missing vs total expected: {pct_missing:.4f}%\\n")
    else:
        print(f"File not found: {path}\\n")'''),
    nbformat.v4.new_markdown_cell("## 6. Visualization"),
    nbformat.v4.new_code_cell('''def plot_features(name, df):
    fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
    
    # 1. Close price (full series)
    axes[0].plot(df['timestamp'], df['close'], label='Close Price')
    axes[0].set_title(f'{name} Close Price')
    axes[0].legend()
    
    # 2. Log return and pct return
    if 'log_return' in df.columns and 'pct_return' in df.columns:
        axes[1].plot(df['timestamp'], df['log_return'], alpha=0.5, label='Log Return')
        axes[1].plot(df['timestamp'], df['pct_return'], alpha=0.5, label='Pct Return')
        axes[1].set_title(f'{name} Returns')
        axes[1].legend()
    
    # 3. Rolling volatility
    vol_cols = [c for c in ['volatility_30m', 'volatility_60m', 'volatility_240m'] if c in df.columns]
    for col in vol_cols:
        axes[2].plot(df['timestamp'], df[col], alpha=0.7, label=col)
    axes[2].set_title(f'{name} Rolling Volatility')
    axes[2].legend()
    
    # 4. Realized volatility 60m
    if 'realized_volatility_60m' in df.columns:
        axes[3].plot(df['timestamp'], df['realized_volatility_60m'], color='purple', label='Realized Vol 60m')
        axes[3].set_title(f'{name} Realized Volatility')
        axes[3].legend()
        
    plt.tight_layout()
    plt.show()

    # Histograms
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    if 'log_return' in df.columns:
        axes[0].hist(df['log_return'].dropna(), bins=100, alpha=0.7)
        axes[0].set_title(f'{name} Log Return Histogram')
        
    if 'realized_volatility_60m' in df.columns:
        axes[1].hist(df['realized_volatility_60m'].dropna(), bins=100, alpha=0.7, color='purple')
        axes[1].set_title(f'{name} Realized Volatility 60m Histogram')
    
    plt.tight_layout()
    plt.show()

print("Plotting BTC...")
plot_features("BTC", df_btc)
print("Plotting ETH...")
plot_features("ETH", df_eth)'''),
    nbformat.v4.new_markdown_cell("## 7. Comparative Plots"),
    nbformat.v4.new_code_cell('''# Overlay BTC vs ETH log_return
plt.figure(figsize=(15, 5))
if 'log_return' in df_btc.columns and 'log_return' in df_eth.columns:
    plt.plot(df_btc['timestamp'], df_btc['log_return'], alpha=0.5, label='BTC Log Return')
    plt.plot(df_eth['timestamp'], df_eth['log_return'], alpha=0.5, label='ETH Log Return')
    plt.title('BTC vs ETH Log Return Overlay')
    plt.legend()
plt.show()

# Overlay volatility_60m
plt.figure(figsize=(15, 5))
if 'volatility_60m' in df_btc.columns and 'volatility_60m' in df_eth.columns:
    plt.plot(df_btc['timestamp'], df_btc['volatility_60m'], alpha=0.7, label='BTC Volatility 60m')
    plt.plot(df_eth['timestamp'], df_eth['volatility_60m'], alpha=0.7, label='ETH Volatility 60m')
    plt.title('BTC vs ETH Volatility 60m Overlay')
    plt.legend()
plt.show()'''),
    nbformat.v4.new_markdown_cell("## 8. Stationarity Insight"),
    nbformat.v4.new_code_cell('''# Quick visual: Rolling mean and rolling std of log_return
if 'log_return' in df_btc.columns:
    plt.figure(figsize=(15, 6))
    
    rolling_mean_btc = df_btc['log_return'].rolling(window=1000).mean()
    rolling_std_btc = df_btc['log_return'].rolling(window=1000).std()
    
    plt.plot(df_btc['timestamp'], rolling_mean_btc, label='BTC Rolling Mean (w=1000)', color='red')
    plt.plot(df_btc['timestamp'], rolling_std_btc, label='BTC Rolling Std (w=1000)', color='blue')
    
    plt.title('BTC Log Return - Stationarity Insight (Rolling Mean & Std)')
    plt.legend()
    plt.show()'''),
    nbformat.v4.new_markdown_cell("## 9. Output for Downstream Use"),
    nbformat.v4.new_code_cell('''if 'log_return' in df_btc.columns:
    btc_log_return = df_btc["log_return"].dropna()
    print(f"Length of btc_log_return series: {len(btc_log_return)}")
    print(f"NaNs remaining in btc_log_return: {btc_log_return.isnull().sum()}")

if 'realized_volatility_60m' in df_btc.columns:
    btc_realized_vol = df_btc["realized_volatility_60m"].dropna()
    print(f"Length of btc_realized_vol series: {len(btc_realized_vol)}")
    print(f"NaNs remaining in btc_realized_vol: {btc_realized_vol.isnull().sum()}")'''),
    nbformat.v4.new_markdown_cell('''## 10. Summary and Next Steps

- **Which signal is most stable:** `log_return` appears to be the most stationary signal, exhibiting properties centered around zero with relatively stable variances over long periods.
- **Most suitable for motif discovery:** `log_return` is broadly accepted as the strongest candidate for motif discovery since motifs depend on scale-invariant and trend-free properties for meaningful local distances, minimizing the effect of non-stationary price drifts.
- **Impact of missing timestamps:** Unhandled gaps implicitly alter the distance profile sequences and can break consecutive motif boundaries, thus highlighting the crucial role of treating or flagging missing time rows accurately.
- **Recommend next step:** Proceed to the matrix profile formulation, utilizing `btc_log_return` to locate the dominant and informative motifs.''')
]

nb.cells.extend(cells)

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbformat.write(nb, f)

print("Cells successfully appended to notebook.")
