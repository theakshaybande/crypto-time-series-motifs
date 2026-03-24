# matrix_profile_lab

This is a concept-first lab to deeply understand matrix profile and motif discovery before applying it to financial data.

This folder is intentionally separate from the main thesis pipeline. It is for learning, experimentation, and visual intuition building.

## What This Lab Is For

- Understanding matrix profile from first principles
- Building intuition with synthetic time series before touching larger datasets
- Comparing motifs, discords, window sizes, and regime effects
- Exploring when raw signals help and when simple features help

## What This Lab Is Not For

- Production pipelines
- Full BTC dataset processing
- Heavy experiments or thesis finalization code

## Learning Path

`intuition -> distance -> matrix profile -> motifs -> regimes -> multivariate`

## Recommended Order

1. `notebooks/00_intuition/00_intuition.ipynb`
2. `notebooks/01_univariate_basics/01_univariate_matrix_profile.ipynb`
3. `notebooks/02_distance_and_similarity/02_distance_and_similarity.ipynb`
4. `notebooks/03_matrix_profile_core/03_matrix_profile_core.ipynb`
5. `notebooks/04_motifs_vs_discords/04_motifs_vs_discords.ipynb`
6. `notebooks/05_window_size_effects/05_window_size_effects.ipynb`
7. `notebooks/07_regime_effects/07_regime_effects.ipynb`
8. `notebooks/06_multivariate_intro/06_multivariate_intro.ipynb`
9. `notebooks/08_feature_vs_raw/08_feature_vs_raw.ipynb`
10. `notebooks/09_visualization/09_visualization.ipynb`

## Notebook Guide

- `00_intuition`: Highlights repeated shapes in a noisy series so motifs feel concrete before any formal algorithm.
- `01_univariate_matrix_profile`: Computes a first univariate matrix profile and shows how low-profile regions point to repetition.
- `02_distance_and_similarity`: Compares raw and z-normalized distance profiles to show what "similarity" means in practice.
- `03_matrix_profile_core`: Contrasts a brute-force matrix profile with the STUMPY wrapper to connect theory with implementation.
- `04_motifs_vs_discords`: Shows the difference between recurring subsequences and unusual subsequences in one simple example.
- `05_window_size_effects`: Demonstrates how motif discovery changes when the subsequence window is too short, well matched, or too long.
- `06_multivariate_intro`: Introduces multivariate matrix profile on a tiny multi-channel synthetic series with a shared motif.
- `07_regime_effects`: Shows how nonstationary behavior and regime changes alter the profile landscape.
- `08_feature_vs_raw`: Compares matrix profile behavior on raw values versus simple derived features like returns and rolling volatility.
- `09_visualization`: Focuses on presentation patterns that make motifs, discords, and profile structure easier to interpret.

## Tree

```text
matrix_profile_lab/
  README.md
  data/
    synthetic/
      README.md
    small_real_samples/
      README.md
      daily_temperature_tiny.csv
  notebooks/
    00_intuition/
      00_intuition.ipynb
    01_univariate_basics/
      01_univariate_matrix_profile.ipynb
    02_distance_and_similarity/
      02_distance_and_similarity.ipynb
    03_matrix_profile_core/
      03_matrix_profile_core.ipynb
    04_motifs_vs_discords/
      04_motifs_vs_discords.ipynb
    05_window_size_effects/
      05_window_size_effects.ipynb
    06_multivariate_intro/
      06_multivariate_intro.ipynb
    07_regime_effects/
      07_regime_effects.ipynb
    08_feature_vs_raw/
      08_feature_vs_raw.ipynb
    09_visualization/
      09_visualization.ipynb
  utils/
    __init__.py
    data_generators.py
    plotting.py
    mp_helpers.py
  outputs/
    figures/
    experiments/
```

## Quick Start

Use the same Python environment as the main thesis project:

```bash
pip install -r ../requirements.txt
```

Open notebooks from inside this lab folder so relative imports resolve cleanly.

## Synthetic Data Placeholders

`utils/data_generators.py` includes lightweight helpers for:

- sine waves
- repeated patterns
- noise
- regime shifts
- small multivariate examples

To materialize example CSV files into `data/synthetic/`, run:

```python
from utils.data_generators import build_synthetic_placeholder_catalog
build_synthetic_placeholder_catalog()
```

## Notes

- Keep experiments small and visual first.
- Prefer understanding behavior over optimizing performance here.
- Do not connect this lab to the full BTC dataset.
