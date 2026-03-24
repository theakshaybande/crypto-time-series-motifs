# Synthetic Data

This folder is reserved for tiny generated datasets used by the lab notebooks.

Use the placeholders in `../../utils/data_generators.py` to generate:

- sine waves
- repeated patterns
- noise-only series
- regime-shifted series

Example:

```python
from utils.data_generators import build_synthetic_placeholder_catalog
build_synthetic_placeholder_catalog()
```

The goal is to keep every experiment lightweight and easy to understand.
