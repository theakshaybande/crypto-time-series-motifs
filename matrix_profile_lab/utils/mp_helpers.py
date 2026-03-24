from __future__ import annotations

import numpy as np

try:
    import stumpy
except ImportError:  # pragma: no cover - handled at runtime in notebooks
    stumpy = None


def _require_stumpy() -> None:
    if stumpy is None:
        raise ImportError("stumpy is required for matrix profile computations. Install project requirements first.")


def sliding_windows(series: np.ndarray, window: int) -> np.ndarray:
    """Return all subsequences of a given length."""
    values = np.asarray(series, dtype=float)
    if window <= 1 or window > len(values):
        raise ValueError("window must be between 2 and the series length")
    return np.lib.stride_tricks.sliding_window_view(values, window_shape=window)


def znormalize(values: np.ndarray) -> np.ndarray:
    """Z-normalize a vector and handle zero-variance edge cases."""
    array = np.asarray(values, dtype=float)
    std = array.std()
    if np.isclose(std, 0.0):
        return np.zeros_like(array)
    return (array - array.mean()) / std


def distance_profile_bruteforce(
    series: np.ndarray,
    query_start: int,
    window: int,
    normalize: bool = True,
    exclusion_zone: int | None = None,
) -> np.ndarray:
    """Compute a brute-force distance profile for one subsequence."""
    values = np.asarray(series, dtype=float)
    if query_start < 0 or query_start + window > len(values):
        raise ValueError("query_start and window must define a valid subsequence")

    candidates = sliding_windows(values, window).copy()
    query = values[query_start : query_start + window].copy()

    if normalize:
        query = znormalize(query)
        means = candidates.mean(axis=1, keepdims=True)
        stds = candidates.std(axis=1, keepdims=True)
        stds[np.isclose(stds, 0.0)] = 1.0
        candidates = (candidates - means) / stds

    distances = np.sqrt(np.sum((candidates - query) ** 2, axis=1))

    zone = exclusion_zone if exclusion_zone is not None else max(1, window // 4)
    left = max(0, query_start - zone)
    right = min(len(distances), query_start + zone + 1)
    distances[left:right] = np.nan
    return distances


def compute_bruteforce_matrix_profile(
    series: np.ndarray,
    window: int,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """Compute a small brute-force matrix profile for educational comparisons."""
    values = np.asarray(series, dtype=float)
    n_subseq = len(values) - window + 1
    if n_subseq <= 1:
        raise ValueError("window is too large for the provided series")

    profile = np.empty(n_subseq, dtype=float)
    indices = np.empty(n_subseq, dtype=int)
    for start in range(n_subseq):
        distances = distance_profile_bruteforce(values, start, window, normalize=normalize)
        nearest = int(np.nanargmin(distances))
        profile[start] = float(distances[nearest])
        indices[start] = nearest

    return {"profile": profile, "indices": indices}


def compute_matrix_profile(
    series: np.ndarray,
    window: int,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """Compute a univariate matrix profile using STUMPY."""
    _require_stumpy()
    values = np.asarray(series, dtype=float)
    if window <= 2 or window >= len(values):
        raise ValueError("window must be greater than 2 and smaller than the series length")

    result = stumpy.stump(values, m=window, normalize=normalize)
    return {
        "profile": result[:, 0].astype(float),
        "indices": result[:, 1].astype(int),
        "left_indices": result[:, 2].astype(int),
        "right_indices": result[:, 3].astype(int),
    }


def multivariate_matrix_profile(
    channels: np.ndarray,
    window: int,
    normalize: bool = True,
) -> dict[str, np.ndarray]:
    """Compute a multivariate matrix profile using STUMPY mstump."""
    _require_stumpy()
    values = np.asarray(channels, dtype=float)
    if values.ndim != 2:
        raise ValueError("channels must have shape (n_channels, n_timepoints)")
    if window <= 2 or window >= values.shape[1]:
        raise ValueError("window must be greater than 2 and smaller than the series length")

    profile, indices = stumpy.mstump(values, m=window, normalize=normalize)
    return {"profile": np.asarray(profile, dtype=float), "indices": np.asarray(indices, dtype=int)}


def extract_top_motif(profile: np.ndarray, indices: np.ndarray) -> tuple[int, int]:
    """Return the start indices of the strongest motif pair."""
    mp = np.asarray(profile, dtype=float)
    nn = np.asarray(indices, dtype=int)
    valid = np.isfinite(mp) & (nn >= 0)
    if not valid.any():
        raise ValueError("No valid motif pair could be extracted")

    best = int(np.argmin(np.where(valid, mp, np.inf)))
    return best, int(nn[best])


def extract_top_discord(profile: np.ndarray) -> int:
    """Return the start index of the strongest discord."""
    mp = np.asarray(profile, dtype=float)
    valid = np.isfinite(mp)
    if not valid.any():
        raise ValueError("No valid discord could be extracted")
    return int(np.argmax(np.where(valid, mp, -np.inf)))
