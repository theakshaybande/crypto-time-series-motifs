from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def sine_wave(
    length: int = 300,
    periods: float = 4.0,
    amplitude: float = 1.0,
    phase: float = 0.0,
    noise: float = 0.0,
    trend: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a simple sine wave with optional trend and noise."""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, periods * 2.0 * np.pi, length)
    signal = amplitude * np.sin(x + phase)
    signal = signal + trend * np.linspace(0.0, 1.0, length)
    if noise > 0.0:
        signal = signal + rng.normal(scale=noise, size=length)
    return signal.astype(float)


def noise_series(length: int = 300, scale: float = 1.0, seed: int | None = None) -> np.ndarray:
    """Generate Gaussian noise."""
    rng = np.random.default_rng(seed)
    return rng.normal(scale=scale, size=length).astype(float)


def repeated_pattern_series(
    length: int = 280,
    motif_length: int = 24,
    motif_positions: tuple[int, ...] = (40, 130, 210),
    noise: float = 0.10,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...]]:
    """Generate a noisy series with the same motif injected multiple times."""
    if motif_length >= length:
        raise ValueError("motif_length must be smaller than length")

    rng = np.random.default_rng(seed)
    series = rng.normal(scale=noise, size=length)

    x = np.linspace(0.0, 1.0, motif_length)
    motif = (
        0.55 * np.sin(2.0 * np.pi * x)
        + 1.20 * np.exp(-((x - 0.32) ** 2) / 0.010)
        - 0.75 * np.exp(-((x - 0.74) ** 2) / 0.012)
    )
    motif = (motif - motif.mean()) / motif.std()
    motif = 0.85 * motif

    valid_positions: list[int] = []
    for start in motif_positions:
        end = start + motif_length
        if start < 0 or end > length:
            raise ValueError(f"Motif window ({start}, {end}) is outside series length {length}")
        series[start:end] += motif + rng.normal(scale=noise * 0.20, size=motif_length)
        valid_positions.append(start)

    return series.astype(float), motif.astype(float), tuple(valid_positions)


def regime_shift_series(
    length: int = 360,
    shift_points: tuple[int, ...] = (120, 240),
    noise: float = 0.12,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a piecewise series with visibly different regimes."""
    if any(point <= 0 or point >= length for point in shift_points):
        raise ValueError("All shift points must be strictly inside the series")

    rng = np.random.default_rng(seed)
    boundaries = (0, *sorted(shift_points), length)
    series = np.zeros(length, dtype=float)
    labels = np.zeros(length, dtype=int)

    for regime, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        segment_length = end - start
        x = np.linspace(0.0, 2.0 * np.pi, segment_length)

        if regime == 0:
            segment = 0.75 * np.sin(2.0 * x)
        elif regime == 1:
            segment = 0.45 + 0.05 * np.arange(segment_length) + 0.55 * np.sin(5.0 * x)
        else:
            segment = -0.55 + 0.90 * np.cos(1.5 * x) + 0.25 * np.sign(np.sin(3.0 * x))

        segment = segment + rng.normal(scale=noise, size=segment_length)
        series[start:end] = segment
        labels[start:end] = regime

    return series, labels


def multivariate_motif_series(
    length: int = 320,
    motif_length: int = 28,
    motif_positions: tuple[int, ...] = (70, 210),
    noise: float = 0.10,
    seed: int | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Generate a tiny three-channel series with a shared motif."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 6.0 * np.pi, length)

    channel_a = 0.6 * np.sin(t) + rng.normal(scale=noise, size=length)
    channel_b = 0.4 * np.cos(1.5 * t + 0.3) + rng.normal(scale=noise, size=length)
    channel_c = 0.2 * np.sin(0.5 * t) + 0.02 * np.arange(length) / length + rng.normal(
        scale=noise, size=length
    )

    x = np.linspace(0.0, 1.0, motif_length)
    motif = 0.90 * np.sin(2.0 * np.pi * x) + 0.80 * np.exp(-((x - 0.40) ** 2) / 0.015)

    for start in motif_positions:
        end = start + motif_length
        if start < 0 or end > length:
            raise ValueError(f"Motif window ({start}, {end}) is outside series length {length}")
        channel_a[start:end] += motif
        channel_b[start:end] += 0.60 * motif[::-1]
        channel_c[start:end] += 0.35 * np.roll(motif, 2)

    channels = np.vstack([channel_a, channel_b, channel_c]).astype(float)
    return channels, motif_positions


def rolling_feature_view(series: np.ndarray, window: int = 10) -> pd.DataFrame:
    """Create a few simple features for raw-vs-feature comparisons."""
    values = np.asarray(series, dtype=float)
    s = pd.Series(values)
    returns = s.diff().fillna(0.0)
    rolling_mean = s.rolling(window, min_periods=1).mean()
    rolling_volatility = returns.rolling(window, min_periods=1).std().fillna(0.0)

    return pd.DataFrame(
        {
            "price": values,
            "returns": returns.to_numpy(),
            "rolling_mean": rolling_mean.to_numpy(),
            "rolling_volatility": rolling_volatility.to_numpy(),
        }
    )


def build_synthetic_placeholder_catalog(output_dir: str | Path | None = None, seed: int = 42) -> Path:
    """Write a few tiny CSVs into data/synthetic for quick experimentation."""
    if output_dir is None:
        output_path = Path(__file__).resolve().parents[1] / "data" / "synthetic"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    sine = sine_wave(length=180, periods=3.0, amplitude=1.0, noise=0.05, seed=seed)
    repeated, _, _ = repeated_pattern_series(length=220, motif_length=24, seed=seed)
    noise_only = noise_series(length=180, scale=0.8, seed=seed)
    regime, labels = regime_shift_series(length=240, shift_points=(80, 160), seed=seed)

    pd.DataFrame({"value": sine}).to_csv(output_path / "sine_wave_series.csv", index=False)
    pd.DataFrame({"value": repeated}).to_csv(output_path / "repeated_pattern_series.csv", index=False)
    pd.DataFrame({"value": noise_only}).to_csv(output_path / "noise_series.csv", index=False)
    pd.DataFrame({"value": regime, "regime": labels}).to_csv(
        output_path / "regime_shift_series.csv",
        index=False,
    )
    return output_path
