"""Standard Normal Variate (SNV) and Multiplicative Scatter Correction (MSC)."""

from __future__ import annotations

import numpy as np


def snv(cube: np.ndarray) -> np.ndarray:
    """Standard Normal Variate — per-pixel normalisation.

    Each spectrum is centred to zero mean and scaled to unit variance.

    Args:
        cube: (H, W, B) or (N, B).

    Returns:
        SNV-normalised cube.
    """
    shape = cube.shape
    flat = cube.reshape(-1, shape[-1]).astype(np.float64)
    means = flat.mean(axis=1, keepdims=True)
    stds = flat.std(axis=1, keepdims=True)
    stds[stds < 1e-10] = 1.0
    normalised = (flat - means) / stds
    return normalised.reshape(shape).astype(np.float32)


def msc(cube: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Multiplicative Scatter Correction.

    Fits a linear model (a + b * ref) to each spectrum and corrects scatter.

    Args:
        cube: (H, W, B) or (N, B).
        reference: (B,) reference spectrum. If None, uses the mean spectrum.

    Returns:
        MSC-corrected cube.
    """
    shape = cube.shape
    flat = cube.reshape(-1, shape[-1]).astype(np.float64)

    if reference is None:
        reference = flat.mean(axis=0)

    corrected = np.zeros_like(flat)
    for i in range(flat.shape[0]):
        # Fit: spectrum = a + b * reference
        coeffs = np.polyfit(reference, flat[i], deg=1)
        b, a = coeffs
        if abs(b) < 1e-10:
            corrected[i] = flat[i] - a
        else:
            corrected[i] = (flat[i] - a) / b

    return corrected.reshape(shape).astype(np.float32)
