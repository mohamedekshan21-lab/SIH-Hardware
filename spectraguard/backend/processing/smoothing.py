"""Savitzky-Golay spectral smoothing."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter


def savitzky_golay_smooth(
    cube: np.ndarray,
    window_length: int = 11,
    polyorder: int = 3,
    deriv: int = 0,
) -> np.ndarray:
    """Apply Savitzky-Golay filter along the spectral (last) axis.

    Args:
        cube: (H, W, B) or (N, B) spectral data.
        window_length: Must be odd and > polyorder.
        polyorder: Polynomial order.
        deriv: Derivative order (0 = smoothing, 1 = first deriv, 2 = second).

    Returns:
        Smoothed cube of the same shape.
    """
    if window_length % 2 == 0:
        window_length += 1
    if window_length > cube.shape[-1]:
        window_length = cube.shape[-1] if cube.shape[-1] % 2 == 1 else cube.shape[-1] - 1

    original_shape = cube.shape
    # Flatten spatial dims for vectorised filtering
    flat = cube.reshape(-1, cube.shape[-1])
    smoothed = savgol_filter(flat, window_length, polyorder, deriv=deriv, axis=1)
    return smoothed.reshape(original_shape).astype(np.float32)
