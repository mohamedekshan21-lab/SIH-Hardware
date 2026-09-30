"""Dark / white reference calibration and reflectance conversion."""

from __future__ import annotations

import numpy as np


def apply_dark_white_correction(
    raw: np.ndarray,
    dark: np.ndarray | None,
    white: np.ndarray | None,
    epsilon: float = 1e-6,
) -> np.ndarray:
    """Convert raw digital numbers to reflectance using dark/white references.

    reflectance = (raw - dark) / (white - dark + ε)

    Args:
        raw: (H, W, B) raw frame.
        dark: (H, W, B) or (B,) dark reference. None → zeros.
        white: (H, W, B) or (B,) white reference. None → ones.
        epsilon: Small value to avoid division by zero.

    Returns:
        Reflectance cube clipped to [0, 1].
    """
    if dark is None:
        dark = np.zeros(1)
    if white is None:
        white = np.ones(1)

    # Broadcast if 1-D
    if dark.ndim == 1:
        dark = dark[np.newaxis, np.newaxis, :]
    if white.ndim == 1:
        white = white[np.newaxis, np.newaxis, :]

    reflectance = (raw - dark) / (white - dark + epsilon)
    return np.clip(reflectance, 0.0, 1.0).astype(np.float32)


def validate_calibration(
    dark: np.ndarray,
    white: np.ndarray,
    min_snr: float = 20.0,
) -> dict:
    """Check calibration quality.

    Returns:
        Dict with 'pass' (bool), 'snr', 'dark_mean', 'white_mean'.
    """
    dark_mean = float(np.mean(dark))
    white_mean = float(np.mean(white))
    diff = white - dark
    noise = float(np.std(diff))
    snr = float(np.mean(diff) / (noise + 1e-9))

    return {
        "pass": snr >= min_snr,
        "snr": round(snr, 2),
        "dark_mean": round(dark_mean, 5),
        "white_mean": round(white_mean, 5),
    }
