"""Band selection and PCA for dimensionality reduction."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA


def select_bands(
    cube: np.ndarray,
    wavelengths: np.ndarray,
    selected_ranges: list[tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Select specific wavelength bands.

    Args:
        cube: (H, W, B) spectral cube.
        wavelengths: (B,) wavelength values.
        selected_ranges: List of (min_wl, max_wl) tuples. If None, selects
            discriminative bands for food safety (literature-based).

    Returns:
        Tuple of (reduced cube, selected wavelengths).
    """
    if selected_ranges is None:
        # Key discriminative bands for pathogen detection in NIR
        selected_ranges = [
            (950, 1000),   # Water / bacterial scattering
            (1020, 1080),  # Protein / Salmonella
            (1130, 1200),  # Lipid / Listeria
            (1250, 1320),  # Biofilm / E. coli
            (1380, 1420),  # Fat / moisture
            (1440, 1480),  # Water absorption
            (1490, 1530),  # Protein amide
        ]

    mask = np.zeros(len(wavelengths), dtype=bool)
    for lo, hi in selected_ranges:
        mask |= (wavelengths >= lo) & (wavelengths <= hi)

    if not mask.any():
        # Fallback: take every 4th band
        mask[::4] = True

    return cube[:, :, mask], wavelengths[mask]


def apply_pca(
    cube: np.ndarray,
    n_components: int = 10,
) -> tuple[np.ndarray, PCA]:
    """Apply PCA along the spectral axis.

    Args:
        cube: (H, W, B) spectral cube.
        n_components: Number of principal components to keep.

    Returns:
        Tuple of (scores array (H, W, n_components), fitted PCA object).
    """
    h, w, b = cube.shape
    flat = cube.reshape(-1, b)

    n_components = min(n_components, b, flat.shape[0])
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(flat)

    return scores.reshape(h, w, n_components).astype(np.float32), pca
