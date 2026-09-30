"""Product segmentation — separate product from belt/background."""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def segment_product(
    cube: np.ndarray,
    threshold: float = 0.15,
    min_area_fraction: float = 0.05,
) -> np.ndarray:
    """Segment the product region from the conveyor belt background.

    Uses mean reflectance across bands — the belt is typically much darker.

    Args:
        cube: (H, W, B) reflectance cube.
        threshold: Reflectance threshold; pixels above this are product.
        min_area_fraction: Minimum fraction of image to be called a product.

    Returns:
        Boolean mask (H, W) — True for product pixels.
    """
    mean_reflectance = np.mean(cube, axis=2)
    mask = mean_reflectance > threshold

    # Clean up with morphological operations
    mask = ndimage.binary_fill_holes(mask)
    mask = ndimage.binary_opening(mask, structure=np.ones((3, 3)))
    mask = ndimage.binary_closing(mask, structure=np.ones((3, 3)))

    # Keep only the largest connected component
    labelled, n_features = ndimage.label(mask)
    if n_features > 1:
        sizes = ndimage.sum(mask, labelled, range(1, n_features + 1))
        largest = int(np.argmax(sizes)) + 1
        mask = labelled == largest

    # Check minimum area
    area_fraction = mask.sum() / mask.size
    if area_fraction < min_area_fraction:
        # Probably no product in frame — return all False
        return np.zeros_like(mask, dtype=bool)

    return mask.astype(bool)


def extract_product_pixels(
    cube: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Extract just the product pixels as a 2-D array.

    Args:
        cube: (H, W, B).
        mask: (H, W) boolean.

    Returns:
        (N, B) array of product-only pixels.
    """
    return cube[mask].reshape(-1, cube.shape[-1])
