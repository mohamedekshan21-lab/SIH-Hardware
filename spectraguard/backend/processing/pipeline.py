"""Full preprocessing pipeline — orchestrates calibration → normalisation → dimensionality reduction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.acquisition.base import HSICube
from backend.processing.calibration import apply_dark_white_correction
from backend.processing.smoothing import savitzky_golay_smooth
from backend.processing.normalization import snv
from backend.processing.dimensionality import select_bands
from backend.processing.segmentation import segment_product, extract_product_pixels


@dataclass
class ProcessedCube:
    """Result of the preprocessing pipeline."""
    reflectance: np.ndarray          # (H, W, B) calibrated reflectance
    smoothed: np.ndarray             # (H, W, B) after SG smoothing
    normalised: np.ndarray           # (H, W, B_selected) after SNV + band selection
    product_mask: np.ndarray         # (H, W) bool
    product_pixels: np.ndarray       # (N, B_selected) for classifier input
    selected_wavelengths: np.ndarray # (B_selected,)
    mean_spectrum: np.ndarray        # (B,) mean spectrum of product
    rgb_preview: np.ndarray          # (H, W, 3) pseudo-RGB for display


class PreprocessingPipeline:
    """End-to-end preprocessing for hyperspectral cubes."""

    def __init__(
        self,
        sg_window: int = 11,
        sg_polyorder: int = 3,
        use_snv: bool = True,
        use_band_selection: bool = True,
        reflectance_threshold: float = 0.15,
    ):
        self.sg_window = sg_window
        self.sg_polyorder = sg_polyorder
        self.use_snv = use_snv
        self.use_band_selection = use_band_selection
        self.reflectance_threshold = reflectance_threshold
        self._dark_ref: Optional[np.ndarray] = None
        self._white_ref: Optional[np.ndarray] = None

    def set_references(self, dark: np.ndarray | None, white: np.ndarray | None):
        """Set calibration references."""
        self._dark_ref = dark
        self._white_ref = white

    def process(self, cube: HSICube) -> ProcessedCube:
        """Run the full pipeline on a single HSI cube.

        Steps:
        1. Dark/white reference correction → reflectance
        2. Savitzky-Golay smoothing
        3. SNV normalisation
        4. Band selection
        5. Product segmentation
        6. Extract product pixels for classifier
        """
        data = cube.data  # (H, W, B)
        wavelengths = cube.wavelengths

        # 1. Reflectance conversion
        reflectance = apply_dark_white_correction(data, self._dark_ref, self._white_ref)

        # 2. Spectral smoothing
        smoothed = savitzky_golay_smooth(reflectance, self.sg_window, self.sg_polyorder)

        # 3. SNV normalisation
        normalised = snv(smoothed) if self.use_snv else smoothed

        # 4. Band selection
        if self.use_band_selection:
            selected, sel_wl = select_bands(normalised, wavelengths)
        else:
            selected = normalised
            sel_wl = wavelengths

        # 5. Product segmentation
        product_mask = segment_product(reflectance, self.reflectance_threshold)

        # 6. Extract product pixels
        if product_mask.any():
            product_pixels = extract_product_pixels(selected, product_mask)
        else:
            # No product detected — return empty
            product_pixels = np.zeros((0, selected.shape[2]), dtype=np.float32)

        # Mean spectrum
        if product_mask.any():
            mean_spectrum = np.mean(smoothed[product_mask], axis=0)
        else:
            mean_spectrum = np.mean(smoothed.reshape(-1, smoothed.shape[-1]), axis=0)

        # Pseudo-RGB preview (pick 3 bands roughly R/G/B equivalent in NIR)
        rgb_preview = self._make_rgb_preview(reflectance, wavelengths)

        return ProcessedCube(
            reflectance=reflectance,
            smoothed=smoothed,
            normalised=selected,
            product_mask=product_mask,
            product_pixels=product_pixels,
            selected_wavelengths=sel_wl,
            mean_spectrum=mean_spectrum,
            rgb_preview=rgb_preview,
        )

    @staticmethod
    def _make_rgb_preview(reflectance: np.ndarray, wavelengths: np.ndarray) -> np.ndarray:
        """Create a pseudo-RGB image from the hyperspectral cube.

        Maps three NIR bands to RGB channels for visual display.
        """
        h, w, b = reflectance.shape
        # Pick bands near 1000 nm (R), 1200 nm (G), 1400 nm (B)
        targets = [1000, 1200, 1400]
        indices = [int(np.argmin(np.abs(wavelengths - t))) for t in targets]

        rgb = np.stack([reflectance[:, :, i] for i in indices], axis=-1)
        # Normalise to [0, 255] for display
        for c in range(3):
            ch = rgb[:, :, c]
            mn, mx = ch.min(), ch.max()
            if mx - mn > 1e-6:
                rgb[:, :, c] = (ch - mn) / (mx - mn)
            else:
                rgb[:, :, c] = 0

        return (rgb * 255).astype(np.uint8)
