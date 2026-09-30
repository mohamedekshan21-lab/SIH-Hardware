"""Simulated HSI source with realistic spectra and contamination injection.

Generates hyperspectral cubes based on literature-derived spectral profiles
for poultry and leafy greens, with pathogen-specific perturbations.
"""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Optional

import numpy as np

from backend.acquisition.base import HSICube, HSISource

# ---------------------------------------------------------------------------
# Spectral profiles (based on published NIR absorption features)
# ---------------------------------------------------------------------------
# Centre wavelengths where key constituents absorb in the 900–1700 nm range:
#   Water:    ~970 nm, ~1190 nm, ~1450 nm
#   Protein:  ~1020 nm, ~1510 nm
#   Fat:      ~1210 nm, ~1390 nm
#   Cellulose:~1200 nm, ~1490 nm
# Pathogen biofilms alter these in subtle ways detectable by hyperspectral imaging.

PATHOGEN_PROFILES = {
    "Salmonella": {
        "absorption_centres": [1050, 1180],
        "absorption_widths": [25, 20],
        "absorption_depths": [0.12, 0.08],
        "fluorescence_shift": 0.05,
    },
    "Listeria": {
        "absorption_centres": [1150, 1350],
        "absorption_widths": [30, 25],
        "absorption_depths": [0.15, 0.10],
        "fluorescence_shift": 0.07,
    },
    "E. coli": {
        "absorption_centres": [980, 1280],
        "absorption_widths": [20, 30],
        "absorption_depths": [0.10, 0.12],
        "fluorescence_shift": 0.04,
    },
    "Biofilm": {
        "absorption_centres": [1100, 1300, 1500],
        "absorption_widths": [40, 35, 30],
        "absorption_depths": [0.08, 0.06, 0.09],
        "fluorescence_shift": 0.03,
    },
}

PRODUCT_PROFILES = {
    "poultry": {
        "base_reflectance": 0.65,
        "water_depth": 0.30,
        "protein_depth": 0.20,
        "fat_depth": 0.15,
        "texture_noise": 0.03,
    },
    "leafy_greens": {
        "base_reflectance": 0.50,
        "water_depth": 0.35,
        "protein_depth": 0.08,
        "fat_depth": 0.02,
        "texture_noise": 0.04,
        "chlorophyll_depth": 0.25,
    },
}


class SimulatedHSISource(HSISource):
    """Generates realistic hyperspectral cubes with optional contamination.

    Args:
        n_bands: Number of spectral bands.
        wl_start: Start wavelength (nm).
        wl_end: End wavelength (nm).
        height: Spatial height of the cube.
        width: Spatial width of the cube.
        product_type: 'poultry' or 'leafy_greens'.
        contamination_rate: Probability of contamination per item.
        noise_std: Gaussian noise level.
        drift_rate: Lighting drift per frame (multiplicative).
        force_contamination: If set, always inject this pathogen.
    """

    def __init__(
        self,
        n_bands: int = 224,
        wl_start: float = 900.0,
        wl_end: float = 1700.0,
        height: int = 64,
        width: int = 64,
        product_type: str = "poultry",
        contamination_rate: float = 0.08,
        noise_std: float = 0.015,
        drift_rate: float = 0.001,
        force_contamination: Optional[str] = None,
    ):
        self.n_bands = n_bands
        self.wavelengths = np.linspace(wl_start, wl_end, n_bands)
        self.height = height
        self.width = width
        self.product_type = product_type
        self.contamination_rate = contamination_rate
        self.noise_std = noise_std
        self.drift_rate = drift_rate
        self.force_contamination = force_contamination

        self._ready = False
        self._frame_count = 0
        self._drift_factor = 1.0
        self._rng = np.random.default_rng(seed=None)

        # References
        self._dark_ref: Optional[np.ndarray] = None
        self._white_ref: Optional[np.ndarray] = None

    def initialise(self) -> None:
        self._dark_ref = self._rng.normal(0.02, 0.005, (self.height, self.width, self.n_bands)).clip(0)
        self._white_ref = np.ones((self.height, self.width, self.n_bands)) * 0.95
        self._white_ref += self._rng.normal(0, 0.01, self._white_ref.shape)
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def close(self) -> None:
        self._ready = False

    def get_dark_reference(self) -> Optional[np.ndarray]:
        return self._dark_ref

    def get_white_reference(self) -> Optional[np.ndarray]:
        return self._white_ref

    def acquire(self) -> HSICube:
        if not self._ready:
            raise RuntimeError("Source not initialised")

        self._frame_count += 1
        # Slow lighting drift
        self._drift_factor += self._rng.normal(0, self.drift_rate)
        self._drift_factor = np.clip(self._drift_factor, 0.9, 1.1)

        profile = PRODUCT_PROFILES[self.product_type]
        cube = self._generate_product_spectrum(profile)

        # Background mask (belt = low reflectance)
        bg_mask = self._generate_background_mask()
        cube[bg_mask] *= 0.1

        # Contamination
        contam_mask = np.zeros((self.height, self.width), dtype=bool)
        contam_labels = np.full((self.height, self.width), "", dtype=object)
        pathogen = None

        if self.force_contamination:
            inject = True
            pathogen = self.force_contamination
        else:
            inject = self._rng.random() < self.contamination_rate

        if inject:
            pathogens = list(PATHOGEN_PROFILES.keys())
            pathogen = pathogen or self._rng.choice(pathogens)
            contam_mask, contam_labels = self._inject_contamination(cube, pathogen)

        # Add sensor noise + drift
        cube *= self._drift_factor
        cube += self._rng.normal(0, self.noise_std, cube.shape)
        cube = np.clip(cube, 0, 1)

        item_id = f"ITEM-{uuid.uuid4().hex[:8].upper()}"

        return HSICube(
            data=cube.astype(np.float32),
            wavelengths=self.wavelengths.copy(),
            timestamp=dt.datetime.utcnow(),
            item_id=item_id,
            product_type=self.product_type,
            metadata={
                "frame": self._frame_count,
                "drift_factor": round(float(self._drift_factor), 4),
                "contaminated": bool(contam_mask.any()),
                "pathogen": pathogen if contam_mask.any() else None,
                "contaminated_area_pct": round(float(contam_mask.sum() / contam_mask.size * 100), 1),
            },
            contamination_mask=contam_mask,
            contamination_labels=contam_labels,
        )

    # ── Private helpers ───────────────────────────────────────────────────

    def _generate_product_spectrum(self, profile: dict) -> np.ndarray:
        """Build a base spectral cube for the product."""
        wl = self.wavelengths
        base = np.full((self.height, self.width, self.n_bands), profile["base_reflectance"])

        # Water absorption
        for centre in [970, 1190, 1450]:
            absorption = profile["water_depth"] * np.exp(-0.5 * ((wl - centre) / 40) ** 2)
            base -= absorption[np.newaxis, np.newaxis, :]

        # Protein
        for centre in [1020, 1510]:
            absorption = profile["protein_depth"] * np.exp(-0.5 * ((wl - centre) / 30) ** 2)
            base -= absorption[np.newaxis, np.newaxis, :]

        # Fat
        for centre in [1210, 1390]:
            absorption = profile["fat_depth"] * np.exp(-0.5 * ((wl - centre) / 25) ** 2)
            base -= absorption[np.newaxis, np.newaxis, :]

        # Chlorophyll (leafy greens only)
        if "chlorophyll_depth" in profile:
            for centre in [960, 1100]:
                absorption = profile["chlorophyll_depth"] * np.exp(-0.5 * ((wl - centre) / 35) ** 2)
                base -= absorption[np.newaxis, np.newaxis, :]

        # Spatial texture variation
        texture = self._rng.normal(0, profile["texture_noise"], (self.height, self.width, 1))
        base += texture

        return np.clip(base, 0.05, 0.95)

    def _generate_background_mask(self) -> np.ndarray:
        """Product occupies central region; edges are belt (background)."""
        mask = np.ones((self.height, self.width), dtype=bool)
        pad_h = self.height // 8
        pad_w = self.width // 8
        mask[pad_h:-pad_h, pad_w:-pad_w] = False

        # Irregular product edge
        for _ in range(3):
            r = self._rng.integers(pad_h, self.height - pad_h)
            c = self._rng.integers(pad_w, self.width - pad_w)
            radius = self._rng.integers(2, 6)
            rr, cc = np.ogrid[-r:self.height - r, -c:self.width - c]
            circle = rr ** 2 + cc ** 2 <= radius ** 2
            mask[circle] = True

        return mask

    def _inject_contamination(
        self, cube: np.ndarray, pathogen: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Add contamination patches to the cube."""
        prof = PATHOGEN_PROFILES[pathogen]
        wl = self.wavelengths
        mask = np.zeros((self.height, self.width), dtype=bool)
        labels = np.full((self.height, self.width), "", dtype=object)

        # 1–3 contamination patches
        n_patches = self._rng.integers(1, 4)
        for _ in range(n_patches):
            # Random patch location (within product area)
            pad_h = self.height // 6
            pad_w = self.width // 6
            cy = self._rng.integers(pad_h + 2, self.height - pad_h - 2)
            cx = self._rng.integers(pad_w + 2, self.width - pad_w - 2)
            radius = self._rng.integers(3, max(4, self.height // 8))

            # Severity variation (0.3 = subtle, 1.0 = severe)
            severity = self._rng.uniform(0.4, 1.0)

            rr, cc = np.ogrid[-cy:self.height - cy, -cx:self.width - cx]
            patch = rr ** 2 + cc ** 2 <= radius ** 2
            mask |= patch
            labels[patch] = pathogen

            # Apply spectral perturbation
            for centre, width, depth in zip(
                prof["absorption_centres"],
                prof["absorption_widths"],
                prof["absorption_depths"],
            ):
                perturbation = severity * depth * np.exp(-0.5 * ((wl - centre) / width) ** 2)
                cube[patch] -= perturbation[np.newaxis, :]

            # Fluorescence shift
            if prof["fluorescence_shift"] > 0:
                cube[patch, :20] += severity * prof["fluorescence_shift"]

        return mask, labels
