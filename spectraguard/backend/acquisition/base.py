"""Abstract base class for hyperspectral image sources."""

from __future__ import annotations

import datetime as dt
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List

import numpy as np


@dataclass
class HSICube:
    """A single hyperspectral acquisition frame.

    Attributes:
        data: 3-D array (height, width, bands) — reflectance values.
        wavelengths: 1-D array of centre wavelengths in nm.
        timestamp: acquisition time.
        item_id: unique identifier for this item / frame.
        product_type: 'poultry' | 'leafy_greens'.
        metadata: arbitrary extra info (lot, position, etc.).
        contamination_mask: ground-truth boolean mask (simulator only).
        contamination_labels: per-pixel pathogen labels (simulator only).
    """

    data: np.ndarray                        # (H, W, B)
    wavelengths: np.ndarray                 # (B,)
    timestamp: dt.datetime = field(default_factory=dt.datetime.utcnow)
    item_id: str = ""
    product_type: str = "poultry"
    metadata: dict = field(default_factory=dict)
    contamination_mask: Optional[np.ndarray] = None       # (H, W) bool
    contamination_labels: Optional[np.ndarray] = None     # (H, W) str


class HSISource(ABC):
    """Common interface for all HSI data sources."""

    @abstractmethod
    def initialise(self) -> None:
        """Open the camera / load the file / start the simulator."""

    @abstractmethod
    def acquire(self) -> HSICube:
        """Return the next hyperspectral cube."""

    @abstractmethod
    def is_ready(self) -> bool:
        """True when the source can deliver frames."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    def get_dark_reference(self) -> Optional[np.ndarray]:
        """Return a dark-reference frame if available."""
        return None

    def get_white_reference(self) -> Optional[np.ndarray]:
        """Return a white-reference frame if available."""
        return None
