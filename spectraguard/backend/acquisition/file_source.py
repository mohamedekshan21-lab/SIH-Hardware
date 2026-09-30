"""File-based HSI source — replays ENVI or .npy hyperspectral cubes."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Optional

import numpy as np

from backend.acquisition.base import HSICube, HSISource


class FileHSISource(HSISource):
    """Replay saved hyperspectral cubes from .npy files or ENVI format.

    Args:
        directory: Path containing .npy files (each is one H×W×B cube).
        wavelength_file: Optional .npy with the wavelength axis.
        loop: Whether to loop back to the start after all files are read.
    """

    def __init__(
        self,
        directory: str | Path,
        wavelength_file: str | Path | None = None,
        loop: bool = True,
    ):
        self.directory = Path(directory)
        self.wavelength_file = Path(wavelength_file) if wavelength_file else None
        self.loop = loop

        self._files: list[Path] = []
        self._index = 0
        self._wavelengths: Optional[np.ndarray] = None
        self._ready = False

    def initialise(self) -> None:
        if not self.directory.exists():
            raise FileNotFoundError(f"HSI data directory not found: {self.directory}")

        self._files = sorted(self.directory.glob("*.npy"))
        if not self._files:
            # Try ENVI .hdr/.dat pairs
            self._files = sorted(self.directory.glob("*.dat"))
        if not self._files:
            raise FileNotFoundError(f"No .npy or .dat files in {self.directory}")

        if self.wavelength_file and self.wavelength_file.exists():
            self._wavelengths = np.load(str(self.wavelength_file))
        else:
            # Default NIR range
            self._wavelengths = np.linspace(900, 1700, 224)

        self._index = 0
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready and len(self._files) > 0

    def close(self) -> None:
        self._ready = False

    def acquire(self) -> HSICube:
        if not self.is_ready():
            raise RuntimeError("FileHSISource not ready")

        filepath = self._files[self._index]
        data = np.load(str(filepath)).astype(np.float32)

        # Advance index
        self._index += 1
        if self._index >= len(self._files):
            if self.loop:
                self._index = 0
            else:
                self._ready = False

        # Ensure wavelength axis matches
        if data.ndim == 3 and data.shape[2] != len(self._wavelengths):
            self._wavelengths = np.linspace(900, 1700, data.shape[2])

        return HSICube(
            data=data,
            wavelengths=self._wavelengths.copy(),
            timestamp=dt.datetime.utcnow(),
            item_id=filepath.stem,
            product_type="unknown",
            metadata={"source_file": str(filepath)},
        )
