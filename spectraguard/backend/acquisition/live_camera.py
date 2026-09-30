"""Live camera HSI source — stub for real sensor SDK integration.

TODO: Replace with actual camera SDK calls. See docs/hardware_integration.md
for the interface contract and timing requirements.
"""

from __future__ import annotations

import datetime as dt
from typing import Optional

import numpy as np

from backend.acquisition.base import HSICube, HSISource


class LiveCameraSource(HSISource):
    """Stub for a real hyperspectral camera.

    To integrate a real camera:
    1. Install the vendor SDK (e.g., Specim, Headwall, Resonon).
    2. Implement initialise() to open the camera connection.
    3. Implement acquire() to grab a frame and return an HSICube.
    4. Implement close() to release the camera handle.
    5. Set the wavelength calibration from the camera's .hdr metadata.

    Args:
        camera_id: SDK-specific camera identifier or serial number.
        exposure_ms: Integration time in milliseconds.
        n_bands: Number of spectral bands the camera captures.
        wl_start: First wavelength (nm).
        wl_end: Last wavelength (nm).
    """

    def __init__(
        self,
        camera_id: str = "CAM-001",
        exposure_ms: float = 10.0,
        n_bands: int = 224,
        wl_start: float = 900.0,
        wl_end: float = 1700.0,
    ):
        self.camera_id = camera_id
        self.exposure_ms = exposure_ms
        self.n_bands = n_bands
        self.wavelengths = np.linspace(wl_start, wl_end, n_bands)
        self._ready = False
        self._handle = None  # TODO: camera SDK handle

    def initialise(self) -> None:
        """TODO: Open the camera connection via the vendor SDK.

        Example (Specim FX-series):
            import specim_sdk
            self._handle = specim_sdk.open(self.camera_id)
            self._handle.set_exposure(self.exposure_ms)
            self._handle.set_binning(1)
            # Read wavelength calibration
            self.wavelengths = self._handle.get_wavelengths()
            self._ready = True
        """
        raise NotImplementedError(
            "LiveCameraSource requires a real camera SDK. "
            "Use SimulatedHSISource for demo mode. "
            "See docs/hardware_integration.md for integration guide."
        )

    def acquire(self) -> HSICube:
        """TODO: Grab a frame from the camera.

        Example:
            raw_frame = self._handle.grab_frame()  # (H, W, B) uint16
            data = raw_frame.astype(np.float32) / 65535.0
            return HSICube(data=data, wavelengths=self.wavelengths, ...)
        """
        if not self._ready:
            raise RuntimeError("Camera not initialised")

        # TODO: Replace with real acquisition
        raise NotImplementedError("LiveCameraSource.acquire() not implemented")

    def is_ready(self) -> bool:
        return self._ready

    def close(self) -> None:
        """TODO: Close the camera connection.

        Example:
            if self._handle:
                self._handle.close()
            self._ready = False
        """
        self._ready = False

    def get_dark_reference(self) -> Optional[np.ndarray]:
        """TODO: Capture dark reference (lens cap on).

        Example:
            self._handle.set_shutter_closed(True)
            dark = self._handle.grab_frame().astype(np.float32) / 65535.0
            self._handle.set_shutter_closed(False)
            return dark
        """
        return None

    def get_white_reference(self) -> Optional[np.ndarray]:
        """TODO: Capture white reference (Spectralon target).

        Example:
            # Prompt operator to place white reference panel
            white = self._handle.grab_frame().astype(np.float32) / 65535.0
            return white
        """
        return None
