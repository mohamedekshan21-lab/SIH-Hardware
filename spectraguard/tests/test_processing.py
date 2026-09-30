"""Unit tests for hyperspectral processing pipeline."""

import numpy as np
import pytest
from backend.acquisition.simulated import SimulatedHSISource
from backend.processing.calibration import apply_dark_white_correction
from backend.processing.smoothing import savitzky_golay_smooth
from backend.processing.normalization import snv
from backend.processing.pipeline import PreprocessingPipeline


def test_calibration_correction():
    data = np.full((10, 10, 5), 0.5, dtype=np.float32)
    dark = np.full((10, 10, 5), 0.05, dtype=np.float32)
    white = np.full((10, 10, 5), 0.95, dtype=np.float32)

    corrected = apply_dark_white_correction(data, dark, white)
    assert corrected.shape == (10, 10, 5)
    assert np.allclose(corrected, (0.5 - 0.05) / (0.95 - 0.05), atol=1e-4)


def test_snv_normalization():
    # Mean should be close to 0 and std close to 1 across spectra
    data = np.random.normal(5, 2, (10, 10, 50)).astype(np.float32)
    norm = snv(data)
    assert norm.shape == data.shape
    assert np.allclose(np.mean(norm, axis=-1), 0, atol=1e-5)
    assert np.allclose(np.std(norm, axis=-1), 1, atol=1e-5)


def test_savitzky_golay_smoothing():
    data = np.random.normal(0.5, 0.05, (8, 8, 30)).astype(np.float32)
    smoothed = savitzky_golay_smooth(data, window_length=7, polyorder=2)
    assert smoothed.shape == data.shape


def test_preprocessing_pipeline():
    source = SimulatedHSISource(height=32, width=32, n_bands=224)
    source.initialise()
    cube = source.acquire()

    pipeline = PreprocessingPipeline()
    pipeline.set_references(source.get_dark_reference(), source.get_white_reference())
    processed = pipeline.process(cube)

    assert processed.reflectance.shape == (32, 32, 224)
    assert processed.rgb_preview.shape == (32, 32, 3)
    assert len(processed.mean_spectrum) == 224
    source.close()
