"""ONNX inference wrapper for per-pixel classification."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

CLASS_NAMES = ["Healthy", "Salmonella", "Listeria", "E. coli", "Biofilm"]


class HSIClassifier:
    """Runs per-pixel pathogen classification via ONNX Runtime or sklearn fallback.

    Args:
        model_path: Path to .onnx or .pkl model file.
    """

    def __init__(self, model_path: str | Path | None = None):
        self.model_path = Path(model_path) if model_path else None
        self._session = None
        self._sklearn_model = None
        self._input_name: Optional[str] = None
        self._ready = False

    def load(self):
        """Load the model (ONNX preferred, sklearn fallback)."""
        if self.model_path is None:
            # Try default locations
            base = Path(__file__).parent / "models"
            onnx_path = base / "demo_model.onnx"
            pkl_path = base / "demo_model.pkl"
            if onnx_path.exists():
                self.model_path = onnx_path
            elif pkl_path.exists():
                self.model_path = pkl_path
            else:
                raise FileNotFoundError(
                    f"No model found in {base}. Run train_demo_model.py first."
                )

        if str(self.model_path).endswith(".onnx"):
            import onnxruntime as ort
            self._session = ort.InferenceSession(
                str(self.model_path),
                providers=["CPUExecutionProvider"],
            )
            self._input_name = self._session.get_inputs()[0].name
        elif str(self.model_path).endswith(".pkl"):
            import pickle
            with open(self.model_path, "rb") as f:
                self._sklearn_model = pickle.load(f)
        else:
            raise ValueError(f"Unsupported model format: {self.model_path}")

        self._ready = True

    def is_ready(self) -> bool:
        return self._ready

    def predict(self, pixels: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
        """Run per-pixel classification.

        Args:
            pixels: (N, n_features) — preprocessed pixel spectra.

        Returns:
            classes: (N,) integer class predictions.
            probabilities: (N, n_classes) probability matrix.
            latency_ms: Inference time in milliseconds.
        """
        if not self._ready:
            raise RuntimeError("Model not loaded. Call load() first.")

        if pixels.shape[0] == 0:
            return (
                np.array([], dtype=int),
                np.zeros((0, len(CLASS_NAMES)), dtype=np.float32),
                0.0,
            )

        t0 = time.perf_counter()

        if self._session is not None:
            # ONNX inference
            input_data = pixels.astype(np.float32)
            outputs = self._session.run(None, {self._input_name: input_data})
            classes = outputs[0].astype(int)
            # ONNX classifiers output probabilities as list of dicts
            if len(outputs) > 1 and isinstance(outputs[1], list):
                probs = np.zeros((len(classes), len(CLASS_NAMES)), dtype=np.float32)
                for i, prob_dict in enumerate(outputs[1]):
                    for cls_idx, prob_val in prob_dict.items():
                        probs[i, int(cls_idx)] = prob_val
            elif len(outputs) > 1:
                probs = np.array(outputs[1], dtype=np.float32)
                if probs.ndim == 1:
                    probs = probs.reshape(1, -1)
            else:
                probs = np.zeros((len(classes), len(CLASS_NAMES)), dtype=np.float32)
                for i, c in enumerate(classes):
                    probs[i, c] = 1.0
        else:
            # Sklearn fallback
            classes = self._sklearn_model.predict(pixels)
            probs = self._sklearn_model.predict_proba(pixels).astype(np.float32)

        latency_ms = (time.perf_counter() - t0) * 1000

        return classes.astype(int), probs, latency_ms
