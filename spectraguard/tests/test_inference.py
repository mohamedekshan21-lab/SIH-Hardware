"""Unit tests for ML classifier and item-level decision engine."""

import numpy as np
import pytest
from backend.inference.decision import DecisionEngine, ItemDecision
from backend.inference.classifier import HSIClassifier


def test_decision_engine_pass():
    engine = DecisionEngine(contamination_threshold=0.5, review_threshold=0.3)
    # All pixels healthy (class 0, prob [1, 0, 0, 0, 0])
    pixel_classes = np.zeros(100, dtype=int)
    pixel_probs = np.zeros((100, 5), dtype=np.float32)
    pixel_probs[:, 0] = 1.0

    product_mask = np.ones((10, 10), dtype=bool)

    decision = engine.decide(
        pixel_classes=pixel_classes,
        pixel_probs=pixel_probs,
        product_mask=product_mask,
        spatial_shape=(10, 10),
    )

    assert decision.verdict == "PASS"
    assert decision.contamination_score == 0.0
    assert decision.contaminated_area_pct == 0.0
    assert decision.predicted_pathogen is None


def test_decision_engine_reject():
    engine = DecisionEngine(contamination_threshold=0.5, review_threshold=0.3)
    # 80 pixels Salmonella (class 1)
    pixel_classes = np.ones(100, dtype=int)
    pixel_probs = np.zeros((100, 5), dtype=np.float32)
    pixel_probs[:, 1] = 0.9  # High pathogen prob

    product_mask = np.ones((10, 10), dtype=bool)

    decision = engine.decide(
        pixel_classes=pixel_classes,
        pixel_probs=pixel_probs,
        product_mask=product_mask,
        spatial_shape=(10, 10),
    )

    assert decision.verdict == "REJECT"
    assert decision.contamination_score > 0.5
    assert decision.predicted_pathogen == "Salmonella"


def test_decision_engine_low_confidence_failsafe():
    engine = DecisionEngine(min_confidence=0.7)
    pixel_classes = np.zeros(100, dtype=int)
    pixel_probs = np.full((100, 5), 0.2, dtype=np.float32)  # Max prob 0.2 < 0.7

    product_mask = np.ones((10, 10), dtype=bool)

    decision = engine.decide(
        pixel_classes=pixel_classes,
        pixel_probs=pixel_probs,
        product_mask=product_mask,
        spatial_shape=(10, 10),
    )

    assert decision.verdict == "REVIEW"
    assert "Low classifier confidence" in decision.explanation


def test_classifier_predict_shape():
    classifier = HSIClassifier()
    classifier.load()

    assert classifier.is_ready()
    n_features = classifier._sklearn_model.n_features_in_ if classifier._sklearn_model else 103
    pixels = np.random.normal(0, 1, (10, n_features)).astype(np.float32)
    classes, probs, latency = classifier.predict(pixels)

    assert len(classes) == 10
    assert probs.shape == (10, 5)
    assert latency > 0

