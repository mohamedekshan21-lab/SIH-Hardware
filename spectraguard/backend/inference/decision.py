"""Item-level decision logic — aggregates per-pixel predictions into a verdict."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from backend.inference.classifier import CLASS_NAMES


@dataclass
class ItemDecision:
    """Decision result for a single scanned item."""
    verdict: str                    # PASS | REJECT | REVIEW
    contamination_score: float      # 0.0–1.0 aggregate score
    predicted_pathogen: Optional[str]  # dominant pathogen class or None
    confidence: float               # confidence in the verdict
    contaminated_area_pct: float    # % of product area flagged
    probability_map: np.ndarray     # per-pixel contamination probability
    class_map: np.ndarray           # per-pixel class predictions
    explanation: str                # plain-language explanation


class DecisionEngine:
    """Applies configurable thresholds to classifier output.

    Args:
        contamination_threshold: Score above this → REJECT.
        review_threshold: Score above this but below contamination → REVIEW.
        min_confidence: Below this confidence → REVIEW (insufficient certainty).
        min_area_pct: Minimum contaminated area to trigger a reject.
    """

    def __init__(
        self,
        contamination_threshold: float = 0.5,
        review_threshold: float = 0.3,
        min_confidence: float = 0.7,
        min_area_pct: float = 1.0,
    ):
        self.contamination_threshold = contamination_threshold
        self.review_threshold = review_threshold
        self.min_confidence = min_confidence
        self.min_area_pct = min_area_pct

    def decide(
        self,
        pixel_classes: np.ndarray,
        pixel_probs: np.ndarray,
        product_mask: np.ndarray,
        spatial_shape: tuple[int, int],
    ) -> ItemDecision:
        """Make an item-level decision from per-pixel predictions.

        Args:
            pixel_classes: (N,) per-pixel class predictions.
            pixel_probs: (N, C) per-pixel probability matrix.
            product_mask: (H, W) bool — which pixels are product.
            spatial_shape: (H, W) for reconstructing spatial maps.

        Returns:
            ItemDecision with verdict, score, and explanation.
        """
        if pixel_classes.shape[0] == 0:
            return ItemDecision(
                verdict="REVIEW",
                contamination_score=0.0,
                predicted_pathogen=None,
                confidence=0.0,
                contaminated_area_pct=0.0,
                probability_map=np.zeros(spatial_shape, dtype=np.float32),
                class_map=np.zeros(spatial_shape, dtype=int),
                explanation="No product pixels detected — item sent for manual review.",
            )

        # Per-pixel contamination probability (1 - P(healthy))
        contam_prob = 1.0 - pixel_probs[:, 0] if pixel_probs.shape[1] > 0 else np.zeros(len(pixel_classes))

        # Contamination score = mean probability of contamination
        contamination_score = float(np.mean(contam_prob))

        # Contaminated area = % of pixels flagged as non-healthy
        n_contaminated = int(np.sum(pixel_classes > 0))
        contaminated_area_pct = (n_contaminated / len(pixel_classes)) * 100

        # Dominant pathogen class (excluding healthy)
        pathogen_classes = pixel_classes[pixel_classes > 0]
        if len(pathogen_classes) > 0:
            dominant_class = int(np.bincount(pathogen_classes).argmax())
            if dominant_class < len(CLASS_NAMES):
                predicted_pathogen = CLASS_NAMES[dominant_class]
            else:
                predicted_pathogen = "Unknown"
        else:
            predicted_pathogen = None

        # Confidence = mean of max probabilities
        confidence = float(np.mean(np.max(pixel_probs, axis=1))) if pixel_probs.shape[0] > 0 else 0.0

        # Build spatial maps
        probability_map = np.zeros(spatial_shape, dtype=np.float32)
        class_map = np.zeros(spatial_shape, dtype=int)
        flat_mask = product_mask.ravel()
        product_indices = np.where(flat_mask)[0]
        for i, idx in enumerate(product_indices[:len(contam_prob)]):
            r, c = divmod(int(idx), spatial_shape[1])
            if r < spatial_shape[0] and c < spatial_shape[1]:
                probability_map[r, c] = contam_prob[i]
                class_map[r, c] = pixel_classes[i]

        # Decision logic
        verdict, explanation = self._apply_thresholds(
            contamination_score, confidence, contaminated_area_pct, predicted_pathogen
        )

        return ItemDecision(
            verdict=verdict,
            contamination_score=round(contamination_score, 4),
            predicted_pathogen=predicted_pathogen,
            confidence=round(confidence, 4),
            contaminated_area_pct=round(contaminated_area_pct, 1),
            probability_map=probability_map,
            class_map=class_map,
            explanation=explanation,
        )

    def _apply_thresholds(
        self,
        score: float,
        confidence: float,
        area_pct: float,
        pathogen: str | None,
    ) -> tuple[str, str]:
        """Apply thresholds to determine verdict and explanation."""

        # Low confidence → always REVIEW (fail-safe)
        if confidence < self.min_confidence:
            return "REVIEW", (
                f"Low classifier confidence ({confidence:.0%}). "
                f"Item diverted for manual inspection. Score: {score:.2f}."
            )

        # High contamination score + significant area
        if score >= self.contamination_threshold and area_pct >= self.min_area_pct:
            pathogen_str = f" ({pathogen})" if pathogen else ""
            return "REJECT", (
                f"Contamination detected{pathogen_str}: score {score:.2f} "
                f"exceeds threshold ({self.contamination_threshold}), "
                f"affecting {area_pct:.1f}% of product area."
            )

        # Moderate score → review
        if score >= self.review_threshold:
            return "REVIEW", (
                f"Borderline contamination signal: score {score:.2f} "
                f"(threshold: {self.contamination_threshold}). "
                f"Sent for manual review."
            )

        # Below all thresholds → PASS
        return "PASS", (
            f"Product clear: contamination score {score:.2f} "
            f"is below threshold ({self.review_threshold}). "
            f"Confidence: {confidence:.0%}."
        )
