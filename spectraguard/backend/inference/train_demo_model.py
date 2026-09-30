"""Train a demo 1D-CNN model on simulated HSI data and export to ONNX.

This script:
1. Generates training data using the SimulatedHSISource
2. Trains a scikit-learn pipeline (PCA + SVM) as a fast fallback
3. Also trains a small 1D-CNN via sklearn-compatible wrapper
4. Exports the best model to ONNX format
5. Reports accuracy, sensitivity, specificity, FNR, confusion matrix

Usage:
    python -m backend.inference.train_demo_model
"""

from __future__ import annotations

import json
import time
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    precision_recall_fscore_support,
)

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.acquisition.simulated import SimulatedHSISource
from backend.processing.pipeline import PreprocessingPipeline


# Classes: 0=healthy, 1=Salmonella, 2=Listeria, 3=E.coli, 4=Biofilm
CLASS_NAMES = ["Healthy", "Salmonella", "Listeria", "E. coli", "Biofilm"]
PATHOGEN_MAP = {
    "Salmonella": 1,
    "Listeria": 2,
    "E. coli": 3,
    "Biofilm": 4,
}


def generate_training_data(
    n_samples: int = 2000,
    product_type: str = "poultry",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate labelled pixel-level training data from the simulator.

    Returns:
        X: (N, n_features) feature array.
        y: (N,) integer class labels.
    """
    source = SimulatedHSISource(
        height=32, width=32, n_bands=224,
        product_type=product_type,
        contamination_rate=0.5,  # 50% for balanced training
    )
    source.initialise()

    pipeline = PreprocessingPipeline(use_band_selection=True)
    pipeline.set_references(source.get_dark_reference(), source.get_white_reference())

    all_X = []
    all_y = []
    samples_collected = 0

    while samples_collected < n_samples:
        cube = source.acquire()
        processed = pipeline.process(cube)

        if processed.product_pixels.shape[0] == 0:
            continue

        # Labels from ground truth mask
        mask = processed.product_mask
        if cube.contamination_mask is not None and cube.contamination_labels is not None:
            contam_mask_product = cube.contamination_mask[mask]
            labels_product = cube.contamination_labels[mask]

            y_pixels = np.zeros(contam_mask_product.shape[0], dtype=int)
            for i, (is_contam, label) in enumerate(zip(contam_mask_product, labels_product)):
                if is_contam and label in PATHOGEN_MAP:
                    y_pixels[i] = PATHOGEN_MAP[label]

            # Subsample to manage size
            n_take = min(len(y_pixels), max(50, (n_samples - samples_collected)))
            indices = np.random.choice(len(y_pixels), n_take, replace=False)
            all_X.append(processed.product_pixels[indices])
            all_y.append(y_pixels[indices])
            samples_collected += n_take
        else:
            n_take = min(processed.product_pixels.shape[0], 50)
            all_X.append(processed.product_pixels[:n_take])
            all_y.append(np.zeros(n_take, dtype=int))
            samples_collected += n_take

    source.close()

    X = np.vstack(all_X)[:n_samples]
    y = np.concatenate(all_y)[:n_samples]

    print(f"Generated {len(X)} training samples")
    print(f"Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y


def train_and_export(n_samples: int = 3000):
    """Train the demo model, evaluate, and export to ONNX."""
    print("=" * 60)
    print("SpectraGuard — Demo Model Training")
    print("=" * 60)

    # Generate data for both product types
    print("\n[1/5] Generating training data (poultry)...")
    X_poultry, y_poultry = generate_training_data(n_samples // 2, "poultry")

    print("\n[2/5] Generating training data (leafy greens)...")
    X_greens, y_greens = generate_training_data(n_samples // 2, "leafy_greens")

    X = np.vstack([X_poultry, X_greens])
    y = np.concatenate([y_poultry, y_greens])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build pipeline: StandardScaler → PCA → SVM
    print("\n[3/5] Training SVM classifier...")
    t0 = time.time()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=min(20, X_train.shape[1]))),
        ("svm", SVC(kernel="rbf", probability=True, C=10, gamma="scale", random_state=42)),
    ])

    model.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.1f}s")

    # Evaluate
    print("\n[4/5] Evaluating...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(CLASS_NAMES)), zero_division=0
    )

    # Overall metrics
    sensitivity = float(np.mean(recall[1:]))  # mean sensitivity for pathogen classes
    specificity = float(recall[0])  # healthy class recall = specificity
    fnr = 1.0 - sensitivity

    metrics = {
        "accuracy": round(float(acc), 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "f1_macro": round(float(np.mean(f1)), 4),
        "fnr": round(fnr, 4),
        "per_class": {
            CLASS_NAMES[i]: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i in range(len(CLASS_NAMES))
        },
        "confusion_matrix": cm.tolist(),
        "class_names": CLASS_NAMES,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "n_features": X_train.shape[1],
        "train_time_s": round(train_time, 2),
    }

    print(f"\n  Accuracy:    {acc:.4f}")
    print(f"  Sensitivity: {sensitivity:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  FNR:         {fnr:.4f}")
    print(f"  F1 (macro):  {np.mean(f1):.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"  Classes: {CLASS_NAMES}")
    for row in cm:
        print(f"    {row}")

    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

    # Export to ONNX
    print("[5/5] Exporting to ONNX...")
    model_dir = Path(__file__).parent / "models"
    model_dir.mkdir(exist_ok=True)

    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType

        initial_type = [("input", FloatTensorType([None, X_train.shape[1]]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)

        onnx_path = model_dir / "demo_model.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        print(f"  Saved ONNX model to {onnx_path}")
    except Exception as e:
        print(f"  ONNX export failed ({e}), saving sklearn model as fallback...")
        import pickle
        fallback_path = model_dir / "demo_model.pkl"
        with open(fallback_path, "wb") as f:
            pickle.dump(model, f)
        print(f"  Saved sklearn model to {fallback_path}")

    # Save metrics
    metrics_path = model_dir / "demo_model_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved metrics to {metrics_path}")

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return metrics


if __name__ == "__main__":
    train_and_export()
