"""Shared inference helpers for the Streamlit application and tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

RECOMMENDATIONS = {
    "cardboard": "Flatten it, keep it dry, and place it in the paper/cardboard recycling bin.",
    "glass": "Rinse it, remove caps if required locally, and place it in the glass recycling bin.",
    "metal": "Rinse the item and place it in the metal or mixed-recycling bin.",
    "paper": "Keep it clean and dry, then place it in the paper recycling bin.",
    "plastic": "Check the recycling symbol, rinse the item, and use the plastic recycling bin.",
    "trash": "This item is likely general waste. Use the general-waste bin and follow local rules.",
}


def load_rgb_image(source: Any) -> Image.Image:
    """Load an image-like object, fix orientation and return RGB pixels."""
    image = Image.open(source)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def prepare_image(image: Image.Image) -> np.ndarray:
    """Resize an image for the self-contained Keras model."""
    resized = image.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
    array = np.asarray(resized, dtype=np.float32)
    return np.expand_dims(array, axis=0)


def normalise_pixels(batch: np.ndarray) -> np.ndarray:
    """Apply the same [-1, 1] normalisation used by MobileNetV2."""
    return (np.asarray(batch, dtype=np.float32) / 127.5) - 1.0


def rank_predictions(probabilities: np.ndarray, class_names: list[str] | None = None) -> list[dict[str, float | str]]:
    """Convert a probability vector to a highest-confidence-first result list."""
    labels = class_names or CLASS_NAMES
    values = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if values.size != len(labels):
        raise ValueError(f"Expected {len(labels)} probabilities, received {values.size}.")
    order = np.argsort(values)[::-1]
    return [
        {"category": labels[index], "confidence": float(values[index])}
        for index in order
    ]


def predict_image(model: Any, image: Image.Image, class_names: list[str] | None = None) -> list[dict[str, float | str]]:
    """Run the Keras model and return ranked class predictions."""
    batch = prepare_image(image)
    probabilities = model.predict(batch, verbose=0)[0]
    return rank_predictions(probabilities, class_names)


def get_recommendation(category: str) -> str:
    """Return a safe, simple recycling recommendation for one class."""
    return RECOMMENDATIONS.get(category.lower(), "Follow your local waste-separation guidelines.")


def find_model_path(project_root: Path) -> Path:
    """Return the expected trained-model path."""
    return project_root / "artifacts" / "waste_model.keras"
