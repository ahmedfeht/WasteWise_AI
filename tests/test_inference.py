import numpy as np
import pytest
from PIL import Image

from wastewise.inference import (
    CLASS_NAMES,
    get_recommendation,
    normalise_pixels,
    prepare_image,
    rank_predictions,
)


def test_prepare_image_shape_and_range():
    image = Image.new("RGB", (640, 480), color=(255, 128, 0))
    batch = prepare_image(image)
    assert batch.shape == (1, 224, 224, 3)
    assert batch.dtype == np.float32
    normalised = normalise_pixels(batch)
    assert -1.0 <= float(normalised.min()) <= float(normalised.max()) <= 1.0


def test_predictions_are_ranked_descending():
    values = np.array([0.05, 0.10, 0.15, 0.20, 0.45, 0.05])
    ranked = rank_predictions(values)
    assert ranked[0]["category"] == "plastic"
    assert ranked[0]["confidence"] == pytest.approx(0.45)
    assert [item["confidence"] for item in ranked] == sorted(values, reverse=True)


def test_wrong_probability_count_is_rejected():
    with pytest.raises(ValueError):
        rank_predictions(np.array([0.5, 0.5]))


def test_every_class_has_a_recommendation():
    for class_name in CLASS_NAMES:
        assert len(get_recommendation(class_name)) > 25
