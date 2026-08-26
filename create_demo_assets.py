"""Select one correctly predicted test image per class for the live demo."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parent
TEST_DIR = PROJECT_ROOT / "data" / "processed" / "test"
OUTPUT_DIR = PROJECT_ROOT / "sample_images"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def main() -> None:
    class_names = json.loads((ARTIFACTS_DIR / "class_names.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(ARTIFACTS_DIR / "waste_model.keras", compile=False)
    dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=(224, 224),
        batch_size=32,
        class_names=class_names,
        label_mode="categorical",
        shuffle=False,
    )
    paths = np.array(dataset.file_paths)
    probabilities = model.predict(dataset, verbose=0)
    predicted = probabilities.argmax(axis=1)
    actual = np.concatenate([labels.numpy().argmax(axis=1) for _, labels in dataset])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for class_index, class_name in enumerate(class_names):
        candidates = np.where((actual == class_index) & (predicted == class_index))[0]
        if candidates.size == 0:
            candidates = np.where(actual == class_index)[0]
        selected = candidates[np.argmax(probabilities[candidates, class_index])]
        source = Path(paths[selected])
        destination = OUTPUT_DIR / f"demo_{class_name}.jpg"
        shutil.copy2(source, destination)
        rows.append(
            {
                "file": destination.name,
                "actual": class_name,
                "predicted": class_names[int(predicted[selected])],
                "confidence": f"{float(probabilities[selected, predicted[selected]]):.6f}",
            }
        )

    with (OUTPUT_DIR / "demo_predictions.csv").open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=["file", "actual", "predicted", "confidence"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {len(rows)} demo images to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

