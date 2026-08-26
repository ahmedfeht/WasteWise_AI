"""Train, fine-tune and evaluate a MobileNetV2 waste classifier."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
IMAGE_SIZE = (224, 224)
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]


def set_reproducible_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def load_datasets(processed_dir: Path, batch_size: int) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    common = {
        "image_size": IMAGE_SIZE,
        "batch_size": batch_size,
        "label_mode": "categorical",
        "class_names": CLASS_NAMES,
        "seed": SEED,
    }
    train = tf.keras.utils.image_dataset_from_directory(processed_dir / "train", shuffle=True, **common)
    validation = tf.keras.utils.image_dataset_from_directory(processed_dir / "validation", shuffle=False, **common)
    test = tf.keras.utils.image_dataset_from_directory(processed_dir / "test", shuffle=False, **common)
    autotune = tf.data.AUTOTUNE
    return train.prefetch(autotune), validation.prefetch(autotune), test.prefetch(autotune)


def build_model() -> tuple[tf.keras.Model, tf.keras.Model]:
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.08),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.10),
        ],
        name="augmentation",
    )
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMAGE_SIZE + (3,), include_top=False, weights="imagenet"
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMAGE_SIZE + (3,), name="image")
    x = augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.30)(x)
    outputs = tf.keras.layers.Dense(len(CLASS_NAMES), activation="softmax", name="waste_category")(x)
    return tf.keras.Model(inputs, outputs, name="wastewise_mobilenetv2"), base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")],
    )


def combine_histories(*histories: tf.keras.callbacks.History) -> pd.DataFrame:
    frames = []
    starting_epoch = 0
    for phase, history in enumerate(histories, start=1):
        frame = pd.DataFrame(history.history)
        frame.insert(0, "epoch", np.arange(starting_epoch + 1, starting_epoch + len(frame) + 1))
        frame.insert(1, "phase", phase)
        frames.append(frame)
        starting_epoch += len(frame)
    return pd.concat(frames, ignore_index=True)


def save_training_curves(history: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(history["epoch"], history["accuracy"], marker="o", label="Train")
    axes[0].plot(history["epoch"], history["val_accuracy"], marker="o", label="Validation")
    axes[0].set(title="Accuracy by epoch", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1))
    axes[1].plot(history["epoch"], history["loss"], marker="o", label="Train")
    axes[1].plot(history["epoch"], history["val_loss"], marker="o", label="Validation")
    axes[1].set(title="Loss by epoch", xlabel="Epoch", ylabel="Loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def evaluate(model: tf.keras.Model, test: tf.data.Dataset, artifacts_dir: Path) -> dict[str, float]:
    raw_metrics = model.evaluate(test, verbose=1, return_dict=True)
    probabilities = model.predict(test, verbose=1)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.concatenate([np.argmax(labels.numpy(), axis=1) for _, labels in test])

    report = classification_report(
        y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(artifacts_dir / "classification_report.csv")

    matrix = confusion_matrix(y_true, y_pred)
    figure, axis = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=[name.title() for name in CLASS_NAMES],
        yticklabels=[name.title() for name in CLASS_NAMES],
        ax=axis,
    )
    axis.set(title="MobileNetV2 confusion matrix", xlabel="Predicted", ylabel="Actual")
    figure.tight_layout()
    figure.savefig(artifacts_dir / "confusion_matrix.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    summary = {
        "test_loss": float(raw_metrics["loss"]),
        "test_accuracy": float(raw_metrics["accuracy"]),
        "test_precision": float(raw_metrics["precision"]),
        "test_recall": float(raw_metrics["recall"]),
        "macro_f1": float(report["macro avg"]["f1-score"]),
        "weighted_f1": float(report["weighted avg"]["f1-score"]),
        "test_images": int(len(y_true)),
    }
    (artifacts_dir / "evaluation_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    args = parser.parse_args()

    set_reproducible_seed()
    args.artifacts.mkdir(parents=True, exist_ok=True)
    train, validation, test = load_datasets(args.processed, args.batch_size)
    model, base_model = build_model()
    compile_model(model, 1e-3)

    common_callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1, min_lr=1e-7),
    ]
    head_history = model.fit(
        train, validation_data=validation, epochs=args.head_epochs, callbacks=common_callbacks
    )

    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
    compile_model(model, 1e-5)
    fine_history = model.fit(
        train,
        validation_data=validation,
        epochs=args.fine_tune_epochs,
        callbacks=common_callbacks,
    )

    history = combine_histories(head_history, fine_history)
    history.to_csv(args.artifacts / "training_history.csv", index=False)
    save_training_curves(history, args.artifacts / "training_curves.png")
    model.save(args.artifacts / "waste_model.keras")
    (args.artifacts / "class_names.json").write_text(json.dumps(CLASS_NAMES, indent=2), encoding="utf-8")
    summary = evaluate(model, test, args.artifacts)
    with (args.artifacts / "model_summary.txt").open("w", encoding="utf-8") as summary_file:
        model.summary(print_fn=lambda line: summary_file.write(line + "\n"))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
