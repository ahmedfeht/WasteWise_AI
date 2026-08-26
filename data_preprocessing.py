"""Clean TrashNet, create reproducible splits and prepare dataset evidence."""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps, UnidentifiedImageError

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_SIZE = (224, 224)
SEED = 42


def image_digest(path: Path) -> str:
    """Hash decoded RGB pixels so identical images with different names are detected."""
    with Image.open(path) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        return hashlib.sha256(rgb.tobytes()).hexdigest()


def scan_dataset(source_dir: Path) -> tuple[dict[str, list[Path]], list[dict[str, str]]]:
    """Validate images and remove exact duplicates."""
    clean_files: dict[str, list[Path]] = defaultdict(list)
    issues: list[dict[str, str]] = []
    seen_hashes: set[str] = set()

    for class_name in CLASS_NAMES:
        class_dir = source_dir / class_name
        if not class_dir.exists():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        for path in sorted(class_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in VALID_EXTENSIONS:
                continue
            try:
                digest = image_digest(path)
                if digest in seen_hashes:
                    issues.append({"file": str(path), "reason": "duplicate"})
                    continue
                seen_hashes.add(digest)
                clean_files[class_name].append(path)
            except (OSError, UnidentifiedImageError, ValueError) as error:
                issues.append({"file": str(path), "reason": f"corrupted: {error}"})
    return dict(clean_files), issues


def stratified_split(files_by_class: dict[str, list[Path]]) -> dict[str, dict[str, list[Path]]]:
    """Create deterministic 70/15/15 class-stratified train/validation/test splits."""
    rng = random.Random(SEED)
    result = {"train": {}, "validation": {}, "test": {}}
    for class_name in CLASS_NAMES:
        paths = list(files_by_class[class_name])
        rng.shuffle(paths)
        total = len(paths)
        train_end = int(total * 0.70)
        validation_end = train_end + int(total * 0.15)
        result["train"][class_name] = paths[:train_end]
        result["validation"][class_name] = paths[train_end:validation_end]
        result["test"][class_name] = paths[validation_end:]
    return result


def save_resized_splits(splits: dict[str, dict[str, list[Path]]], output_dir: Path) -> pd.DataFrame:
    """Resize images to 224x224, save them by split and return a manifest."""
    if output_dir.exists():
        shutil.rmtree(output_dir)
    rows: list[dict[str, str]] = []
    for split_name, by_class in splits.items():
        for class_name, paths in by_class.items():
            destination_dir = output_dir / split_name / class_name
            destination_dir.mkdir(parents=True, exist_ok=True)
            for number, source_path in enumerate(paths, start=1):
                destination = destination_dir / f"{class_name}_{number:04d}.jpg"
                with Image.open(source_path) as image:
                    image = ImageOps.exif_transpose(image).convert("RGB")
                    image = ImageOps.fit(image, IMAGE_SIZE, method=Image.Resampling.LANCZOS)
                    image.save(destination, "JPEG", quality=92, optimize=True)
                rows.append(
                    {
                        "split": split_name,
                        "class": class_name,
                        "source": str(source_path),
                        "processed": str(destination),
                    }
                )
    return pd.DataFrame(rows)


def save_statistics(manifest: pd.DataFrame, issues: list[dict[str, str]], output_dir: Path) -> pd.DataFrame:
    """Save class and split counts used in the report and presentation."""
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = manifest.groupby(["class", "split"]).size().unstack(fill_value=0)
    for column in ["train", "validation", "test"]:
        if column not in counts.columns:
            counts[column] = 0
    counts = counts[["train", "validation", "test"]]
    counts["total"] = counts.sum(axis=1)
    counts.loc["TOTAL"] = counts.sum(axis=0)
    counts.reset_index().to_csv(output_dir / "dataset_statistics.csv", index=False)
    manifest.to_csv(output_dir / "split_manifest.csv", index=False)
    pd.DataFrame(issues, columns=["file", "reason"]).to_csv(output_dir / "cleaning_log.csv", index=False)
    return counts


def save_sample_grid(manifest: pd.DataFrame, output_path: Path) -> None:
    """Create one representative processed image per class."""
    figure, axes = plt.subplots(2, 3, figsize=(11, 7))
    for axis, class_name in zip(axes.flat, CLASS_NAMES):
        record = manifest[(manifest["split"] == "train") & (manifest["class"] == class_name)].iloc[0]
        with Image.open(record["processed"]) as image:
            axis.imshow(image)
        axis.set_title(class_name.title(), fontsize=14, fontweight="bold")
        axis.axis("off")
    figure.suptitle("TrashNet sample images after 224 × 224 preprocessing", fontsize=17, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/raw/dataset-resized"))
    parser.add_argument("--processed", type=Path, default=Path("data/processed"))
    parser.add_argument("--evidence", type=Path, default=Path("data/evidence"))
    args = parser.parse_args()

    files_by_class, issues = scan_dataset(args.source)
    splits = stratified_split(files_by_class)
    manifest = save_resized_splits(splits, args.processed)
    counts = save_statistics(manifest, issues, args.evidence)
    save_sample_grid(manifest, args.evidence / "sample_images.png")

    print("Dataset preprocessing completed.")
    print(counts.to_string())
    print(f"Removed files: {Counter(item['reason'] for item in issues)}")


if __name__ == "__main__":
    main()

