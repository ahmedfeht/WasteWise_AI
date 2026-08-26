"""Download and extract the public TrashNet resized dataset."""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path

DATASET_URL = "https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / "trashnet.zip"
    raw_dir = args.output / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading TrashNet to {archive} ...")
    urllib.request.urlretrieve(DATASET_URL, archive)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(raw_dir)
    print(f"Dataset ready at {raw_dir / 'dataset-resized'}")


if __name__ == "__main__":
    main()

