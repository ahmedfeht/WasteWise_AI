# WasteWise AI

WasteWise AI is a BIT4443 Deep Learning group project. It classifies one uploaded waste image as **cardboard, glass, metal, paper, plastic, or trash**, reports a confidence score, and gives a simple disposal recommendation.

## Why the project meets the assignment

- Computer Vision application.
- Uses MobileNetV2 with ImageNet pre-trained weights.
- Applies transfer learning, then fine-tunes the final MobileNetV2 layers.
- Uses a public dataset and records cleaning, preprocessing and split statistics.
- Provides a Streamlit interface with image upload, prediction button, category and confidence.
- Includes test metrics, a classification report, training graphs and a confusion matrix.

## Project structure

```text
WasteWise_AI/
├── app.py                         # Person 3: Streamlit application
├── data_preprocessing.py          # Person 1: cleaning, resize and split
├── download_dataset.py            # Downloads public TrashNet data
├── train_model.py                 # Person 2: training and evaluation
├── wastewise/inference.py         # Shared inference logic
├── artifacts/                     # Trained model and evaluation evidence
├── data/evidence/                 # Dataset statistics and sample grid
├── sample_images/                 # Images for the live demonstration
├── tests/                         # Automated unit tests
├── PRESENTATION_SCRIPT.md         # Simple script divided across 3 people
└── TEAM_DETAILS.md                # Replace names and student IDs
```

## Quick start on macOS

Open the project folder in VS Code, then open **Terminal > New Terminal** and run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If the model file is already inside `artifacts/waste_model.keras`, you do not need to train it again.

## Full reproducible workflow

```bash
python download_dataset.py
python data_preprocessing.py
python train_model.py
python -m pytest -q
python -m streamlit run app.py
```

The preprocessing script performs these steps:

1. Opens every image and records corrupted files.
2. Uses a decoded-pixel hash to remove exact duplicates.
3. Creates deterministic 70/15/15 train, validation and test splits.
4. Fits every image to 224 × 224 RGB pixels.
5. Saves class statistics, a split manifest, a cleaning log and sample images.

MobileNetV2 normalisation to the range `[-1, 1]` is applied inside the saved Keras model. Training-only augmentation uses horizontal flipping, rotation, zoom and contrast.

## Dataset

TrashNet contains 2,527 images across six classes: 403 cardboard, 501 glass, 410 metal, 594 paper, 482 plastic and 137 trash images. Dataset source: Gary Thung and Mindy Yang, [TrashNet repository](https://github.com/garythung/trashnet).

## Important limitations

- TrashNet mostly contains a single object on a plain background, so real-world clutter can reduce accuracy.
- The classes are imbalanced; the trash class is the smallest.
- The app classifies the main image into one category and does not detect several separate objects.
- Recycling rules differ by location, so the recommendation is general guidance.

## Submission

The assignment asks for **slides and source code only**. Before submission, replace all placeholders in `TEAM_DETAILS.md` and the first slide, then test the live demo on the presentation laptop.

