# WasteWise AI — Project Documentation

## 1. Problem definition

Waste is often placed in the wrong bin because people are unsure about its material. Incorrect sorting can contaminate recyclable materials and makes recycling less efficient. WasteWise AI provides an easy image-based first suggestion for households, schools, recycling centres and waste-management organisations.

## 2. Objectives

1. Classify an uploaded waste image into cardboard, glass, metal, paper, plastic or trash.
2. Display the predicted category and confidence score in a simple Streamlit application.
3. Give a short recycling recommendation for the predicted category.
4. Evaluate the model using unseen test images, accuracy, precision, recall, F1-score and a confusion matrix.

## 3. Dataset and preprocessing

The project uses the public TrashNet dataset by Gary Thung and Mindy Yang. It contains 2,527 resized images in six folders. Each file is opened with Pillow to detect corrupted images, and decoded RGB pixels are hashed to detect exact duplicates.

Three exact duplicate images were removed, leaving 2,524 cleaned images. They are split using a fixed random seed of 42:

- Training: 70% (1,764 images)
- Validation: 15% (377 images)
- Testing: 15% (383 images)

Images are converted to RGB and fitted to 224 × 224 pixels. MobileNetV2 preprocessing scales pixels to `[-1, 1]`. The training pipeline adds random horizontal flip, rotation, zoom and contrast to reduce overfitting.

## 4. Model selection and adaptation

MobileNetV2 is used because it is smaller and more efficient than many large convolutional neural networks while still providing strong pre-trained visual features. The convolutional base starts with ImageNet weights. The original ImageNet classification head is removed and replaced with:

- Global Average Pooling
- Dropout (0.30)
- Dense Softmax layer with six outputs

Training uses two phases. First, the MobileNetV2 base is frozen while the new classification head learns. Second, the final 30 base layers are unfrozen and fine-tuned with a small learning rate. Batch-normalisation layers remain frozen for stable transfer learning.

## 5. Evaluation

The evaluation used 383 unseen test images. The measured results were **83.81% accuracy**, **85.52% precision**, **80.16% recall**, **81.56% macro F1**, and **83.75% weighted F1**. The script also saves per-class results and a confusion matrix in `artifacts/`.

## 6. Streamlit application

The user uploads a JPG, JPEG, PNG or WebP image and presses **Classify Waste**. The app converts the image to RGB, resizes it to 224 × 224, calls the saved Keras model and displays:

- Predicted category
- Confidence percentage
- Low-confidence warning
- Recycling recommendation
- Top three predictions as a chart and table

## 7. Advantages and limitations

Advantages include transfer learning, a lightweight model, reproducible preprocessing, a simple web interface and clear confidence output. Limitations include the small and imbalanced dataset, plain backgrounds in TrashNet, one-label classification, and different local recycling rules.

## 8. Future improvements

- Add more real-world and locally collected images.
- Balance the six classes and test class weighting.
- Add object detection for multiple waste items in one image.
- Add Malaysian recycling guidance and multilingual output.
- Deploy the application online or integrate it into a mobile application.

## Sources

- TrashNet dataset: https://github.com/garythung/trashnet
- TensorFlow MobileNetV2: https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2
- TensorFlow transfer learning tutorial: https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub
- Streamlit file uploader: https://docs.streamlit.io/develop/api-reference/widgets/st.file_uploader
