# Simple Presentation Script — 3 People

Replace the names and practise slowly. The full presentation is designed for approximately 8–10 minutes, including the live demo.

## Person 1 — Problem and dataset

### Slide 1 — Introduction

Hello everyone. Our project is called WasteWise AI. It is a waste image classification application. It uses deep learning to help users identify the material of a waste item.

### Slide 2 — Problem and objectives

Many people are not sure which recycling bin to use. Wrong waste sorting can contaminate recyclable materials. Our objectives are to classify the image, show the confidence score, and give a simple recycling recommendation.

### Slide 3 — Dataset

We used the public TrashNet dataset. It contains 2,527 images and six categories: cardboard, glass, metal, paper, plastic, and trash. Paper is the largest class, and trash is the smallest class.

### Slide 4 — Preprocessing

We checked the images for corrupted and duplicate files, and removed three exact duplicates. We resized every image to 224 by 224 pixels. We divided the cleaned data into 70 percent training, 15 percent validation, and 15 percent testing. We also used image augmentation during training.

Transition: Now, Person 2 will explain our model and results.

## Person 2 — Model and results

### Slide 5 — MobileNetV2

We used MobileNetV2 with ImageNet pre-trained weights. MobileNetV2 is efficient and has fewer parameters than many larger models. We removed its original output layer and added a new six-category classifier.

### Slide 6 — Transfer learning

First, we froze the pre-trained base and trained the new classification layers. Next, we unfroze the final 30 layers and fine-tuned them with a small learning rate. This helps the model learn our waste categories without training from zero.

### Slide 7 — Results

We evaluated the model on 383 unseen images. The test accuracy was 83.81 percent, precision was 85.52 percent, recall was 80.16 percent, and macro F1 was 81.56 percent. The confusion matrix helps us see which waste categories are sometimes confused.

Transition: Now, Person 3 will show the Streamlit application.

## Person 3 — Application, demo and conclusion

### Slide 8 — Streamlit application

Our application is built with Streamlit. The user uploads one image and presses Classify Waste. The application shows the predicted category, confidence score, recycling recommendation, and top three predictions.

### Live demonstration

First, I upload a waste image. Then I press Classify Waste. The model processes the image and displays the result. Here, the prediction is shown with its confidence. The user can also read the recommended recycling action.

### Slide 9 — Limitations

Our dataset is small and many images have a plain background. Real-life images with several objects or complex backgrounds may be more difficult. Recycling rules can also be different in each location.

### Slide 10 — Conclusion

In conclusion, WasteWise AI combines MobileNetV2, transfer learning and Streamlit in one working computer vision application. In the future, we can add more real-world images, multiple-object detection and local recycling guidance. Thank you. We are ready for your questions.

## Quick Q&A

**Why did you choose MobileNetV2?**  
It is efficient, lightweight and suitable for transfer learning and future mobile deployment.

**Did you train the model from scratch?**  
No. We used ImageNet pre-trained weights, trained a new classification head, and fine-tuned only the final layers.

**Why resize to 224 × 224?**  
It is the standard input size used in our MobileNetV2 pipeline and gives consistent model input.

**What happens when confidence is low?**  
The app shows a warning and asks the user to try a clearer image.

**What is the main limitation?**  
TrashNet is small and mostly uses plain backgrounds, so more real-world data would improve generalisation.
