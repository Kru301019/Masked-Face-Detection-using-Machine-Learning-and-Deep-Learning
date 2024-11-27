# Masked Face Detection using Machine Learning and Deep Learning

This repository demonstrates the implementation of a **Masked Face Detection** system. The goal of the project is to classify whether a person is wearing a mask or not in images using both traditional machine learning and deep learning approaches.

## Project Overview

This project focuses on detecting masked faces in images by utilizing:

1. **Random Forest Classifier**: A machine learning approach used to classify masked and non-masked faces based on extracted features.
2. **Convolutional Neural Network (CNN)**: A deep learning approach designed to automatically learn spatial features and patterns from the image data.

Challenges in this task include variations in mask types, lighting conditions, and occluding objects that may obstruct facial features.

## Approach

### 1. Data Loading and Preprocessing

The data loading and preprocessing stage is critical for preparing the image data for model training. The following steps are involved:

- **Loading Images**: The `load_dataset` function is used to load images and their corresponding labels (masked or not) from a ZIP file.
- **Resizing and Normalization**: All images are resized to 64x64 pixels to standardize input size. The pixel values are normalized to a range suitable for model input.
- **Visualization**: Class distribution and sample images are visualized to identify potential patterns and ensure the dataset is well-formed.

### 2. Model Training

Two models are implemented for classification:

#### 2.1 Random Forest Classifier

- **Feature Extraction**: Flattened features are extracted from images and fed into the Random Forest classifier.
- **Model Performance**: Precision, recall, and F1-score metrics are calculated to assess the performance of the classifier across different classes (masked and non-masked).

#### 2.2 Convolutional Neural Network (CNN)

- **CNN Architecture**: A Convolutional Neural Network (CNN) is designed using Keras' Sequential API, consisting of:
  - 2 convolution layers with max-pooling to extract spatial features from the images.
  - A flattening layer that converts 2D feature maps into a 1D vector.
  - Dense layers with ReLU activation functions for non-linearity.
  - Binary cross-entropy loss function for training, as it is a binary classification task (masked or not).
- **Data Augmentation**: To increase the diversity of the training dataset and reduce overfitting, data augmentation techniques such as rotation, zoom, and flip are applied using `ImageDataGenerator`.
- **Training**: The model is trained for 10 epochs, and training history is visualized to track progress.

### 3. Justification for Model Selection

- **Random Forest**: Random Forest is chosen for its simplicity, robustness, and ability to handle tabular data well. It's effective for problems with structured data and is interpretable, which is useful for model evaluation.
- **CNN**: CNNs are ideal for image classification tasks due to their ability to automatically learn spatial features and patterns. This makes CNNs especially powerful for tasks involving visual data like face detection.
- **Data Augmentation**: Augmentation helps to increase the variety of data available for training, which is crucial for improving the model's ability to generalize to new, unseen data.

## Results

- **Random Forest Classifier**: Achieved good performance in terms of accuracy, but may not perform as well as deep learning models on more complex image data.
- **CNN**: The CNN model outperformed the Random Forest classifier in terms of accuracy, as it is better suited for learning complex patterns in image data. It effectively captured spatial features, leading to improved performance in detecting masked faces.

## Future Work

- **Improvement in CNN Architecture**: Experiment with deeper architectures, including more convolution layers or advanced models like ResNet, to further improve classification accuracy.
- **Mask Type Detection**: Extend the project to not only detect the presence of a mask but also identify the type of mask (cloth, surgical, N95, etc.).
- **Integration of Attention Mechanisms**: Incorporate attention mechanisms to improve the model's focus on important regions of the face.

## Requirements

- Python 3.7 or higher
- Libraries:  
  - `pandas`  
  - `numpy`  
  - `scikit-learn`  
  - `matplotlib`  
  - `seaborn`  
  - `keras`  
  - `tensorflow`  
  - `opencv-python`


