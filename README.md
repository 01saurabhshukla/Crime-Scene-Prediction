# Crime Scene Detection using LRCN Network

![crime_scene_detection](crime_scene_detection.jpg)

## Overview

Crime Scene Detection is a deep learning project that aims to automatically detect crime scenes in images and videos using an LRCN (Long-term Recurrent Convolutional Networks) network. The LRCN model combines the power of Convolutional Neural Networks (CNNs) for image feature extraction with Long Short-Term Memory (LSTM) networks for temporal information processing, making it effective for sequential data like videos.

This repository contains the necessary code and data to train and evaluate the Crime Scene Detection model. The trained model achieves good accuracy in detecting crime scenes in both images and videos.

## Requirements

- Python (>= 3.6)
- TensorFlow (>= 2.x)
- Keras (>= 2.x)
- NumPy (>= 1.18)

You can install the required packages using `pip`:

```bash
pip install tensorflow keras numpy
```

## Dataset

To train the Crime Scene Detection model, a labeled dataset of crime scene images and videos is required. The dataset will be organized into two main folders:

It's essential to ensure that the dataset is balanced and representative of real-world scenarios. Additionally, a portion of the dataset should be reserved for validation and testing.

## Model Architecture

The Crime Scene Detection model follows the LRCN (Long-term Recurrent Convolutional Networks) architecture, which is designed to process sequential data such as videos. The architecture consists of two main parts:

1. **Convolutional Neural Network (CNN)**: The CNN part is responsible for extracting spatial features from images or video frames. It can be based on well-known CNN architectures like VGG, ResNet, or MobileNet, or a custom-designed CNN.

2. **Long Short-Term Memory (LSTM)**: The LSTM part processes the temporal information from the extracted features obtained from the CNN. It learns the dependencies between frames in videos, which helps in better understanding the context and sequential nature of the data.

## Training

Follow these steps to train the Crime Scene Detection model:

1. Prepare the dataset: Organize the dataset , and split it into training, validation, and test sets.

2. Data Preprocessing: Implement data preprocessing code to load and augment images and video frames. Perform necessary transformations and resizing to ensure compatibility with the LRCN model.

3. Define the LRCN Model: Create the LRCN model architecture using TensorFlow/Keras. Combine the CNN and LSTM components and specify appropriate loss function and optimizer.

4. Train the Model: Use the training set to train the LRCN model. Monitor the validation performance to prevent overfitting.

5. Evaluate the Model: Once training is complete, evaluate the model's performance on the test set to measure its accuracy.

## Evaluation

To evaluate the trained model, the following metrics can be used:

- **Accuracy**: The proportion of correctly predicted crime scenes over the total number of samples in the test set.
- **Precision**: The ratio of true positive predictions to the total positive predictions made by the model.
- **Recall**: The ratio of true positive predictions to the total actual positive samples in the test set.
- **F1 Score**: The harmonic mean of precision and recall, providing a balanced measure between the two metrics.

## Results

The Crime Scene Detection model trained using the LRCN architecture achieves good accuracy on the test set. The specific evaluation metrics and their values are included in the project report or evaluation section.

## Usage

1. Clone the repository:

```bash
git clone (https://github.com/01saurabhshukla/Crime-Scene-Prediction)
cd Crime-Scene-Detection
```
2. The example dataset used in this project is sourced from a publicly available 50 action dataset [[source link](https://www.crcv.ucf.edu/data/UCF50.rar)]

3. Prepare your dataset 

4. Implement data preprocessing code and ensure it loads data in the required format.

5. Create and train the LRCN model using the provided code or your custom implementation.

6. Evaluate the model on the test set and analyze the results.



## Contact

For any questions or inquiries, please contact [103saurabhshukla@gmail.com].
