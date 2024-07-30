# Cat-Vs-Dog-Classification
This is a Cats vs Dogs classifier model that uses a convolutional neural network (CNN) to distinguish between images of cats and dogs. 

![image about the final project](<Cat vs Dog classification app.PNG>)

## Prerequisites
Before running the code, make sure you have the following dependencies installed:
- NumPy
- OpenCV (cv2)
- TensorFlow
- Keras
- Matplotlib

## Overview of the Code
1-Download the Cats vs Dogs dataset and unzip it. The dataset contains images of cats and dogs.

2-Load and preprocess the dataset:
-Load and resize the images.
-Shuffle the data.
-Split the data into input (X) and label (Y) data.
-Normalize the image pixel values to the range [0, 1].

3-Build a convolutional neural network (CNN) model for image classification.

4-Compile and train the model using the training data.

5-Save the trained model for future use.

6-Visualize the model's accuracy and loss during training.

7-Evaluate the model's performance on the entire dataset.

8-Create a predictive system that takes an image path as input, preprocesses the image, and uses the model to predict whether the image contains a cat(0) or a dog(1).

## Model Accuracy
The model achieves an accuracy of 96.5%.

## Contribution
Contributions to this project are welcome. You can help improve the model's accuracy, explore more advanced neural network architectures, or enhance the data preprocessing and visualization steps.
Feel free to make any contributions and submit pull requests.

