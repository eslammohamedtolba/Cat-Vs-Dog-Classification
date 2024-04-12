! wget 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_5340.zip'
! unzip kagglecatsanddogs_5340.zip

import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
import pickle as pk



directory = r'PetImages'
categories = ['Cat','Dog']
# Load all the dataset images in a list
data = []
img_size = 100
for category in categories:
    folder = os.path.join(directory,category)
    label = categories.index(category)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        image = cv2.imread(img_path)
        # Check if the image was successfully loaded
        if image is not None:
            image = cv2.resize(image, (img_size, img_size))
            data.append([image,label])
        else:
            print(f"Error loading image: {img_path}")

# Shuffle the data before split it
random.shuffle(data)



# Split the data into input and label data
X = []
Y = []
for feature,label in data:
    X.append(feature)
    Y.append(label)
X = np.array(X)
X = X/255
Y = np.array(Y)
# Print the shape of the input data and the first value 
print(X.shape)
print(X[0])



# Define the input shape of the images
input_shape = (100, 100, 3)
# Define the number of classes
num_classes = 2
# Define the model architecture
Model = keras.Sequential([
        keras.layers.Input(shape=input_shape),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),
        keras.layers.Flatten(),
        keras.layers.Dense(128, input_shape=X.shape[1:],activation="relu"),
        keras.layers.Dense(num_classes, activation="softmax")
])
# Compile the model
Model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
# Train the model
result = Model.fit(X,Y, epochs=6,validation_split=0.1)


# Visualize the accuracy with validation accuracy
plt.figure(figsize=(5,5))
plt.plot(result.history['accuracy'],color='red')
plt.plot(result.history['val_accuracy'],color='blue')
plt.title('model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.show()
# Visualize the loss with validation loss
plt.figure(figsize=(5,5))
plt.plot(result.history['loss'],color='red')
plt.plot(result.history['val_loss'],color='blue')
plt.title('model loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()

# Avaluate the model
evaluation = Model.evaluate(X,Y)
print("the loss value is: ",evaluation[0])
print("the accracy value is: ",evaluation[1])




# Make a predictive system

image_path = input("Enter image path to be predicted: ") # Load the image
input_image = cv2.imread(image_path)

# Ensure the image has 3 color channels (e.g., convert from grayscale to RGB)
if input_image.shape[-1] == 1:
    input_image = cv2.cvtColor(input_image, cv2.COLOR_GRAY2BGR)
# Resize the image into 100x100
img_size = 100
resized_image = cv2.resize(input_image, (img_size, img_size))
# Normalize the image by scaling it
resized_image = resized_image / 255

# Make the model predict what is in the image if it's dog will print 1 otherwise will print 0
prediction = Model.predict(np.expand_dims(resized_image, axis=0))
predicted_class = np.argmax(prediction)

print("Predicted class:", predicted_class)


# Save the model
pk.dump(Model,open('model.pkl','wb'))


