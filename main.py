from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries for math and data
import numpy as np
import matplotlib.pyplot as plt

# The version of tensorflow that it's using
print(tf.__version__)

# Download the fashion dataset and load it
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# The types of clothing
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# The train shape of images and label values
train_images.shape
len(train_labels)
train_labels

# The test shape of images and labels
test_images.shape
len(test_labels)

# Image info like the number of pixels and their values
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# To scale the pixel values of the images as 0 or 1
train_images = train_images / 255.0
test_images = test_images / 255.0

# Display the first 25 images from training and show the class names
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# Flatten transforms images to 1d
# Dense layers are fully connected and produce scores on how likely the the current image is one of the 10 classes
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Loss = accuracy of answers
# Optimizer = how it improves 
# Metrics = accuracy of training
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Start training
model.fit(train_images, train_labels, epochs=5)

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_acc)
