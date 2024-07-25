# Implement deep learning for recognizing classes for datasets like CIFAR-10 images for previously unseen images and assign them to one of the 10 classes.
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = datasets.cifar10.load_data()

# Display shapes of the datasets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Reshape labels to 1D arrays
y_train = y_train.reshape(-1,)
y_test = y_test.reshape(-1,)

# Define class names
classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Visualize CIFAR-10 images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(X_train[i], cmap=plt.cm.binary)
    plt.xlabel(classes[y_train[i]])
plt.show()

# Function to plot a single sample with true and predicted labels
def plot_sample(X, y_true, y_pred, classes, index):
    plt.figure(figsize=(2, 2))
    plt.imshow(X[index])
    plt.xlabel(f"True: {classes[y_true[index]]}\nPredicted: {classes[y_pred[index]]}")
    plt.show()

# Normalize pixel values to be between 0 and 1
X_train = X_train / 255.0
X_test = X_test / 255.0

# Build and train an Artificial Neural Network (ANN)
ann = models.Sequential([
    layers.Flatten(input_shape=(32, 32, 3)),
    layers.Dense(3000, activation='relu'),
    layers.Dense(1000, activation='relu'),
    layers.Dense(10, activation='softmax')
])

ann.compile(optimizer='SGD', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
ann.fit(X_train, y_train, epochs=5)

# Evaluate ANN on test data
ann_loss, ann_accuracy = ann.evaluate(X_test, y_test)
print("ANN Test Loss:", ann_loss)
print("ANN Test Accuracy:", ann_accuracy)

# Generate predictions for ANN
y_pred_ann = ann.predict(X_test)
y_pred_classes_ann = [np.argmax(element) for element in y_pred_ann]
print("ANN Classification Report:\n", classification_report(y_test, y_pred_classes_ann, target_names=classes))

# Build and train a Convolutional Neural Network (CNN)
cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(X_train, y_train, epochs=5)

# Evaluate CNN on test data
cnn_loss, cnn_accuracy = cnn.evaluate(X_test, y_test)
print("CNN Test Loss:", cnn_loss)
print("CNN Test Accuracy:", cnn_accuracy)

# Generate predictions for CNN
y_pred_cnn = cnn.predict(X_test)
y_pred_classes_cnn = [np.argmax(element) for element in y_pred_cnn]
print("CNN Classification Report:\n", classification_report(y_test, y_pred_classes_cnn, target_names=classes))

# Plot a sample from test set with true and predicted labels
plot_sample(X_test, y_test, y_pred_classes_cnn, classes, 4)
print("True Class:", classes[y_test[4]])
print("Predicted Class:", classes[y_pred_classes_cnn[4]])