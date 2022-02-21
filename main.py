import os
import cv2
import numpy as np
import nnfs
import pickle
import copy
from model import *

# Label index to label name relation
fashion_mnist_labels = {
    0: 'T-shirt/top',
    1: 'Pants',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

# Input image PreProcessing
image_data = cv2.imread('tshirt.png', cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28, 28))
image_data = 255 - image_data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5

model = Model.load('mymodel.model')

# Predict on the image
confidences = model.predict(image_data)

# Get prediction instead of confidence levels
predictions = model.output_layer_activation.predictions(confidences)

# Get label name from label index
prediction = fashion_mnist_labels[predictions[0]]

print(prediction)




# # Create dataset
# X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')

# # Shuffle the training dataset
# keys = np.array(range(X.shape[0]))
# np.random.shuffle(keys)
# X = X[keys]
# y = y[keys]

# # Scale and reshape samples
# X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
# X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) -
#              127.5) / 127.5

# # Instantiate the model
# model = Model()


# # Add layers
# model.add(Layer_Dense(X.shape[1], 128))
# model.add(Activation_ReLU())
# model.add(Layer_Dense(128, 128))
# model.add(Activation_ReLU())
# model.add(Layer_Dense(128, 10))
# model.add(Activation_Softmax())

# # Set loss, optimizer and accuracy objects
# model.set(
#     loss=Loss_CategoricalCrossentropy(),
#     optimizer=Optimizer_Adam(decay=1e-3),
#     accuracy=Accuracy_Categorical()
# )

# # Finalize the model
# model.finalize()

# # Train the model
# model.train(X, y, validation_data=(X_test, y_test),
#             epochs=10, batch_size=128, print_every=100)
# model.save('mymodel.model')