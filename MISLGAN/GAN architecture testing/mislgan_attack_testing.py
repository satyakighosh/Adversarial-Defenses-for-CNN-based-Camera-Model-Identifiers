
import numpy as np
from multiprocessing import dummy
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from sklearn.feature_extraction import image

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization

import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import confusion_matrix, classification_report

# TEST THE MISLGAN ATTACKS

# LOAD THE CLASSIFIER
def create_model():
    model = Sequential()

    #constrained_layer
    model.add(Conv2D(3, (5, 5), input_shape=(IMG_SIZE,IMG_SIZE,3)))       #convRes
    model.layers[0].set_weights([wgt, bias])

    # layer 2
    model.add(Conv2D(96,(7,7),strides = (2,2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (3,3), strides = 2 ))

    # layer 3
    model.add(Conv2D(64,(5,5),strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))

    # layer 4
    model.add(Conv2D(64,(5,5),strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))

    # layer 5
    model.add(Conv2D(128,(1,1),strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size = (3,3), strides = 2))

    model.add(Flatten())

    model.add(Dense(200))
    model.add(Activation('relu'))

    model.add(Dense(200))
    model.add(Activation('relu'))

    model.add(Dense(5, activation='softmax'))

    opt = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9, decay = 0.0005)

    model.compile(loss = "sparse_categorical_crossentropy", optimizer = opt, metrics = ['accuracy'])
    return model

checkpoint_dir = "E:\IIT GUWAHATI\MTP\MTP_CODES\Rajesh\Camera-model identifier\training_1"
#classifier = tf.keras.models.load_model('colourchannel_patches10_classes5.h5')    
latest = tf.train.latest_checkpoint(checkpoint_dir)
# Create a new model instance
classifier = create_model()
# Load the previously saved weights
classifier.load_weights(latest)


# LOAD THE GENERATOR
generator = tf.keras.models.load_model('E:\IIT GUWAHATI\MTP\camera-model-classfier\save_path\generator_030.h5')
y_pred_GAN = generator.predict(X_test_GAN_image)
y_pred_classes = [np.argmax(element) for element in classifier.predict(y_pred_GAN)]


# Re-evaluate the model
loss, acc = classifier.evaluate(X_test, Y_test, verbose=2)
print("Accuracy: {:5.2f}%".format(100 * acc))
Y_pred = classifier.predict(X_test)
Y_pred = np.argmax(Y_pred, axis=1)

# Checking label distributions
print(f'Y_train label distrib: {np.unique(Y, return_counts=True)}')
print(f'Y_test label distrib: {np.unique(Y_test, return_counts=True)}')
print(f'Y_pred label distrib: {np.unique(Y_pred, return_counts=True)}')


# Confusion Matrix
print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")
print(f"Confusion matrix: \n{confusion_matrix(Y_test, Y_pred)}")
print(f"Classification Report:\n {classification_report(Y_test, Y_pred)}")