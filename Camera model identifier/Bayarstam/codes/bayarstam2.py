# -*- coding: utf-8 -*-
"""BayarStam2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/16BeazpfeIbB10xqYJHt59xCQ86pvpgrT
"""
import tensorflow as tf
import numpy as np
np.random.seed(123)
from tensorflow import keras
import os
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from time import time
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, BatchNormalization, AveragePooling2D
from keras.utils import np_utils
from keras import optimizers
from keras import initializers
from keras.constraints import Constraint
from tensorflow.keras.callbacks import Callback
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import model_from_json
from keras.models import load_model
from keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.callbacks import TensorBoard
from keras.initializers import TruncatedNormal
keras.backend.set_image_data_format('channels_last')
from sklearn.metrics import classification_report, confusion_matrix


#DYNAMICALLY GROW THE GPU MEMORY
def start_GPU_session():
    import tensorflow as tf
    gpu_options = tf.compat.v1.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=0.4)
    config = tf.compat.v1.ConfigProto(device_count={'GPU':1,'CPU':1},
            gpu_options=gpu_options,
            intra_op_parallelism_threads=1,
            inter_op_parallelism_threads=1)
    sess=tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(sess)

def reset_TF_session():
    import tensorflow as tf
    tf.compat.v1.keras.backend.clear_session()


#Parameters
classes = 2
n_epochs = 1
batch = 32
image_row = 128
image_col = 128
#To track the loss history
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
#Function to normalise and prepare univ layer kernel
def normalise(w):
    j=int(w.shape[0]/2)
    for i in range(w.shape[-1]):
        w[j,j,:,i]=0
        wsum=w[:,:,:,i].sum()
        w[:,:,:,i]/=wsum
        w[j,j,:,i]=-1
    return w
class Normalise(Callback):
    def on_batch_end(self,batch,logs=None):
        total_w = self.model.layers[0].get_weights()
        w=np.array(total_w[0])
        bias = np.array(total_w[1])
        w = normalise(w)
        self.model.layers[0].set_weights([w,bias])

#Loading Data
train_datagen = ImageDataGenerator(data_format = 'channels_last', rescale=1.0/255.0) #pixel values are normalized
Val_datagen =  ImageDataGenerator(data_format = 'channels_last', rescale=1.0/255.0)  #normalized
#Val_datagen = ImageDataGenerator(data_format ='channels_first')                                        
test_datagen = ImageDataGenerator(data_format = 'channels_last', rescale= 1.0/255.0) #normalized
#test_datagen = ImageDataGenerator(data_format = 'channels_first',rescale=1./255)

train_generator = train_datagen.flow_from_directory(
		'/media/rijuban/0CF40770F4075B7A/RAISE/patches_sample/Train_data/',
		target_size=(image_row, image_col),
		color_mode = "grayscale",
		batch_size=batch,
		class_mode='binary',
                shuffle= True, seed=42)

valid_generator = Val_datagen.flow_from_directory(
		'/media/rijuban/0CF40770F4075B7A/RAISE/patches_sample/Val_data/',
		target_size=(image_row, image_col),
		color_mode="grayscale",
		batch_size=batch,
		class_mode='binary', shuffle=True, seed=42)
test_generator = test_datagen.flow_from_directory(
		'/media/rijuban/0CF40770F4075B7A/RAISE/patches_sample/Test_data/',
		target_size=(image_row, image_col),
		color_mode="grayscale",
		batch_size=1,
		class_mode=None, shuffle=True, seed=42)

num_filter=12
w = np.random.rand(5,5,1,num_filter)             #changing the number of filters change accordingly the last number 
wgt = normalise(w)
bias = np.zeros(num_filter)

#Architecture 
model = Sequential()
model.add(Convolution2D(12, (5, 5), input_shape=(image_row,image_col,1)))       #convRes
model.layers[0].set_weights([wgt, bias])
model.add(Convolution2D(64, (7, 7), strides=(2,2)))            #conv1
model.add(BatchNormalization(epsilon=1e-04, momentum=0.9, weights=None))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))                        #Max Pooling
model.add(Convolution2D(64, (3, 3), strides = (1,1)))        #conv2
model.add(BatchNormalization(epsilon=1e-04, momentum=0.9, weights=None))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Convolution2D(48, (1, 1),  strides = (1,1)))        #conv2
model.add(BatchNormalization(epsilon=1e-04, momentum=0.9, weights=None))
model.add(Activation('tanh'))
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
model.add(Flatten())#FC1
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(#Testing
STEP_SIZE_TEST = test_generator.n/test_generator.batch_size
test_generator.reset()
pred=model.predict(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
print('Confusion_Matrix')
print(confusion_matrix(test_generator.classes,predicted_class_indices))
Dense(classes, activation='softmax'))
model.summary()

#optimizer
opt = tf.keras.optimizers.Adam(learning_rate = 0.0001)
#Model
model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),optimizer=opt,metrics=['accuracy'])

#callbacks

filepath="weights-improvement-bayandstam{epoch:02d}-{val_accuracy:.2f}.hdf5"
#for each experiment create a separate folder for checkpoints
if not os.path.exists('/media/rijuban/0CF40770F4075B7A/RAISE/patches_sample/Checkpoints/'):                                       #change the name of the main directory i.e. Data_Cr
    os.makedirs('/media/rijuban/0CF40770F4075B7A/RAISE/patches_sample/Checkpoints/')
checkpoint = ModelCheckpoint('/media/rijuban/0CF40770F4075B7A/RAISE/patches_sample/Checkpoints/'+filepath , save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    verbose=1,
    save_best_only=True)
norm_callback = Normalise()
#EarlyStopping(monitor='val_acc', patience=30, mode='max', min_delta=0.0001)
history = LossHistory()
tensorboard = TensorBoard(log_dir='logs/{}'.format(time()), histogram_freq=0,write_graph=True, write_images=False)
csv_logger = CSVLogger('/media/rijuban/0CF40770F4075B7A/RAISE/patches_sample/Checkpoints/constrained.log')                                  #change the lof file name
#model.save('/media/rijuban/0CF40770F4075B7A/RAISE/patches_sample/constrained_Y_18oct.h5')

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=valid_generator.n//valid_generator.batch_size
STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

model.fit_generator(
		train_generator,
		callbacks = [norm_callback,checkpoint,history,tensorboard,csv_logger],
		steps_per_epoch=STEP_SIZE_TRAIN,
		epochs = n_epochs,
		validation_data = valid_generator,
		validation_steps = STEP_SIZE_VALID,
		verbose = 1)

#Testing
STEP_SIZE_TEST = test_generator.n/test_generator.batch_size
test_generator.reset()
pred=model.predict(test_generator,
steps=STEP_SIZE_TEST,
verbose=1)
predicted_class_indices=np.argmax(pred,axis=1)
print('Confusion_Matrix')
print(confusion_matrix(test_generator.classes,predicted_class_indices))

