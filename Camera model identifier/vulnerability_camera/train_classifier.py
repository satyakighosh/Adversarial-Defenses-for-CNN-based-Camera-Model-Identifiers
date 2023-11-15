
import numpy as np
import os
import random
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.callbacks import Callback,ModelCheckpoint,LearningRateScheduler
import pickle
import math

# https://stackoverflow.com/questions/41233635/meaning-of-inter-op-parallelism-threads-and-intra-op-parallelism-threads
# https://www.tensorflow.org/guide/migrate/multi_worker_cpu_gpu_training
# cluster_resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
# worker_config = tf.compat.v1.ConfigProto()
# worker_config.inter_op_parallelism_threads = 4
# for i in range(3):
#   tf.distribute.Server(
#       cluster_resolver.cluster_spec(),
#       job_name="worker",
#       task_index=i,
#       config=worker_config)

# Include the epoch in the file name (uses `str.format`)
# checkpoint_path = "E:\IIT GUWAHATI\MTP\MTP_CODES\Rajesh\Camera-model identifier\\training_1\cp-{epoch:04d_}.ckpt"
# print(checkpoint_path)
# checkpoint_dir = os.path.dirname(checkpoint_path)
# print(checkpoint_dir)
# os.listdir(checkpoint_dir)


DATADIR = "E:\IIT GUWAHATI\MTP\Datasets\Dresden"
CATEGORIES = [
    "D01_Samsung_GalaxyS3Mini",
    "D02_Apple_iPhone4s",
    "D03_Huawei_P9",
    "D04_LG_D290",
    "D05_Apple_iPhone5c",
    "D06_Apple_iPhone6"]

IMG_SIZE = 256

# LOADING THE DATSET
with open('training_data2.npy','rb') as f:
    dummy = np.load(f, allow_pickle=True)
dataset = list(dummy)

# TRAIN-TEST SPLIT AND SHUFFLE THE PATCHES : (train,val,test) = (0.8,0.1,0.1)
random.shuffle(dataset)
training_data = dataset[0:8325]
testing_data = dataset[8325:]


# CREATING INPUT AND LABELS
X = []
Y = []
for features,label in training_data :
  X.append(features)
  Y.append(label)


# PREPROCESSING
X = np.array(X).reshape(-1,IMG_SIZE,IMG_SIZE,3)
X = X/255.0
Y = np.array(Y)


# CONSTRAINED LAYER: NORMALIZE THE WEIGHTS AND SET MIDDLE VALUE TO -1
def normalise(w): # w has a shape [X, Y, num_layers_per_filter, num_filter]
    j=int(w.shape[0]/2)
    for i in range(w.shape[-1]): 
        w[j,j,:,i]=0
        for k in range(w.shape[-2]):
          wsum = w[:,:,k,i].sum()
          w[:,:,k,i]/=wsum
        w[j,j,:,i]=-1
    return w

class Normalise(Callback):
    def on_batch_end(self,batch,logs=None):
        total_w = self.model.layers[0].get_weights()
        w=np.array(total_w[0])
        bias = np.array(total_w[1])
        w = normalise(w)
        self.model.layers[0].set_weights([w,bias])


# TO DECAY THE LEARNING RATE WITH INCREASING EPOCHS
def step_decay(epoch):
  initial_lrate = 0.001
  drop = 0.5
  epochs_drop = 5.0
  lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
  return lrate

lrate = LearningRateScheduler(step_decay)

# INITIALIZING FILTERS TO APPLY ON PATCHES
num_filter=3
w = np.random.rand(5,5,3,num_filter)             #changing the number of filters change accordingly the last number 
wgt = normalise(w)
bias = np.zeros(num_filter)


# TRAINING
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

# RAJESH'S CODE (only next line):
#history = model.fit(X,Y,batch_size = 64, epochs = 70,callbacks = [norm_callback], validation_split = 0.1, verbose = 1)

batch_size = 64
epochs = 70

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training_1/cp_{epoch:04d}__{accuracy:.4f}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
checkpoint_epoch_freq = 1

# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True,
    save_freq='epoch' # checkpoint_epoch_freq*batch_size
  )

# Create a new model instance
model = create_model()
model.summary()
norm_callback = Normalise()

# Save the weights using the `checkpoint_path` format
# model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the TWO callbacks
history=model.fit(X,Y,epochs=epochs,batch_size=batch_size,
          callbacks=[norm_callback,cp_callback,lrate],
          validation_split = 0.1,verbose=1)
# print(history.history)
with open('train_history_dict','wb') as f:
    pickle.dump(history.history,f)
np.save('train_history_dict.npy',history.history)

# LOAD THE HISTORY AS A DICT
# history=np.load('my_history.npy',allow_pickle='TRUE').item()





# TESTING
# CREATING INPUT AND LABELS
X_test = []
Y_test = []
for features,label in testing_data :
  X_test.append(features)
  Y_test.append(label)

# PREPROCESSING
X_test = np.array(X_test).reshape(-1,IMG_SIZE,IMG_SIZE,3)
X_test = X_test/255.0
Y_test = np.array(Y_test)
print(f'Test Data shape: {np.shape(X_test)}')

# print(f'Checkpoint models: {os.listdir(checkpoint_dir)}')
latest = tf.train.latest_checkpoint(checkpoint_dir)
print(f'Latest model : {latest}')

# Create a new model instance
model = create_model()

# Load the previously saved weights
model.load_weights(latest)

# Re-evaluate the model
loss, acc = model.evaluate(X_test, Y_test, verbose=2)
print("Accuracy: {:5.2f}%".format(100 * acc))
Y_pred = model.predict(X_test)
Y_pred = np.argmax(Y_pred, axis=1)

# Checking label distributions
print(f'Y_train label distrib: {np.unique(Y, return_counts=True)}')
print(f'Y_test label distrib: {np.unique(Y_test, return_counts=True)}')
print(f'Y_pred label distrib: {np.unique(Y_pred, return_counts=True)}')


# Confusion Matrix
print(f"Accuracy: {accuracy_score(Y_test, Y_pred)}")
print(f"Confusion matrix: \n{confusion_matrix(Y_test, Y_pred)}")
print(f"Classification Report:\n {classification_report(Y_test, Y_pred)}")