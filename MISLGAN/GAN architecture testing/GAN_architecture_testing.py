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
from keras.backend.tensorflow_backend import set_session


from sklearn.metrics import confusion_matrix, classification_report

# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#DYNAMICALLY GROW THE GPU MEMORY
print(tf.test.is_gpu_available(cuda_only=True))
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True      
sess =  tf.compat.v1.Session(config=config)
set_session(sess) 

batch_size = 30
steps_per_epoch = 70
epochs = 120 # 30
IMG_SIZE = 256
DATADIR = "/home/mtech/2020/satyaki_ghosh/codes/Rajesh/MISLGAN/GAN_architecture_testing"


# LOAD SEED DATA
seed = []
with open('seed_patches.npy','rb') as f:
    xx = np.load(f, allow_pickle=True)
seed = list(xx)
random.shuffle(seed)
seed = np.array(seed)
seed = seed.reshape((-1,IMG_SIZE, IMG_SIZE, 3))
seed = seed/255.0
print(f"Seed data loaded. Shape : {seed.shape}")

# LOAD TRAINING DATA
x_train = []
with open('training_patches.npy','rb') as f:
    xx = np.load(f, allow_pickle=True)
x_train = list(xx)

x_train = np.array(x_train)
x_train = x_train/255.0
x_train = x_train.reshape((-1, IMG_SIZE, IMG_SIZE, 3))
if x_train.shape[0] > 0:
    print(f"Training data loaded. Shape: {x_train.shape}")
else:
    print("Training data too big to load.")


if not os.path.isdir("save_path"):
    os.mkdir("save_path")


# LOAD GAN TESTING DATA
testing_data_GAN = []
with open('testing_patches.npy','rb') as f:
    xx = np.load(f, allow_pickle=True)
testing_data_GAN = list(xx)

# MAKE TEST SET
X_test_GAN_image = []
X_test_GAN_bayer = []
Y_test_GAN = []

for features_image,features_bayer,label in testing_data_GAN :
  X_test_GAN_image.append(features_image)
  X_test_GAN_bayer.append(features_bayer)
  Y_test_GAN.append(label)


X_test_GAN_bayer = np.array(X_test_GAN_bayer).reshape(-1,IMG_SIZE,IMG_SIZE,1)
X_test_GAN_bayer = X_test_GAN_bayer/255.0
X_test_GAN_image = np.array(X_test_GAN_image).reshape(-1,IMG_SIZE,IMG_SIZE,3)
X_test_GAN_image = X_test_GAN_image/255.0
Y_test_GAN = np.array(Y_test_GAN)
print("Testing data loaded.")
print(f"X_test_GAN_bayer : {X_test_GAN_bayer.shape}")
print(f"X_test_GAN_image : {X_test_GAN_image.shape}")
print(f"Y_test_GAN : {Y_test_GAN.shape}")


# EVALUTATE GAN
def GAN_performance(epoch, generator,seed_test_bayer,seed_test_image, g_loss):
	# prepare fake examples
    y_pred_GAN = generator.predict(seed_test_bayer)
    #seed_test_image = X_test_GAN_image[50,100,150,200,250,350,400,450,500,550,650,700,750,800,850,950,1000,1050,1100,1150,1250,1300,1350,1400,1450]
    for j in range(5 * 5):
        plt.subplot(5, 5, j+1)
        plt.axis('off')
        plt.imshow(np.squeeze(seed_test_image[j]))
    plt.savefig('./seed_plots/seed_plot_%03d.png' % (epoch+1))
    plt.close()
   
	# plot images
    for i in range(5 * 5):
        plt.subplot(5, 5, i+1)
        plt.axis('off')
        plt.imshow(y_pred_GAN[i])
	# save plot to file
    plt.savefig('./save_path/generated_plot_%03d.png' % (epoch+1))
    plt.close()
	# save the generator model
    generator.save('./save_path/generator_%03d_%03d.h5' % (epoch+1, g_loss))


# CREATE GENERATOR
optimizer = Adam(0.0001, 0.5)
def create_generator(shape = (IMG_SIZE,IMG_SIZE,3)):
    input = Input(shape = (IMG_SIZE,IMG_SIZE,3))    
    
    layer_1 = Conv2D(64, (3, 3), strides = 1, padding = 'same')(input)
    layer_2 = Activation('relu')(layer_1)
    layer_3 = Conv2D(64, (3, 3), strides = 1, padding = 'same')(layer_2)
    layer_4 = Activation('relu')(layer_3)
    layer_5 = Conv2D(64,(1, 1), strides = 1, padding = 'same')(layer_4)
    layer_6 = Activation('relu')(layer_5)
        
    
    layer_7 = Conv2D(128, (3, 3), strides = 1, padding = 'same')(layer_6)
    layer_8 = Activation('relu')(layer_7)
    layer_9 = Conv2D(128, (3, 3), strides = 1, padding = 'same')(layer_8)
    layer_10 = Activation('relu')(layer_9)
    layer_11 = Conv2D(128, (1, 1), strides = 1, padding = 'same')(layer_10)
    layer_12 = Activation('relu')(layer_11)
    
    layer_13 = Conv2D(3, (3, 3), strides = 1, padding = 'same')(layer_12)
    layer_14 = Activation('relu')(layer_13)
    
    model = Model(inputs = input, outputs = layer_14)
    model.compile(loss = "binary_crossentropy", optimizer=optimizer, run_eagerly = True)
    
    return model


# CREATE DISCRIMINATOR
def normalise(w):
    j=int(w.shape[0]/2)
    for i in range(w.shape[-1]):
        w[j,j,:,i]=0
        for k in range(w.shape[-2]):
          wsum = w[:,:,k,i].sum()
          w[:,:,k,i]/=wsum
        w[j,j,:,i]=-1
    return w

num_filter=3
w = np.random.rand(5,5,3,num_filter)             #changing the number of filters change accordingly the last number 
wgt = normalise(w)
bias = np.zeros(num_filter)

opt = tf.keras.optimizers.SGD(learning_rate = 0.001, momentum = 0.9, decay = 0.0005)
# define the standalone discriminator model
def create_discriminator():
    model = Sequential()    

    #constrained_layer
    model.add(Conv2D(3, (5, 5), input_shape=(IMG_SIZE,IMG_SIZE,3)))       #convRes
    model.layers[0].set_weights([wgt, bias])

    model.add(Conv2D(96,(7,7),strides = (2,2), padding = 'same', input_shape=(IMG_SIZE,IMG_SIZE,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (3,3), strides = 2 ))

    model.add(Conv2D(64,(5,5),strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))


    model.add(Conv2D(64,(5,5),strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))


    model.add(Conv2D(128,(1,1),strides = (1,1), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size = (3,3), strides = 2))

    model.add(Flatten())

    model.add(Dense(200))
    model.add(Activation('relu'))

    model.add(Dense(200))
    model.add(Activation('relu'))

    model.add(Dense(1, activation='sigmoid'))
    
	# compile model
    model.compile(loss = "binary_crossentropy", optimizer = opt , metrics = ['accuracy'])

    return model


# TRAIN THE GAN
discriminator = create_discriminator()
generator = create_generator()
discriminator.trainable = False
gan_input = Input(shape=(IMG_SIZE,IMG_SIZE,3))
fake_image = generator(gan_input)
gan_output = discriminator(fake_image)

gan = Model(inputs = [gan_input],outputs = [gan_output])
gan.compile(loss='binary_crossentropy', optimizer=optimizer)
print("Generator and Discriminator initialized. Starting training...")
   
d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
set = np.random.randint(0, X_test_GAN_image.shape[0], size=25)


import gc
gc.collect()

for epoch in range(epochs):
    for batch in range(steps_per_epoch) : 
        # print("Starting GAN training")
        seed_epoch = seed[np.random.randint(0, seed.shape[0],size = batch_size)]#images of different classes
        seed_epoch = np.asarray(seed_epoch).astype('float32')
        fake_x = generator.predict(seed_epoch, verbose=10) #output of generator

        # print("Starting training of discriminator")
        real_x = x_train[np.random.randint(0, x_train.shape[0],size = batch_size)] #images of target class        
        d_train_y = np.zeros(2*batch_size)
        d_train_y[:batch_size] = 0.9
        d_train_x = np.concatenate((real_x,fake_x))
        d_train_x = np.asarray(d_train_x).astype('float32')
        d_loss,d_acc = discriminator.train_on_batch(d_train_x, d_train_y) #discriminator training 
        # print("Discriminator trained")
        '''
        total_w = discriminator.layers[0].get_weights()
        w=np.array(total_w[0])
        bias = np.zeros(3)
        w = normalise(w)
        discriminator.layers[0].set_weights([w,bias])
        
        '''
        
        #generator training with different classes and keeping output to real
        y_gen = np.ones(batch_size)
        g_loss = gan.train_on_batch([seed_epoch],[y_gen]) 
        
    
        d1_hist.append(d_loss)
        d2_hist.append(d_acc)
        g_hist.append(g_loss)
        print(f'Epoch: {epoch} \t d_loss: {d_loss}  d_acc: {d_acc} g_loss: {g_loss}')

    # plot generator and discriminator loss
    plt.subplot(2, 1, 1)
    plt.plot(d1_hist, label='d-loss')
    plt.plot(g_hist, label='g_loss')
    plt.legend()
    plt.savefig('Loss')
    plt.close()

    # plot discriminator accuracy
    plt.subplot(2, 1, 2)
    plt.plot(d2_hist, label='d-acc')
    plt.legend()
    plt.savefig('Accuracy')
    plt.close()        
        
    seed_test_image = X_test_GAN_image[set]
    seed_test_bayer = X_test_GAN_image[set]
    #seed_test = X_test_GAN_bayer[50,100,150,200,250,350,400,450,500,550,650,700,750,800,850,950,1000,1050,1100,1150,1250,1300,1350,1400,1450]
    GAN_performance(epoch, generator,seed_test_bayer,seed_test_image, g_loss)
print("GAN training complete.")

# plot generator and discriminator loss
plt.subplot(2, 1, 1)
plt.plot(d1_hist, label='d-loss')
plt.plot(g_hist, label='g_loss')
plt.legend()
plt.savefig('Loss')
plt.close()


# plot discriminator accuracy
plt.subplot(2, 1, 2)
plt.plot(d2_hist, label='d-acc')
plt.legend()
plt.savefig('Accuracy')
plt.close()


## TODOS:
# How to decrease generator loss?
# print discriminator outputs to check if it is randomly guessing
# keep train and test data separate
# write the attack code
# use rajesh's hyperparameters
# DATASETS? seed: 700 image of each class with 12 patches per image
# train/real: only Kodak images 2000
# BUT discriminator never sees any real image other than Kodak
# ensure to use whole data in every epoch instead of resampling due to randint
# is real and fake dataset distrib correct?
