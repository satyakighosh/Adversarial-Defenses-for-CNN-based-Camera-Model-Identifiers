
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

#DYNAMICALLY GROW THE GPU MEMORY
print(tf.test.is_gpu_available(cuda_only=True))
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True      
sess =  tf.compat.v1.Session(config=config)
set_session(sess) 


batch_size = 32   #30
steps_per_epoch = 125  #70 
epochs = 50  #120
IMG_SIZE = 256
channels = 1
DATADIR = "/home/mtech/2020/satyaki_ghosh/codes/Rajesh/MISLGAN/GAN_architecture_testing"



# LOAD SEED DATA
seed = []
with open('/home/mtech/2020/satyaki_ghosh/codes/Rajesh/MISLGAN/GAN_architecture_testing/seed_patches.npy','rb') as f:
    xx = np.load(f, allow_pickle=True)
seed = list(xx)
random.shuffle(seed)
seed = np.array(seed)
seed = seed.reshape((-1,IMG_SIZE, IMG_SIZE, 1))
seed = seed/255.0
print(f"Seed data loaded. Shape : {seed.shape}")


# LOAD TRAINING DATA
x_train = []
with open('/home/mtech/2020/satyaki_ghosh/codes/Rajesh/MISLGAN/GAN_architecture_testing/training_patches.npy','rb') as f:
    xx = np.load(f, allow_pickle=True)
x_train = list(xx)

x_train = np.array(x_train)
x_train = x_train/255.0
x_train = x_train.reshape((-1, IMG_SIZE, IMG_SIZE, 1))


if not os.path.isdir("save_path"):
    os.mkdir("save_path")
if not os.path.isdir("seed_plots"):
    os.mkdir("seed_plots")

# LOAD GAN TESTING DATA
testing_data_GAN = []
with open('/home/mtech/2020/satyaki_ghosh/codes/Rajesh/MISLGAN/GAN_architecture_testing/testing_patches.npy','rb') as f:
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


X_test_GAN_bayer = np.array(X_test_GAN_bayer).reshape(-1,IMG_SIZE,IMG_SIZE,channels)
X_test_GAN_bayer = X_test_GAN_bayer/255.0
X_test_GAN_image = np.array(X_test_GAN_image).reshape(-1,IMG_SIZE,IMG_SIZE,channels)
X_test_GAN_image = X_test_GAN_image/255.0
Y_test_GAN = np.array(Y_test_GAN)
print("Testing data loaded.")
print(f"X_test_GAN_bayer : {X_test_GAN_bayer.shape}")
print(f"X_test_GAN_image : {X_test_GAN_image.shape}")
print(f"Y_test_GAN : {Y_test_GAN.shape}")


# CUSTOM LOSS FUNCTION
alpha = 1
beta = 1
gamma = 1    

import keras.backend as K
from math import log2
# calculate cross entropy
def cross_entropy(p, q):
    q = q.reshape(5)
    return -sum([p[i]*log2(q[i]) for i in range(len(p))])
 

def custom_loss_function() :

    import tensorflow as tf
    classifier = tf.keras.models.load_model('/home/mtech/2020/satyaki_ghosh/codes/Rajesh/Camera-model identifier/greenchannel_patches10_classes5.h5')
    
    def loss(y_actual,y_predicted) :      
        print(f'Shape of ypred: {y_predicted.shape}')
        Lp = K.mean(K.abs(y_actual - y_predicted))
        t = [0,1,0,0,0]
        Lc = cross_entropy(t,classifier.predict(y_predicted, steps = 1))
        La = K.log(y_predicted.shape[0] - K.sum(discriminator.predict(y_predicted)))
        Lg = alpha * Lp + beta * Lc + gamma * La
        return Lg
    return loss


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
        plt.imshow(np.squeeze(y_pred_GAN[i]))
	# save plot to file
    plt.savefig('./save_path/generated_plot_%03d.png' % (epoch+1))
    plt.close()
	# save the generator model
    generator.save('./save_path/generator_%03d_%05d.h5' % (epoch+1, g_loss[0]))



# CREATE GENERATOR
optimizer = Adam(0.0001, 0.5)
def create_generator():
    generator = Sequential()
    
    generator.add(Conv2D(64, (3, 3), strides = 1, padding = 'same', input_shape=(IMG_SIZE,IMG_SIZE,channels)))
    generator.add(Activation('relu'))
    generator.add(Conv2D(64, (3, 3), strides = 1, padding = 'same'))
    generator.add(Activation('relu'))
    generator.add(Conv2D(64, (1, 1), strides = 1, padding = 'same'))
    generator.add(Activation('relu'))



    generator.add(Conv2D(128, (3, 3), strides = 1, padding = 'same'))
    generator.add(Activation('relu'))
    generator.add(Conv2D(128, (3, 3), strides = 1, padding = 'same'))
    generator.add(Activation('relu'))
    generator.add(Conv2D(128, (1, 1), strides = 1, padding = 'same'))
    generator.add(Activation('relu'))

    generator.add(Conv2D(1, (3, 3), strides = 1, padding = 'same'))
    generator.add(Activation('relu'))
    
    generator.compile(loss=custom_loss_function(), optimizer=optimizer, run_eagerly = True)
    return generator



def normalise(w):
    j=int(w.shape[0]/2)
    for i in range(w.shape[-1]):
        w[j,j,:,i]=0
        wsum=w[:,:,:,i].sum()
        w[:,:,:,i]/=wsum
        w[j,j,:,i]=-1
    return w




# CREATE DISCRIMINATOR
num_filter=3
w = np.random.rand(5,5,channels,num_filter)          #changing the number of filters change accordingly the last number 
wgt = normalise(w)
bias = np.zeros(num_filter)


# define the standalone discriminator model
def create_discriminator():
    model = Sequential()

    model.add(Conv2D(3, (5, 5), input_shape=(IMG_SIZE,IMG_SIZE,channels)))       #convRes
    model.layers[0].set_weights([wgt, bias])

    model.add(Conv2D(96,(7,7),strides = (2,2), padding = 'same'))
    model.add(BatchNormalization())
    model.add(Activation('tanh'))
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
    model.compile(loss = "binary_crossentropy", optimizer = optimizer, metrics = ['accuracy'])
    return model




import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

discriminator = create_discriminator()
generator = create_generator()
classifier = tf.keras.models.load_model('/home/mtech/2020/satyaki_ghosh/codes/Rajesh/Camera-model identifier/greenchannel_patches10_classes5.h5')
classifier._name = 'trainedmodule'
discriminator.trainable = False
classifier.trainable = False

gan_input = Input(shape=(IMG_SIZE,IMG_SIZE,channels))
fake_image = generator(gan_input)

gan_output = discriminator(fake_image)

gan_output2 = classifier(fake_image)

gan = Model(inputs = [gan_input], outputs = [gan_output, gan_output2])
gan.compile(loss='binary_crossentropy', optimizer=optimizer)
print("Generator and Discriminator initialized. Starting training...")
   
d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
set = np.random.randint(0, X_test_GAN_image.shape[0], size=25)


for epoch in range(epochs):
    for batch in range(steps_per_epoch):
        # print("Starting GAN training")
        seed_epoch = seed[np.random.randint(0, seed.shape[0], size=batch_size)]
        seed_epoch = np.asarray(seed_epoch).astype('float32')
        fake_x = generator.predict(seed_epoch)

        # print("Starting training of discriminator")
        real_x = x_train[np.random.randint(0, x_train.shape[0], size=batch_size)]        
        x = np.concatenate((real_x, fake_x))
        disc_y = np.zeros(2*batch_size)
        disc_y[:batch_size] = 0.9
        x = np.asarray(x).astype('float32')
        d_loss,d_acc = discriminator.train_on_batch(x, disc_y)
        # print("Discriminator trained")
        
        total_w = discriminator.layers[0].get_weights()
        w=np.array(total_w[0])
        bias = np.array(total_w[1])
        w = normalise(w)
        discriminator.layers[0].set_weights([w,bias])
        # print("Discriminator constrained layer updated")

        t = np.array([np.array([0,1,0,0,0]).transpose() ]* batch_size)
        y_gen = np.ones(batch_size)
        g_loss = gan.train_on_batch([seed_epoch], [y_gen,t])
        # print("Generator trained.")

        d1_hist.append(d_loss)
        d2_hist.append(d_acc)
        g_hist.append(g_loss)
        print(f'Batch_number: {batch}  Discriminator Loss: {d_loss}  Discriminator accuracy: {d_acc}  Generator Loss: {g_loss}')
    print(f'Epoch: {epoch} \t Discriminator Loss: {d_loss} \t\t Generator Loss: {g_loss}')

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


input = patches[0]
input = np.expand_dims(input,axis=0)
input = np.expand_dims(input,axis=3)
input =input/255
np.argmax(classifier.predict(input))
plt.imshow(np.squeeze(input))


output = generator.predict(input)

output = output*255
np.argmax(classifier.predict(output))
plt.imshow(np.squeeze(output))


# TODOS:
# generator output should be 3 channel instead of 1
# generator outputs only black images
# train classifier for single channel (green)
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