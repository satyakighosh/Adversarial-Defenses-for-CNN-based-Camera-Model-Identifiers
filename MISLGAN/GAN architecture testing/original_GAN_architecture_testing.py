#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 13:29:30 2021

@author: gullipalli.rajesh
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2



batch_size = 30
steps_per_epoch = 70
epochs = 30


DATADIR = "/home/mtech/gullipalli.rajesh/GAN_training"

seed = []

IMG_SIZE = 256

from sklearn.feature_extraction import image

def create_seed_training_data() :
    path = os.path.join(DATADIR,"Seed_images")
    for img in os.listdir(path) :
      img_array = cv2.imread(os.path.join(path,img))
      patches = image.extract_patches_2d(img_array, (256, 256),max_patches = 1)
      for i in range(1) :
        
        srcArray = patches[i]
        # Create target array, twice the size of the original image
        resArray = np.zeros((256, 256, 1))
        
        # Map the RGB values in the original picture according to the BGGR pattern# 
        
        # Blue
        resArray[::2, ::2, 0] = srcArray[::2, ::2, 2]
        
        # Green (top row of the Bayer matrix)
        resArray[1::2, ::2, 0] = srcArray[1::2, ::2, 1]
        
        # Green (bottom row of the Bayer matrix)
        resArray[::2, 1::2, 0] = srcArray[::2, 1::2, 1]
        
        # Red
        resArray[1::2, 1::2, 0] = srcArray[1::2, 1::2, 0] 
                  
        seed.append(patches[i])
        


create_seed_training_data()


import random
random.shuffle(seed)
seed = np.array(seed)
seed = seed.reshape((-1,IMG_SIZE, IMG_SIZE, 3))
seed = seed/255.0

x_train = []
def create_training_data() :
    path = os.path.join(DATADIR,"Kodak_M1063")
    for img in os.listdir(path) :
      img_array = cv2.imread(os.path.join(path,img))
      patches = image.extract_patches_2d(img_array, (256, 256),max_patches = 1)
      for i in range(1) :
        x_train.append([patches[i]])

create_training_data()

x_train = np.array(x_train)
x_train = x_train/255.0
x_train = x_train.reshape((-1, IMG_SIZE, IMG_SIZE, 3))


if not os.path.isdir("save_path"):
    os.mkdir("save_path")




import os
import cv2
import numpy as np
from sklearn.feature_extraction import image
CATEGORIES = ["Casio_EX-Z150","Kodak_M1063","Nikon_CoolPixS710","Olympus_mju_1050SW","Panasonic_DMC-FZ50"]



TESTDIR = "/home/mtech/gullipalli.rajesh/test_classes"

testing_data_GAN = []

def create_testing_data_GAN() :
  for category in CATEGORIES :
    path = os.path.join(TESTDIR,category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path) :
      img_array = cv2.imread(os.path.join(path,img))
      patches = image.extract_patches_2d(img_array, (256, 256),max_patches = 1)
      for i in range(1) :
        
        srcArray = patches[i]
        # Create target array, twice the size of the original image
        resArray = np.zeros((256, 256, 1))
        
        # Map the RGB values in the original picture according to the BGGR pattern# 
        
        # Blue
        resArray[::2, ::2, 0] = srcArray[::2, ::2, 2]
        
        # Green (top row of the Bayer matrix)
        resArray[1::2, ::2, 0] = srcArray[1::2, ::2, 1]
        
        # Green (bottom row of the Bayer matrix)
        resArray[::2, 1::2, 0] = srcArray[::2, 1::2, 1]
        
        # Red
        resArray[1::2, 1::2, 0] = srcArray[1::2, 1::2, 0] 
                  
        testing_data_GAN.append([patches[i],resArray,class_num])

create_testing_data_GAN()








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



def GAN_performance(epoch, generator,seed_test_bayer,seed_test_image):
	# prepare fake examples
    y_pred_GAN = generator.predict(seed_test_bayer)
    #seed_test_image = X_test_GAN_image[50,100,150,200,250,350,400,450,500,550,650,700,750,800,850,950,1000,1050,1100,1150,1250,1300,1350,1400,1450]
    for j in range(5 * 5):
        plt.subplot(5, 5, j+1)
        plt.axis('off')
        plt.imshow(np.squeeze(seed_test_image[j]))
    plt.savefig('/home/mtech/gullipalli.rajesh/save_path/seed_plot_%03d.png' % (epoch+1))
    plt.close()
    
	# plot images
    for i in range(5 * 5):
        plt.subplot(5, 5, i+1)
        plt.axis('off')
        plt.imshow(y_pred_GAN[i])
	# save plot to file
    plt.savefig('/home/mtech/gullipalli.rajesh/save_path/generated_plot_%03d.png' % (epoch+1))
    plt.close()
	# save the generator model
    generator.save('/home/mtech/gullipalli.rajesh/save_path/generator_%03d.h5' % (epoch+1))



'''
alpha = 1
beta = 1
gama = 1


import keras.backend as K

# calculate cross entropy
def cross_entropy(p, q):
    
    q = q.reshape(5)
    return 


def custom_loss_function(input) :
    
    
    
    
    def loss(y_actual,y_predicted) :
        import tensorflow as tf
        classifier = tf.keras.models.load_model('colourchannel_patches10_classes5.h5')
        import numpy as np
        prediction_loss = classifier.predict(y_predicted, steps = None)
        adversarial_loss = discriminator.predict(y_predicted,steps = None)
        t = np.array([0,1,0,0,0])
        x2 = 0
        x3 = 0
        x4 = 0
        from math import log2,log
        for j in range(input[0]) :
            x1 = 0
            for k in range(input[-1]) :
                variable1 = np.mean(np.abs(np.subtract(input[j,:,:,k] , y_predicted[j,:,:,k])))
                x1 = x1 + variable1
            x2 = x2 + x1/range(input[-1])
        Lp = x2/range(input[0])
        
        for j in range(input[0]) :
            variable2 = -sum([t[i]*log2(prediction_loss[j][i]) for i in range(len(t))])
            x3 = x3 + variable2
        Lc = x3/range(input[0])
            
        for j in range(input[0]) :
            variable3 = log(1 - np.sum(adversarial_loss[j]))
            x4 = x4 + variable3
        La = x4/range(input[0])
        
        Lg = Lp + Lc + La
        

        return Lg
    return loss


import numpy as np
from keras import backend as K
from keras.layers import Layer
class MyCustomLayer(Layer): 
   def __init__(self,input_data): 
      super(MyCustomLayer, self).__init__() 
   def call(self, input_data):
     rgb = input_data
     dmcg  = np.zeros(256,256,1)
     dmcg[::2, ::2, 0] = rgb[::2, ::2, 2]
     dmcg[1::2, ::2, 0] = rgb[1::2, ::2, 1]
     dmcg[::2, 1::2, 0] = rgb[::2, 1::2, 1]
     dmcg[1::2, 1::2, 0] = rgb[1::2, 1::2, 0]
     return dmcg
   def get_config(self):
       return


'''
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Dense, Activation, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

optimizer = Adam(0.0001, 0.5)
def create_generator(shape = (IMG_SIZE,IMG_SIZE,3)):
    input = Input(shape = (IMG_SIZE,IMG_SIZE,3))
    '''
    def custom_layer(input):
    
     
     srcArray = input
     #K.array(input).reshape(-1,256,256,3)
    
    # Create target array, twice the size of the original image
     resArray = np.zeros((256, 256, 1))
    
    # Map the RGB values in the original picture according to the BGGR pattern# 
    
    # Blue
     resArray[::2, ::2, 0] = srcArray[::2, ::2, 2]
    
    # Green (top row of the Bayer matrix)
     resArray[1::2, ::2, 0] = srcArray[1::2, ::2, 1]
    
    # Green (bottom row of the Bayer matrix)
     resArray[::2, 1::2, 0] = srcArray[::2, 1::2, 1]
    
    # Red
     resArray[1::2, 1::2, 0] = srcArray[1::2, 1::2, 0] 
     return resArray

    
    
    
    srcArray = input
          
    
    # Create target array, twice the size of the original image
    resArray = np.zeros((256, 256, 1))
    
    # Map the RGB values in the original picture according to the BGGR pattern# 
    
    # Blue
    resArray[::2, ::2, 0] = srcArray[::2, ::2, 2]
    
    # Green (top row of the Bayer matrix)
    resArray[1::2, ::2, 0] = srcArray[1::2, ::2, 1]
    
    # Green (bottom row of the Bayer matrix)
    resArray[::2, 1::2, 0] = srcArray[::2, 1::2, 1]
    
    # Red
    resArray[1::2, 1::2, 0] = srcArray[1::2, 1::2, 0]
    
    '''
    #layer_0 = MyCustomLayer(input)
    #lambda_layer = Lambda(custom_layer)(input) 
    
    
    
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




from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization



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
    model.compile(loss = "binary_crossentropy", optimizer = optimizer , metrics = ['accuracy'])

    return model
discriminator = create_discriminator()
generator = create_generator()



discriminator.trainable = False

gan_input = Input(shape=(IMG_SIZE,IMG_SIZE,3))
fake_image = generator(gan_input)

gan_output = discriminator(fake_image)


f
gan = Model(inputs = [gan_input],outputs = [gan_output])
gan.compile(loss='binary_crossentropy', optimizer=optimizer)

from matplotlib import pyplot

def plot_history(d1_hist, d2_hist, g_hist, a1_hist, a2_hist):
	# plot loss
	pyplot.subplot(2, 1, 1)
	pyplot.plot(d1_hist, label='d-real')
	pyplot.plot(d2_hist, label='d-fake')
	pyplot.plot(g_hist, label='gen')
	pyplot.legend()
	# plot discriminator accuracy
	pyplot.subplot(2, 1, 2)
	pyplot.plot(a1_hist, label='acc-real')
	pyplot.plot(a2_hist, label='acc-fake')
	pyplot.legend()
	# save plot to file
	
   
d1_hist, d2_hist, g_hist, a1_hist, a2_hist = list(), list(), list(), list(), list()
set = np.random.randint(0, X_test_GAN_image.shape[0], size=25)

for epoch in range(epochs):
    for batch in range(steps_per_epoch) :


        
        
        seed_epoch = seed[np.random.randint(0, seed.shape[0],size = batch_size)]#images of different classes
        fake_x = generator.predict(seed_epoch) #output of generator
        real_x = x_train[np.random.randint(0, x_train.shape[0],size = batch_size)] #images of target class
        
        d_train_y = np.zeros(2*batch_size)
        d_train_y[:batch_size] = 0.9
        d_train_x = np.concatenate((real_x,fake_x))
        
        d_loss,d_acc = discriminator.train_on_batch(d_train_x, d_train_y) #discriminator training 
        '''
        total_w = discriminator.layers[0].get_weights()
        w=np.array(total_w[0])
        bias = np.zeros(3)
        w = normalise(w)
        discriminator.layers[0].set_weights([w,bias])
        
        '''
        
        y_gen = np.ones(batch_size)
        g_loss = gan.train_on_batch([seed_epoch],[y_gen]) #generator training with different classes and keeping output to real
        
    
        d1_hist.append(d_loss)
        d2_hist.append(d_acc)
        g_hist.append(g_loss)
        
        
        print(f'Epoch: {epoch} \t d_loss: {d_loss}  d_acc: {d_acc} g_loss: {g_loss}')
    seed_test_image = X_test_GAN_image[set]
    seed_test_bayer = X_test_GAN_image[set]
    #seed_test = X_test_GAN_bayer[50,100,150,200,250,350,400,450,500,550,650,700,750,800,850,950,1000,1050,1100,1150,1250,1300,1350,1400,1450]
    GAN_performance(epoch, generator,seed_test_bayer,seed_test_image)
    

# plot generator and discriminator loss
pyplot.subplot(2, 1, 1)
pyplot.plot(d1_hist, label='d-loss')
pyplot.plot(g_hist, label='g_loss')
pyplot.legend()


# plot discriminator accuracy
pyplot.subplot(2, 1, 2)
pyplot.plot(d2_hist, label='d-acc')
pyplot.legend()
import tensorflow as tf
classifier = tf.keras.models.load_model('colourchannel_patches10_classes5.h5')    

from sklearn.metrics import confusion_matrix, classification_report
y_pred_GAN = generator.predict(X_test_GAN_image)
y_pred_classes = [np.argmax(element) for element in classifier.predict(y_pred_GAN)]



