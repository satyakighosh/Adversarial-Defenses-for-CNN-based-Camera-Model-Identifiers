import numpy as np
from multiprocessing import dummy
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
from sklearn.feature_extraction import image



CATEGORIES = ["Casio_EX-Z150","Kodak_M1063","Nikon_CoolPixS710","Olympus_mju_1050SW","Panasonic_DMC-FZ50"]
TESTDIR = "/home/mtech/2020/satyaki_ghosh/datasets/Dresden"
testing_data_GAN = []


def create_testing_data_GAN() :
  for category in CATEGORIES :
    path = os.path.join(TESTDIR,category)
    class_num = CATEGORIES.index(category)
    img_count = 0
    # shuffling the images to avoid similar/close images
    photos = os.listdir(path)
    random.shuffle(photos)

    for img in os.listdir(path) :
      img_array = cv2.imread(os.path.join(path,img))
      img_count += 1
      patches = image.extract_patches_2d(img_array[:,:,1], (256, 256),max_patches = 1)
      for i in range(1) :
        
        srcArray = patches[i]
        # Create target array, twice the size of the original image
        resArray = np.zeros((256, 256, 1))
        resArray = srcArray
        
        # # Map the RGB values in the original picture according to the BGGR pattern# 
        
        # # Blue
        # resArray[::2, ::2, 0] = srcArray[::2, ::2, 2]
        
        # # Green (top row of the Bayer matrix)
        # resArray[1::2, ::2, 0] = srcArray[1::2, ::2, 1]
        
        # # Green (bottom row of the Bayer matrix)
        # resArray[::2, 1::2, 0] = srcArray[::2, 1::2, 1]
        
        # # Red
        # resArray[1::2, 1::2, 0] = srcArray[1::2, 1::2, 0] 
                  
        testing_data_GAN.append([patches[i],resArray,class_num])

      # taking 200 photos per class
      if img_count == 200:
        break   

  # save the dataset
  dummy = np.asarray(testing_data_GAN, dtype=object)
  with open('testing_patches.npy','wb') as f:
      np.save(f,dummy)


create_testing_data_GAN()

