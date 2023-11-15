
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import cv2
from sklearn.feature_extraction import image


DATADIR = "/home/mtech/2020/satyaki_ghosh/datasets/Dresden"
camera_model = "Kodak_M1063"
IMG_SIZE = 256
patches_per_photo = 3


x_train = []

def create_training_data() :
    path = os.path.join(DATADIR, camera_model)
    photos = os.listdir(path)
    random.shuffle(photos)
    for img in photos:
      img_array = cv2.imread(os.path.join(path,img))
      patches = image.extract_patches_2d(img_array[:,:,1], (256, 256),max_patches = patches_per_photo)
      for i in range(patches_per_photo) :
        x_train.append([patches[i]])
      if len(x_train) > 2000: 
        break
    # save the dataset
    dummy = np.asarray(x_train, dtype=object)
    with open('training_patches.npy','wb') as f:
        np.save(f,dummy)

create_training_data()

