import numpy as np
import os
import json
import pickle
import cv2
import random
from sklearn.feature_extraction import image


DATADIR = "/home/mtech/2020/satyaki_ghosh/datasets/Dresden"
CATEGORIES = ["Casio_EX-Z150","Kodak_M1063","Nikon_CoolPixS710","Olympus_mju_1050SW","Panasonic_DMC-FZ50"]

IMG_SIZE = 256
patches_per_photo = 4
training_data = []


def create_seed_training_data() :
  # iterating through each camera-model
  for category in CATEGORIES : 
    
    print(f"Camera-model : {category}")
    path = os.path.join(DATADIR,category)
    img_count = 1

    # shuffling the images to avoid similar/close images
    photos = os.listdir(path)
    random.shuffle(photos)

    for img in photos:
      if img_count % 100 == 0:
        print(f'{img_count}. Processing {img}', end = ' .')
      img_count+=1
      img_array_test = cv2.imread(os.path.join(path,img))
      patches = image.extract_patches_2d(img_array_test[:,:,1], (256, 256), max_patches = patches_per_photo)
      for i in range(patches_per_photo) :
        training_data.append([patches[i]])
      # print("Done.")

      # taking 200 photos per class
      if img_count == 200:
        break   

  # save the dataset
  dummy = np.asarray(training_data, dtype=object)
  with open('seed_patches.npy','wb') as f:
      np.save(f,dummy)

create_seed_training_data()
