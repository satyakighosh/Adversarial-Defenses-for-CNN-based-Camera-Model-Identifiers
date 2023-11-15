
import numpy as np
import os
import json
import pickle
import cv2
import random
from sklearn.feature_extraction import image


DATADIR = "E:\IIT GUWAHATI\MTP\Datasets\Dresden"
CATEGORIES = [
    "D01_Samsung_GalaxyS3Mini",
    "D02_Apple_iPhone4s",
    "D03_Huawei_P9",
    "D04_LG_D290",
    "D05_Apple_iPhone5c",
    "D06_Apple_iPhone6"]

phototype = ["flat","nat"]
IMG_SIZE = 256
patches_per_photo = 2
training_data = []

def create_training_data() :
  # iterating through each camera-model
  for category in CATEGORIES:
    print(f"Camera-model : {category}")
    class_num = CATEGORIES.index(category)
    for ptype in phototype:
        path = os.path.join(DATADIR,category)
        path = os.path.join(path,ptype)
        # shuffling the images to avoid similar/close images
        photos = os.listdir(path)
        random.shuffle(photos)

        for img in photos:
        print(f'{img_count}. Processing {img}', end = ' .')
        img_count+=1
        img_array_test = cv2.imread(os.path.join(path,img))
        patches = image.extract_patches_2d(img_array_test, (256, 256),max_patches = patches_per_photo)
        for i in range(patches_per_photo) :
            training_data.append([patches[i],class_num])
        print("Done.")
        # taking the min number of photos of all camera-models to avoid imbalance (925 here for Casio)
        if img_count == 925:
            break   
  # save the dataset
  dummy = np.asarray(training_data, dtype=object)
  with open('training_data2.npy','wb') as f:
      np.save(f,dummy)

create_training_data()
