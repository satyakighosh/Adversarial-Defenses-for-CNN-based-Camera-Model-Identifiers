import numpy as np

path = "E:\IIT GUWAHATI\MTP\MTP_CODES\Rajesh\Camera-model identifier\\"
print(path)
with open(path + "training_data.npy",'rb') as f:
    dummy = np.load(f, allow_pickle=True)
training_data = dummy

print(np.shape(training_data))