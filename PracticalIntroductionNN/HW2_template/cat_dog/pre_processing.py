from os import listdir
from numpy import asarray
import numpy as np
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import h5py
from sklearn.model_selection import train_test_split

train_dir = "PracticalIntroductionNN\HW2_template\cat_dog\\train\\"
test_dir = "PracticalIntroductionNN\HW2_template\cat_dog\\test1\\"
train_img, train_labels = [], []
test_img = []

f = h5py.File("D:\Work\EPE\PracticalIntroductionNN\HW2_template\cat_dog\\dc_227.h5", "w")

for file in listdir(train_dir):
    classes = 0.0
    if file.startswith("cat"):
        classes = 1.0

    photo = load_img(train_dir + file, target_size=(227,227))    
    photo = img_to_array(photo)
    photo /= 255

    train_img.append(photo)
    train_labels.append(classes)

train_img = asarray(train_img)
train_labels = asarray(train_labels)

train_img, test_img, train_labels, test_labels = train_test_split(train_img, train_labels, test_size=0.2, random_state=666)
train_img, val_img, train_labels, val_labels = train_test_split(train_img, train_labels, test_size=0.05, random_state=666)
f.create_dataset("train_img", data=train_img)
f.create_dataset("test_img", data=test_img)
f.create_dataset("train_lab", data=train_labels)
f.create_dataset("test_lab", data=test_labels)
f.create_dataset("val_lab", data=val_labels)
f.create_dataset("val_img", data=val_img)

"""train_img0, train_img1, train_labels0, train_labels1 = train_test_split(train_img, train_labels, test_size=0.5, random_state=1)
train_img0, test_img0, train_labels0, test_labels0 = train_test_split(train_img0, train_labels0, test_size=0.2, random_state=1)
train_img1, test_img1, train_labels1, test_labels1 = train_test_split(train_img1, train_labels1, test_size=0.2, random_state=1)

f.create_dataset("train_img0", data=train_img0)
f.create_dataset("test_img0", data=test_img0)
f.create_dataset("train_lab0", data=train_labels0)
f.create_dataset("test_lab0", data=test_labels0)

f.create_dataset("train_img1", data=train_img1)
f.create_dataset("test_img1", data=test_img1)
f.create_dataset("train_lab1", data=train_labels1)
f.create_dataset("test_lab1", data=test_labels1)"""

"""for file in listdir(test_dir):
    photo = load_img(test_dir + file, target_size=(255,255))    
    photo = img_to_array(photo)
    photo = asarray(photo)
    photo /= 255

    test_img.append(photo)


val_img = asarray(test_img)

f.create_dataset("val", data=val_img)"""
f.close()
# np.savez("dg_file.npz", train_img=train_img, train_labels=train_labels, test_img=test_img)