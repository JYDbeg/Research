import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import glob
import numpy as np
from tensorflow.python.keras.preprocessing.image import array_to_img
import os
labels = ["happiness","neural"]
himgs = []
nimgs = []
expr = ["angcl","angop","discl","disop","exc","fea","hap","rel","sad","sle","sur"]
datagen = ImageDataGenerator(rotation_range=5,width_shift_range=0.2,height_shift_range=0.1,shear_range=0,zoom_range=[0.8,1.1],horizontal_flip=True,vertical_flip=False,fill_mode="constant",cval = 255)
names=["f01","f02","f03","f04","m01","m02","m03","m04"]
name_s=["f04","m04"]
'''
for label in labels:
    for file in glob.glob("dataset/train/{}/*.jpg".format(label)):
        img = image.load_img(file)
        x = image.img_to_array(img)
        if label == "happiness":
            himgs.append(x)
        if label == "neural":
            nimgs.append(x)
himgs = np.array(himgs)
nimgs = np.array(nimgs)

ih =[]
ins = []
for d in datagen.flow(himgs,batch_size=1):
    ih.append(image.array_to_img(d[0],scale=True))
    if (len(ih))==2000 :
        break
for d in datagen.flow(nimgs,batch_size=1):
    ins.append(image.array_to_img(d[0],scale=True))
    if (len(ins))==2000 :
        break
'''

def save_image(n,label):
    for index,i in enumerate(n):
        image.save_img("dataset/data_aument/{}/{}.jpg".format(label,index),i)
def save_image_root(name,n,label):
    if not os.path.exists(f"ha_alll/{name}/{label}/{label}/augment/"):
        os.makedirs(f"ha_alll/{name}/{label}/{label}/augment/")
    
    for index,i in enumerate(n):
        image.save_img(f"ha_alll/{name}/{label}/{label}/augment/{name}{label}{index}.jpg",i)

#save_image(ih,labels[0])
#save_image(ins,labels[1])


for name in names:
    for label in expr:
        imgs = []
        ih = []
        for file in glob.glob("ha_alll/{}/{}/{}/*.jpg".format(name,label,label)):
            img = image.load_img(file)
            x = image.img_to_array(img)   
            imgs.append(x)      
        imgs = np.array(imgs)

        for d in datagen.flow(imgs,batch_size=1):
            ih.append(image.array_to_img(d[0],scale=True))
            if len(ih) == 20:
                break
        save_image_root(name,ih,label)
        imgs = []
        ih = []
        for file in glob.glob("ha_alll/{}/{}/neural/*.jpg".format(name,label)):
            img = image.load_img(file)
            x = image.img_to_array(img)   
            imgs.append(x)      
        imgs = np.array(imgs)

        for d in datagen.flow(imgs,batch_size=1):
            ih.append(image.array_to_img(d[0],scale=True))
            if len(ih) == 5:
                break
        if not os.path.exists(f"ha_alll/{name}/{label}/neural/augment/"):
            os.makedirs(f"ha_alll/{name}/{label}/neural/augment/")
    
        for index,i in enumerate(ih):
            image.save_img(f"ha_alll/{name}/{label}/neural/augment/{name}neural{index}.jpg",i)






