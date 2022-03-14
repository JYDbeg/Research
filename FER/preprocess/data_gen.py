
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import glob
import numpy as np
from tensorflow.python.keras.preprocessing.image import array_to_img
import os
#水増し
labels = ["happiness","neural"]
expr = ["angcl","angop","discl","disop","exc","fea","hap","rel","sad","sle","sur"]
datagen = ImageDataGenerator(rotation_range=5,width_shift_range=0.2,height_shift_range=0.1,shear_range=0,zoom_range=[0.8,1.1],horizontal_flip=True,vertical_flip=False,fill_mode="constant",cval = 255)
names=["f01","f02","f03","f04","m01","m02","m03","m04"]


def save_image(name,n,label):
    if not os.path.exists(f"ha_alll/{name}/{label}/{label}/augment/"):
        os.makedirs(f"ha_alll/{name}/{label}/{label}/augment/")
    for index,i in enumerate(n):
        image.save_img(f"ha_alll/{name}/{label}/{label}/augment/{name}{label}{index}.jpg",i)

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
        save_image(name,ih,label)







