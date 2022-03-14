import numpy as np
from keras.preprocessing import image
import glob
from matplotlib import pyplot as plt
from PIL import Image
import tensorflow as tf
names=["f01","f02","f03","m01","m02","m03"]
labels = ["happiness","neural"]

def data_agumention(n,i):
    x = tf.image.random_brightness(n,i*0.005)
    y = tf.image.random_contrast(n,0,i*0.04)
    image.save_img(f"dataset/train/contrast/{label}/{name}{i}x.jpg",x)
    image.save_img(f"dataset/train/contrast/{label}/{name}{i}y.jpg",y)

for name in names:
    for label in labels:
        for file in glob.glob(f"ha_j/{name}/{label}/onlyface/*.jpg"):
            img = tf.io.read_file(file)
            img = tf.image.decode_jpeg(img,3)
            #x =tf.image.resize(img,(224,224))/255.0 
            #a = np.sum(x,axis=2)>2.8
            #a= np.stack([a,a,a],axis=2)
            #x = x-a
            for i in range(200):
                data_agumention(img,i+1)