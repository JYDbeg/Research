from __future__ import absolute_import, division, print_function
import tensorflow as tf
from models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
import config
from prepare_data import generate_datasets
import math
import cv2
import imageio
from mlxtend.image import extract_face_landmarks
from train import get_model
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
model = get_model()
types = ["jpg","jpeg"]
smile = []
normal=[]
net =[]
allimages= []
labels = ["happiness","neural"]
target = []
model.load_weights(filepath = config.save_model_dir)
'''
for t in types:
    for index,file in enumerate(glob.glob("netgazou/*.{}".format(t))):
        img = tf.io.read_file(file)
        img = tf.image.decode_image(img,3)
        img =tf.image.resize(img,(224,224))/255.0
        net.append(img)'''
'''
for t in labels:
    if t =="normal":
        continue
    for index,file in enumerate(glob.glob("netgazou/{}/*.jpg".format(t))):
        print(file)
        img = tf.io.read_file(file)
        img = tf.image.decode_jpeg(img,3)
        img =tf.image.resize(img,(224,224))/255.0
        target.append(labels.index(t))
        net.append(img)

        if index ==50:
            break
        '''
if not os.path.exists("./FEDB/Landmark/happiness"):
        os.makedirs("./FEDB/Landmark/happiness")
if not os.path.exists("./FEDB/Landmark/neural"):
        os.makedirs("./FEDB/Landmark/neural")
'''
for index,file in enumerate(glob.glob("dataset/test/happiness/*.jpg")):
    try:
        img = imageio.imread(file)
        img = cv2.resize(img,(224,224))
        landmarks = extract_face_landmarks(img)
        fig = plt.figure()
        plt.axis("off")

        plt.scatter(landmarks[:, 0], -landmarks[:, 1], alpha=0.8)
        fig.savefig("./FEDB/Landmark/happiness/{}.jpg".format(index))
        imageio.imwrite("./FEDB/Landmarks/happiness/{}.jpg".format(index),img)
        allimages.append(img)
        img = tf.io.read_file("./FEDB/Landmark/happiness/{}.jpg".format(index))
        img = tf.image.decode_jpeg(img,3)
        img =tf.image.resize(img,(224,224))/255.0
        target.append(0)
        net.append(img)
        plt.close()
    except:
        continue
 
for index,file in enumerate(glob.glob("dataset/test/neural/*.jpg")):
    try:
        img = imageio.imread(file)
        img = cv2.resize(img,(224,224))
        landmarks = extract_face_landmarks(img)
        fig = plt.figure()
        plt.axis("off")
        plt.scatter(landmarks[:, 0], -landmarks[:, 1], alpha=0.8)
        allimages.append(img)
        fig.savefig("./FEDB/Landmark/neural/{}.jpg".format(index))
        imageio.imwrite("./FEDB/Landmarks/neural/{}.jpg".format(index),img)
        img = tf.io.read_file("./FEDB/Landmark/neural/{}.jpg".format(index))
        img = tf.image.decode_jpeg(img,3)
        img =tf.image.resize(img,(224,224))/255.0
        target.append(1)
        net.append(img)
        plt.close()
    except:
        continue
'''
'''
for index,file in enumerate(glob.glob("./FEDB/Landmarks/happiness/*.jpg")):

    print(file)
    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img,3)
    img =tf.image.resize(img,(224,224))/255.0
    target.append(0)
    allimages.append(img)

for index,file in enumerate(glob.glob("./FEDB/Landmarks/neural/*.jpg")):
    print(file)

    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img,3)
    img =tf.image.resize(img,(224,224))/255.0
    target.append(1)
    allimages.append(img)
'''

for index,file in enumerate(glob.glob("./dataset/test/happiness/*.jpg")):

    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img,3)
    img =tf.image.resize(img,(48,48))/255.0
    net.append(img)
    target.append(0)
 
for index,file in enumerate(glob.glob("./dataset/test/neural/*.jpg")):

    img = tf.io.read_file(file)
    img = tf.image.decode_jpeg(img,3)
    img =tf.image.resize(img,(48,48))/255.0
    net.append(img)
    target.append(1)

    #a=tf.random.uniform(shape=[224,224,1], minval=0., maxval=1.0)
'''
netsd =[]
for item in net:
    for i in range(30):
        netsd.append(item + i*0.01)'''
#for i in range(100):
#a =tf.ones([224,224,3])*0.8
    #b = tf.zeros([224,224,3])+i*0.01
#net.append(a)
    #net.append(b)
pa = "facepluslandrtest"
result = model(net)
for index,item in enumerate(net):
    #if tf.math.argmax(result,1)[index] != target[index]:
    if not os.path.exists("./FEDB/90/{}/all".format(pa)):
        os.makedirs("./FEDB/90/{}/all".format(pa))
    if not os.path.exists("./FEDB/90/{}/miss".format(pa)):
        os.makedirs("./FEDB/90/{}/miss".format(pa))
    if not os.path.exists("./FEDB/90/{}/miss/happiness".format(pa)):
            os.makedirs("./FEDB/90/{}/miss/happiness".format(pa))
    if not os.path.exists("./FEDB/90/{}/miss/neural".format(pa)):
        os.makedirs("./FEDB/90/{}/miss/neural".format(pa))   
    fig = plt.figure()
    plt.imshow(item)
    #plt.title(labels[tf.math.argmax(result,1)[index]])
    plt.title(labels[tf.math.argmax(result,1)[index]]+"  target="+labels[target[index]])
    plt.text(5,25,"Happiness P:"+str(round(result[index][0].numpy()*100,3))+"%"+"\nNeural P:"+str(round(result[index][1].numpy()*100,3))+"%",color = "red")
    #plt.show()
    fig.savefig("./FEDB/90/{}/all/".format(pa)+str(index)+".jpg")
    plt.close()
    
    if tf.math.argmax(result,1)[index] != target[index]:
        fig = plt.figure()
        plt.imshow(item)
        #plt.title(labels[tf.math.argmax(result,1)[index]])
        plt.title(labels[tf.math.argmax(result,1)[index]]+"  target="+labels[target[index]])
        plt.text(5,25,"Happiness P:"+str(round(result[index][0].numpy()*100,3))+"%"+"\nNeural P:"+str(round(result[index][1].numpy()*100,3))+"%",color = "red")
        #plt.show()
        if target[index] ==0:
            fig.savefig("./FEDB/90/{}/miss/happiness/".format(pa)+str(index)+".jpg")
        if target[index] ==1:
            fig.savefig("./FEDB/90/{}/miss/neural/".format(pa)+str(index)+".jpg")
        plt.close()
        
print(result)
print(tf.math.argmax(result,1))