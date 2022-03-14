import keras 
import tensorflow as tf
import numpy as np
import tensorflow as tf
from config import NUM_CLASSES
from models.residual_block import make_basic_block_layer, make_bottleneck_layer
class multimodalnn(tf.keras.Model):
    def __init__(self,layer_params=[3,4,6,3]):
        super(multimodalnn, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(filters=8,
                                            kernel_size=(5, 5),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=8,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=16,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=32,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)
        
        self.conv2 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(2,1),
                                            strides=2,
                                            padding="same")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(1,1),
                                               strides=2,
                                               padding="same")
        self.layer5 = tf.keras.layers.Conv2D(filters=128,
                                            kernel_size=(2,1),
                                            strides=2,
                                            padding="same")
        self.layer6 = tf.keras.layers.BatchNormalization()
        self.layer7 = tf.keras.layers.MaxPool2D(pool_size=(1,1),
                                               strides=2,
                                               padding="same")
        self.layer8 = make_basic_block_layer(filter_num=256,
                                             blocks=2,
                                             stride=2,firstLayerKernelsize=1)
        '''self.layer5 = make_basic_block_layer(filter_num=64,
                                             blocks=2,firstLayerKernelsize=1)
        self.layer6 = make_basic_block_layer(filter_num=128,
                                             blocks=2,
                                             stride=2,firstLayerKernelsize=1)
        self.layer7 = make_basic_block_layer(filter_num=256,
                                             blocks=2,
                                             stride=2,firstLayerKernelsize=1)
        self.layer8 = make_basic_block_layer(filter_num=512,
                                             blocks=2,
                                             stride=2,firstLayerKernelsize=1)'''
    def call(self, inputs,landmarksinput,training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        y = self.conv2(landmarksinput)
        y = self.bn2(y, training=training)
        y = tf.nn.relu(y)
        y = self.pool2(y)
        y = self.layer5(y, training=training)
        y = self.layer6(y, training=training)
        y = self.layer7(y, training=training)
        y = self.layer8(y, training=training)

        y = self.avgpool(y)
        
        z =  tf.concat([x,y],1)
        output = self.fc(z)

        return output
'''
file = "newdata/postive-iwata/negative/20211115-142727_Task1_iwata_Column_076078.jpg"
from PIL.Image import new
import imageio
from mlxtend.image import extract_face_landmarks
import matplotlib.pyplot as plt
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import glob
import numpy as np
from tensorflow.python.keras.preprocessing.image import array_to_img
import os
import face_recognition 
import cv2
import csv
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_face_mesh = mp.solutions.face_mesh
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image,ImageDraw
    # For static images:
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5) as face_mesh:
    for idx, file in enumerate(glob.glob(file)):
        #image = cv2.imread(file)
        image=face_recognition.load_image_file(file)
        face_location = face_recognition.face_locations(image,0,"cnn")
        for face in face_location:
            top,right,bottom,left = face
            face = image[top:bottom,left:right]
        # Convert the BGR image to RGB before processing.
        results = face_mesh.process(face)
        plt.imshow(face)
        plt.show()
    # Print and draw face mesh landmarks on the image.
        if not results.multi_face_landmarks:
            continue
        annotated_image = image.copy()
        for face_landmarks in results.multi_face_landmarks:
            landmarks =[]
            x=[]
            y=[]
            z=[]
            qq = np.zeros((image.shape[0], image.shape[1],3))
            print('face_landmarks:', face_landmarks)
            #mp_drawing.draw_landmarks(image, face_landmarks,mp_face_mesh.FACEMESH_TESSELATION,None,mp_drawing_styles
        #   .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(image, face_landmarks,mp_face_mesh.FACEMESH_CONTOURS,None,mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
            ih,iw,ic = image.shape
            for idx,lm in enumerate(face_landmarks.landmark):
                landmarks.append([int(lm.x*iw)+int(lm.y*ih)+int(lm.z*ih), abs(int(lm.x*iw)+int(lm.y*ih)+int(lm.z*ih))])
                x.append(int(lm.x*iw))
                y.append(int(lm.y*ih))
                z.append(int(lm.z*ih))
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(x,y,z,s=10,c="green")
        plt.show()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        for p in range(len(landmarks)):
            qq[landmarks[p][1]:landmarks[p][1]+2,landmarks[p][0]:landmarks[p][0]+2,:] =[255,255,255] 
        plt.imshow(qq)
        plt.show()'''
