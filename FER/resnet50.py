import keras
from PIL import Image
import numpy as np
import glob
import os
from keras import optimizers
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense,Input, Dropout
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.applications.resnet50 import ResNet50

names=["f01","f02","f03","f04","m01","m02","m03","m04"]
labels = ["happiness","neural"]
root = "D:/BaiduNetdiskDownload/py/OpenFace_2.2.0_win_x64/kyou/resnet/TensorFlow2.0_ResNet-master/dataset/train/"
subset_1 = '/normal'
subset_2 = '/smile'

epochs = 30
batch_size = 8

X = []
Y = []
for name in names:
      for label in labels:
            for file in glob.glob(f'{root}/{label}/no90kao/*.jpg'):
                  image = Image.open(file)
                  image = image.convert("RGB")
                  image = image.resize((224,224))
                  data = np.asarray(image)
                  X.append(data)
                  Y.append(labels.index(label))
Y = np_utils.to_categorical(Y, 2)
X = np.array(X).reshape((len(X),224,224,3))
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3)
X_test,X_val,Y_test,Y_val = train_test_split(X_test,Y_test,test_size = 0.15)
image_size =224
datagen = ImageDataGenerator(rescale=1./255,rotation_range=10,zoom_range=[0.8,1.1],horizontal_flip=True,vertical_flip=False,fill_mode="constant",cval = 255)
datagen.fit(X_train)
input_tensor = Input(shape=(image_size, image_size, 3))
ResNet50 = ResNet50(include_top=False, weights='imagenet',input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=ResNet50.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.4))
top_model.add(Dense(2, activation='softmax'))

top_model = Model(inputs=ResNet50.input, outputs=top_model(ResNet50.output))
top_model.compile(loss='categorical_crossentropy',optimizer=optimizers.SGD(lr=1e-3, momentum=0.9),metrics=['accuracy'])

top_model.summary()
top_model.fit_generator(datagen.flow(X_train, Y_train,batch_size = batch_size),steps_per_epoch=len(X_train)/epochs,epochs=epochs)
result = top_model.predict(X_test)
result = np.argmax(result,axis = 1)
print(Y_test,result)
top_model.save("resnt50")