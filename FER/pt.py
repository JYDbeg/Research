import timm
import torch
import torch.nn as nn
import torch.optim as optim 
import tfimm
import tensorflow as tf
from tensorflow.keras import Sequential
import os
import glob
from prepare_data import load_and_preprocess_image
import numpy as np
class mymodel(tf.keras.Model):
    def __init__(self):
        super(mymodel, self).__init__()
        self.preprocess = tfimm.create_model("poolformer_s12",pretrained=True,nb_classes=0,in_channels=1,input_size=(164,164))
        self.preprocess.trainable = False
        self.preprocess.layers[-2].trainable = True
        self.preprocess.layers[-1].trainable = True
        self.dense1 = tf.keras.layers.Dense(3)
        self.lstm1 = tf.keras.layers.LSTM(50,return_sequences=True)
        self.lstm2 = tf.keras.layers.LSTM(50)
        self.dense2 = tf.keras.layers.Dense(3)
        self.flate1 = tf.keras.layers.Flatten()
        self.flate2 = tf.keras.layers.Flatten()
    def call(self, inputs,time_series,training=None, mask=None):
        x = self.preprocess(inputs)
        #x = self.dense1(x,training=training)
        #x = tf.nn.softmax(x)
        x = self.flate1(x)
        y = self.lstm1(time_series,training=training)
        y = self.lstm2(y,training=training)
        y = self.flate1(y)
        y = tf.concat([x,y],1)
        y = self.dense2(y,training=training)
        y = tf.nn.softmax(y)
        #z =  0.2*x+0.8*y
        

        return y
'''
preprocess = tfimm.create_model("poolformer_s12",pretrained=True,nb_classes=0,in_channels=1,input_size=(164,164))
preprocess.trainable = False
preprocess.layers[-2].trainable = True
preprocess.layers[-1].trainable = True
model = Sequential(name="my_model")
model.add(preprocess)
#model.add(tf.keras.layers.Dense(3,activation="softmax"))'''
def sw(stride,frames,windows):
    new_window = []
    for i in range(0,len(windows)-frames+1,stride):
        new_window.append(windows[i:i+frames])
    return new_window
def swf(stride,frames,windows):
    new_window = []
    for i in range(0,len(windows)-frames+1,stride):
        new_window.append(tf.math.add_n(windows[i:i+frames])/frames)
    return new_window
model = mymodel()

#spo2 = []
time_series = []
valid_time_series = []
names = ["iwata","nisio","tanaka","miymoto"]
validname = ["tanaka"]
expr = ["negative","postive","normal"]
face_images = []
valid_face_images = []
targets =[]
valid_targets = []

#valid_spo2 = []
'''for i in range(3):
    valid_time_series.append([(36+np.random.rand())/37.5,(50+np.random.randint(10,20))/80,(95+np.random.randint(5,15))/130])
for i in range(3):
    time_series.append([(36+np.random.rand())/37.5,(50+np.random.randint(10,20))/80,(95+np.random.randint(5,15))/130])'''

for name in names:
    if name in validname:
        for ex in expr:
            valid_target = []
            valid_face_image = []
            valid_tempretura = []
            valid_hr = []
            valid_bp = []
            valid_time_serie = []
            for file in glob.glob(f"newdata/faceonly/{name}/{ex}/*.jpg"):
                valid_face_image.append(load_and_preprocess_image(file,image_height=164,image_width=164,channels=1))
                valid_target.append(expr.index(ex))
                if ex =="negative":
                    valid_tempretura.append((36+np.random.rand()*0.5)/37)
                    valid_hr.append((50+np.random.randint(15,30))/80)
                    valid_bp.append((95+np.random.randint(10,20))/130)
                if ex =="postive":
                    valid_tempretura.append((36.2+np.random.rand()*0.5)/37)
                    valid_hr.append((50+np.random.randint(10,20))/80)
                    valid_bp.append((95+np.random.randint(5,10))/130)
                if ex =="normal":
                    valid_tempretura.append((36.2+np.random.rand()*0.3)/37)
                    valid_hr.append((50+np.random.randint(10,20))/80)
                    valid_bp.append((95+np.random.randint(5,15))/130)
            for i in range(len(valid_tempretura)):
                valid_time_serie.append([valid_tempretura[i],valid_hr[i],valid_bp[i]])
            valid_targets+=valid_target[2:]
            valid_face_images+=swf(1,3,valid_face_image)
            valid_time_series+=sw(1,3,valid_time_serie)
        continue
    for ex in expr:
        target = []
        face_image = []
        tempretura = []
        hr = []
        time_serie=[]
        bp = []
        for file in glob.glob(f"newdata/faceonly/{name}/{ex}/*.jpg"):
            face_image.append(load_and_preprocess_image(file,image_height=164,image_width=164,channels=1))
            target.append(expr.index(ex))
            if ex =="negative":
                tempretura.append((36+np.random.rand()*0.5)/37)
                hr.append((50+np.random.randint(15,30))/80)
                bp.append((95+np.random.randint(10,20))/130)
            if ex =="postive":
                tempretura.append((36.2+np.random.rand()*0.5)/37)
                hr.append((50+np.random.randint(10,20))/80)
                bp.append((95+np.random.randint(5,10))/130)
            if ex =="normal":
                tempretura.append((36.2+np.random.rand()*0.3)/37)
                hr.append((50+np.random.randint(10,20))/80)
                bp.append((95+np.random.randint(5,15))/130)
        for i in range(len(tempretura)):
            time_serie.append([tempretura[i],hr[i],bp[i]])
        time_series+=sw(1,3,time_serie)
        face_images+=swf(1,3,face_image)
        targets+=target[2:]
'''for file in glob.glob(f"KMU-FED/onlyface/hukai/*.jpg"):
    face_image.append(load_and_preprocess_image(file,image_height=164,image_width=164,channels=1))
    target.append(0)
    tempretura.append(36.2+np.random.rand())
    hr.append(50+np.random.randint(20,30))
    spo2.append(6+np.random.rand())
for file in glob.glob(f"KMU-FED/onlyface/kai/*.jpg"):
    face_image.append(load_and_preprocess_image(file,image_height=164,image_width=164,channels=1))
    target.append(1)
    tempretura.append(36.3+np.random.rand())
    hr.append(50+np.random.randint(10,20))
    spo2.append(5+np.random.rand())
for i in range(len(tempretura)):
    time_series.append([tempretura[i],hr[i],bp[i]])
for i in range(len(valid_tempretura)):
    valid_time_series.append([valid_tempretura[i],valid_hr[i],valid_bp[i]])
time_series = sw(1,3,time_series)
valid_time_series = sw(1,3,valid_time_series)'''
train_value = tf.data.Dataset.from_tensor_slices(face_images)
train_label = tf.data.Dataset.from_tensor_slices(targets)
train_time = tf.data.Dataset.from_tensor_slices(time_series)
train_dataset = tf.data.Dataset.zip((train_value,train_time, train_label))
valid_value = tf.data.Dataset.from_tensor_slices(valid_face_images)
valid_label = tf.data.Dataset.from_tensor_slices(valid_targets)
valid_time = tf.data.Dataset.from_tensor_slices(valid_time_series)
valid_dataset = tf.data.Dataset.zip((valid_value,valid_time, valid_label))
train_count = len(face_images)
valid_count = len(valid_face_images)
train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=32)
valid_dataset = valid_dataset.batch(batch_size=32)

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.0001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
epochs = 100
@tf.function
def train_step_image(images,time_series, labels):
    with tf.GradientTape() as tape:
        predictions = model(images,time_series, training=True)
        loss = loss_object(y_true=labels, y_pred=predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def valid_step_image(images,time_series, labels):
    predictions = model(images,time_series, training=False)
    v_loss = loss_object(labels, predictions)

    valid_loss(v_loss)
    valid_accuracy(labels, predictions)    
import math
import matplotlib.pyplot as plt
result_all_accuracy = []
min_val_loss = 1e9
result_train_accuracy = []
max_valid_accuracy = 0
esepoch = 0
for epoch in range(epochs):
    if esepoch>10:
        break
    train_loss.reset_states()
    train_accuracy.reset_states()
    valid_loss.reset_states()
    valid_accuracy.reset_states()
    step = 0
    for image,time_series, labels in train_dataset:
        step += 1
        train_step_image(image,time_series, labels)
        print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                epochs,
                                                                                step,
                                                                                math.ceil(train_count / 32),
                                                                                train_loss.result(),
                                                                                train_accuracy.result()))

    for image,time_series, valid_labels in valid_dataset:
        valid_step_image(image,time_series, valid_labels)

    print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
        "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                            epochs,
                                                            train_loss.result(),
                                                            train_accuracy.result(),
                                                            valid_loss.result(),
                                                            valid_accuracy.result()))
    if epoch>20:
        if valid_loss.result().numpy()>min_val_loss:
            esepoch+=1
    min_val_loss = min(min_val_loss,valid_loss.result().numpy())
    max_valid_accuracy = max(max_valid_accuracy,valid_accuracy.result().numpy())
    result_all_accuracy.append(valid_accuracy.result().numpy())
    result_train_accuracy.append(train_accuracy.result().numpy())
fig = plt.figure()
plt.plot(range(len(result_train_accuracy)), result_train_accuracy,label="train_accuracy")
plt.plot(range(len(result_all_accuracy)), result_all_accuracy,label="valid_accuracy")
plt.xlabel("epoch")
plt.ylabel("accuracy")

plt.legend()
fig.savefig(f"{validname[0]}times+face.png")
print(max_valid_accuracy)