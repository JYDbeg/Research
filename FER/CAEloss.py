from re import S
import tensorflow as tf
import config
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import glob
from multimodal import multimodalnn
from models.residual_block import make_basic_block_layer, make_bottleneck_layer
from keras.layers import Input,Dense,BatchNormalization,Conv2D,SeparableConv2D,MaxPool2D,UpSampling2D
from keras.models import Sequential
from prepare_data import load_and_preprocess_image
from sklearn.model_selection import train_test_split
import keras
class CAE(tf.keras.Model):
        def __init__(self):
            super(CAE, self).__init__()
            self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=(1, 2),
                                                strides=1,
                                                padding="same",activation="relu")
            self.bn1 = tf.keras.layers.BatchNormalization()
            self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(1, 1),
                                                    strides=(2,1),padding="same"
                                                    )
            self.conv2 = tf.keras.layers.Conv2D(filters=128,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same",activation="relu")
            self.bn2 = tf.keras.layers.BatchNormalization()
            self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(1, 1),
                                                    strides=(2,1),padding="same"
                                                    
                                                    )
            self.conv3 = tf.keras.layers.Conv2D(filters=256,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same",activation="relu")
            self.bn3 = tf.keras.layers.BatchNormalization()
            self.pool3 = tf.keras.layers.MaxPool2D(pool_size=(1, 1),
                                                    strides=(2,1),padding="same"
                                                    )
            self.conv4 = tf.keras.layers.Conv2D(filters=256,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same",activation="relu")
            self.bn4 = tf.keras.layers.BatchNormalization()
            self.pool4 = tf.keras.layers.UpSampling2D(size=(2,1))
            self.conv5 = tf.keras.layers.Conv2D(filters=128,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same",activation="relu")
            self.bn5 = tf.keras.layers.BatchNormalization()
            self.pool5 = tf.keras.layers.UpSampling2D(size=(2, 1),)
            self.conv6 = tf.keras.layers.Conv2D(filters=64,
                                                kernel_size=(1, 1),
                                                strides=1,
                                                padding="same",activation="relu")
            self.bn6 = tf.keras.layers.BatchNormalization()
            self.pool6 = tf.keras.layers.UpSampling2D(size=(2, 1),
                                                    )
            self.conv7 = tf.keras.layers.Conv2D(filters=1,
                                                kernel_size=(3, 1),
                                                strides=1
                                                ,activation="sigmoid")
            
        def __call__(self,inputs,training=None):
            x = self.conv1(inputs)
            x = self.bn1(x,training=training)
            x = self.pool1(x)
            x = self.conv2(x,training=training)
            x  = self.bn2(x,training=training)
            x =self.pool2(x)
            x = self.conv3(x,training=training)
            x  = self.bn3(x,training=training)
            x =self.pool3(x)
            x = self.conv4(x,training=training)
            x  = self.bn4(x,training=training)
            x =self.pool4(x)
            x = self.conv5(x,training=training)
            x  = self.bn5(x,training=training)
            x =self.pool5(x)
            x = self.conv6(x,training=training)
            x  = self.bn6(x,training=training)
            x =self.pool6(x)
            x =self.conv7(x,training=training)
            return x
if __name__ == '__main__':        
    model = CAE()

    loss_object = tf.keras.losses.MeanSquaredError()
    optimizer = tf.keras.optimizers.Adam(0.0001)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.Accuracy(name='valid_accuracy')
    face_image = []
    valid_face_image = []
    target =[]
    valid_target = []
    dataname = ["iwata","nisio","tanaka","miymoto"]
    expr = ["negative","postive","normal"]
    validname = ["iwata"]
    valid_data =np.empty([0,956])
    v_targets = np.empty(0)
    datasets = np.empty([0,956])
    targets = np.empty(0)
    zero_train =np.empty([0,956])
    zero_v =np.empty([0,956])
    for name in dataname:
        file =f"newdata/postive-{name}-landmarks.npz"
        if name in validname:
            data =  np.load(file)

            #valid_data = np.concatenate([valid_data,data["arr_0"]],axis=0)
            #v_targets= np.concatenate([v_targets,data["arr_1"]])
            index_z = np.where(data["arr_1"] == 2)
            zero_v=np.concatenate([zero_v,data["arr_0"][index_z[0][0]:index_z[0][-1]]],axis=0)
            continue
        data =  np.load(file)
        #datasets=np.concatenate([datasets,data["arr_0"]],axis=0)
        #targets=np.concatenate([targets,data["arr_1"]])
        index_z = np.where(data["arr_1"] == 2)
        zero_train=np.concatenate([zero_train,data["arr_0"][index_z[0][0]:index_z[0][-1]]],axis=0)
    zero_train = zero_train.reshape((len(zero_train),478,2,1))
    zero_v =zero_v.reshape((len(zero_v),478,2,1))
    train_value = tf.data.Dataset.from_tensor_slices(zero_train)
    train_label = tf.data.Dataset.from_tensor_slices(zero_train)
    train_dataset = tf.data.Dataset.zip((train_value, train_label))
    valid_value = tf.data.Dataset.from_tensor_slices(zero_v)
    valid_label = tf.data.Dataset.from_tensor_slices(zero_v)
    valid_dataset = tf.data.Dataset.zip((valid_value, valid_label))
    train_count = len(zero_train)
    valid_count = len(zero_v)
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)
    @tf.function
    def train_step_image(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def valid_step_image(images, labels):
        predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(labels, predictions)   
    for epoch in range(60):
                train_loss.reset_states()
                train_accuracy.reset_states()
                valid_loss.reset_states()
                valid_accuracy.reset_states()
                step = 0
                for image, labels in train_dataset:
                    step += 1
                    train_step_image(image, labels)
                    print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                            60,
                                                                                            step,
                                                                                            math.ceil(train_count / config.BATCH_SIZE),
                                                                                            train_loss.result(),
                                                                                            train_accuracy.result()))

                for image, valid_labels in valid_dataset:
                    valid_step_image(image, valid_labels)
    model.save_weights(filepath="loss_train/2/model_2", save_format='tf')
