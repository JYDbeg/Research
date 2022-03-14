import tensorflow as tf
import config
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
from keras.layers import Input,Dense,BatchNormalization,Conv2D,SeparableConv2D,MaxPool2D,GlobalAveragePooling2D,add
from keras.layers.core import Activation
from keras.models import Model
def Xception():
        inputs = Input(shape=(1291, 6, 1))

        # entry flow

        x = Conv2D(32, (3,3), strides=2, padding='same')(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        residual = Conv2D(128, (1,1), strides=2, padding='same')(x)
        residual = BatchNormalization()(residual)

        x = SeparableConv2D(128, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(128, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((3, 3), strides=2, padding='same')(x)
        x = add([x, residual])

        residual = Conv2D(256, (1,1), strides=2, padding='same')(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu')(x)
        x = SeparableConv2D(256, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(256, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((3, 3), strides=2, padding='same')(x)
        x = add([x, residual])

        residual = Conv2D(728, (1,1), strides=2, padding='same')(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((3, 3), strides=2, padding='same')(x)
        x = add([x, residual])
        for i in range(8):
            residual = x

            x = Activation('relu')(x)
            x = SeparableConv2D(728, (3,3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(728, (3,3), padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = SeparableConv2D(728, (3,3), padding='same')(x)
            x = BatchNormalization()(x)
            x = add([x, residual])
        residual = Conv2D(1024, (1,1), strides=2, padding='same')(x)
        residual = BatchNormalization()(residual)

        x = Activation('relu')(x)
        x = SeparableConv2D(728, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(1024, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D((3, 3), strides=2, padding='same')(x)

        x = add([x, residual])

        x = SeparableConv2D(1536, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = SeparableConv2D(2048, (3,3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = GlobalAveragePooling2D()(x)
        x = Dense(10, kernel_initializer='he_normal', activation='softmax')(x)

        model = Model(inputs=inputs, outputs=x)
        return model
