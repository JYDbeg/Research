import keras
import tensorflow as tf

class class_dnn(tf.keras.Model):
    def __init__(self):
        super(class_dnn, self).__init__()
        self.denselayer1 = keras.layers.Dense(2048,"relu")
        self.denselayer2 = keras.layers.Dense(4096,"relu")
        self.denselayer3 = keras.layers.Dense(3072,"relu")
        self.bn2 = keras.layers.BatchNormalization()
        self.denselayer4 = keras.layers.Dense(1536,"relu")
        self.fc = keras.layers.Dense(units=3, activation=tf.keras.activations.softmax)
        
    def call(self,input,training=None, mask=None):
        x = self.denselayer1(input)
        x = self.denselayer2(x,training = training)
        x = self.denselayer3(x,training = training)
        x = self.bn2(x,training = training)
        x = self.denselayer4(x,training = training)
        output = self.fc(x)
        return output
