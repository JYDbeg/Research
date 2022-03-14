import tensorflow as tf
from config import NUM_CLASSES
from residual_block import make_basic_block_layer, make_bottleneck_layer
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
