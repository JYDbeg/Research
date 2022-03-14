from __future__ import absolute_import, division, print_function
import tensorflow as tf
from config import NUM_CLASSES
from models.residual_block import make_basic_block_layer, make_bottleneck_layer
from models.resnet import resnet_18, resnet_34, resnet_50, resnet_101, resnet_152
import config
from prepare_data import generate_datasets
import math

class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params,firstLayerKernelsize=[2,1],poollayer=[1,1]):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(filters=32,
                                            kernel_size=(firstLayerKernelsize[0],firstLayerKernelsize[1]),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(poollayer[0],poollayer[1]),
                                               strides=2,
                                               padding="same")

        self.layer1 = make_basic_block_layer(filter_num=32,
                                             blocks=layer_params[0],firstLayerKernelsize=1)
        self.layer2 = make_basic_block_layer(filter_num=64,
                                             blocks=layer_params[1],
                                             stride=2,firstLayerKernelsize=1)
        self.layer3 = make_basic_block_layer(filter_num=128,
                                             blocks=layer_params[2],
                                             stride=2,firstLayerKernelsize=1)
        self.layer4 = make_basic_block_layer(filter_num=256,
                                             blocks=layer_params[3],
                                             stride=2,firstLayerKernelsize=1)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)


    def call(self, inputs, training=None, mask=None,ifTest=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)
        return output
def get_model():
    model = resnet_50()
    if config.model == "resnet18":
        model = resnet_18()
    if config.model == "resnet34":
        model = resnet_34()
    if config.model == "resnet101":
        model = resnet_101()
    if config.model == "resnet152":
        model = resnet_152()
    model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    model.summary()
    return model
from sklearn.model_selection import train_test_split
import numpy as np
if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)


    # get the original_dataset
    #train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()
    names  =["01","02","03","04","05","06","07","08","09","10","11","12","13","17","19","20","22","23"]
    names = ["iwata","nisio","tanaka","miymoto"]
    valid_names = ["miymoto"]
    valid_data =np.empty([0,956])
    v_targets = np.empty(0)
    datasets = np.empty([0,956])
    targets = np.empty(0)
    for name in names:
        file =f"newdata/postive-{name}-landmarks.npz"#f"Actor_video/{name}-landmarks.npz"
        if name in valid_names:
            data =  np.load(file)
            valid_data = np.concatenate([valid_data,data["arr_0"]],axis=0)
            v_targets= np.concatenate([v_targets,data["arr_1"]])
            continue
        data =  np.load(file)
        datasets=np.concatenate([datasets,data["arr_0"]],axis=0)
        targets=np.concatenate([targets,data["arr_1"]])
    datasets = datasets.reshape((len(datasets),478,2,1))
    valid_data =valid_data.reshape((len(valid_data),478,2,1))
    #x_train,x_valid,y_train,y_valid = train_test_split(datasets,targets,test_size =0.9)
    
    train_value = tf.data.Dataset.from_tensor_slices(datasets)
    train_label = tf.data.Dataset.from_tensor_slices(targets)
    train_dataset = tf.data.Dataset.zip((train_value, train_label))
    valid_value = tf.data.Dataset.from_tensor_slices(valid_data)
    valid_label = tf.data.Dataset.from_tensor_slices(v_targets)
    valid_dataset = tf.data.Dataset.zip((valid_value, valid_label))
    train_count = len(datasets)
    valid_count = len(valid_data)
    train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
    valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)

    # create model
    model = ResNetTypeI(layer_params=[2, 2, 2, 2],firstLayerKernelsize=[2,1],poollayer=[1,1])#get_model()
    #model.load_weights(filepath = config.save_model_dir)
    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(0.0001,momentum=0.9)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

    valid_loss = tf.keras.metrics.Mean(name='valid_loss')
    valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(images, labels):
        with tf.GradientTape() as tape:
            predictions = model(images, training=True)
            loss = loss_object(y_true=labels, y_pred=predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

        train_loss(loss)
        train_accuracy(labels, predictions)

    @tf.function
    def valid_step(images, labels):
        predictions = model(images, training=False)
        v_loss = loss_object(labels, predictions)

        valid_loss(v_loss)
        valid_accuracy(labels, predictions)
    resul =[]
    # start training
    for epoch in range(config.EPOCHS):
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0
        for images, labels in train_dataset:
            step += 1
            train_step(images, labels)
            print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                     config.EPOCHS,
                                                                                     step,
                                                                                     math.ceil(train_count / config.BATCH_SIZE),
                                                                                     train_loss.result(),
                                                                                     train_accuracy.result()))

        for valid_images, valid_labels in valid_dataset:
            valid_step(valid_images, valid_labels)

        print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
              "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                  config.EPOCHS,
                                                                  train_loss.result(),
                                                                  train_accuracy.result(),
                                                                  valid_loss.result(),
                                                                  valid_accuracy.result()))
        resul.append(valid_accuracy.result())

    model.save_weights(filepath=config.save_model_dir, save_format='tf')
    print(resul)