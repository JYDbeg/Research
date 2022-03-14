
import tensorflow as tf
import numpy as np
from models.residual_block import make_basic_block_layer, make_bottleneck_layer
import keras
from CAE_train import CAE
class landmark_only_resnet_50(tf.keras.Model):
    def __init__(self,layer_params=[3,4,6,3]):
        super(landmark_only_resnet_50, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters=256,
                                            kernel_size=(1, 2),
                                            strides=(1,1),
                                            )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 1),
                                                strides=(2,1),
                                                )
        self.layer1 = make_basic_block_layer(filter_num=512,
                                             blocks=3,
                                             stride=2,firstLayerKernelsize=1)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=3, activation=tf.keras.activations.softmax)
    def call(self, inputs,training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)

        x = self.avgpool(x)
        output = self.fc(x)

        return output
model0 =CAE()
model1 = CAE()
model2 = CAE()
model3 = landmark_only_resnet_50()
model0.load_weights(filepath="loss_train/0/model_0")
model1.load_weights(filepath="loss_train/1/model_1")
model2.load_weights(filepath="loss_train/2/model_2")
model3.load_weights(filepath="loss_train/target/model_3")
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.000001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')
@tf.function
def train_step_image(images, labels):
    with tf.GradientTape() as tape:
        predictions = model3(images, training=True)
        loss = loss_object(y_true=labels, y_pred=predictions)
    gradients = tape.gradient(loss, model3.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, model3.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def valid_step_image(images, labels):
    predictions = model3(images, training=False)
    v_loss = loss_object(labels, predictions)

    valid_loss(v_loss)
    valid_accuracy(labels, predictions)   
file ="newdata/postive-iwata-landmarks.npz"
data = np.load(file)
test_data = data["arr_0"]
test_target = data["arr_1"]
count = 0
indexs=[]
from tqdm import tqdm
for i in tqdm(range(len(test_data))):
    losss = []
    target = test_target[i]
    data_ = test_data[i].reshape((1,478,2,1))
    zero_ = model0(data_)
    one_ = model1(data_)
    two_=model2(data_)
    for j in range(3):
        model3.load_weights(filepath="loss_train/target/model_3")
        if j ==0:
            train_value = tf.data.Dataset.from_tensor_slices(zero_)
            train_label = tf.data.Dataset.from_tensor_slices([0])
            train_dataset = tf.data.Dataset.zip((train_value, train_label))
            train_dataset = train_dataset.shuffle(buffer_size=1).batch(batch_size=1)
        if j ==1:
            train_value = tf.data.Dataset.from_tensor_slices(one_)
            train_label = tf.data.Dataset.from_tensor_slices([1])
            train_dataset = tf.data.Dataset.zip((train_value, train_label))
            train_dataset = train_dataset.shuffle(buffer_size=1).batch(batch_size=1)
        if j ==2:
            train_value = tf.data.Dataset.from_tensor_slices(two_)
            train_label = tf.data.Dataset.from_tensor_slices([2])
            train_dataset = tf.data.Dataset.zip((train_value, train_label))
            train_dataset = train_dataset.shuffle(buffer_size=1).batch(batch_size=1)
        for image, labels in train_dataset:
            train_loss.reset_states()
            train_accuracy.reset_states()
            train_step_image(image, labels)
            losss.append(train_loss.result().numpy())
    losss = np.array(losss)
    indexs.append(np.argmin(losss))
    if np.argmin(losss) == target:
        count+=1
acc = count/len(data["arr_0"])
print(acc,indexs)

