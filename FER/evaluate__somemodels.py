import tensorflow as tf
import config
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import glob
from multimodal import multimodalnn
from models.residual_block import make_basic_block_layer, make_bottleneck_layer
from prepare_data import load_and_preprocess_image
from sklearn.model_selection import train_test_split
from config import NUM_CLASSES
names = ["iwata","nisio","tanaka","miymoto"]
expr = ["negative","postive","normal"]
nnames  =["01","02","03","04","05","06","07","08","09","10","11","12","13","17","19","20","22","23"]

mode={"face_only":0,"landmark_only":1,"landmarks_transsion":2,"complex":3}
def dataprepare(dataname,validname,expr,mode=0):
    if mode == 0:
        face_image = []
        valid_face_image = []
        target =[]
        valid_target = []
        for name in dataname:
            if name in validname:
                for ex in expr:
                    for file in glob.glob(f"newdata/faceonly/{name}/{ex}/*.jpg"):
                        valid_face_image.append(load_and_preprocess_image(file,image_height=164,image_width=164,channels=1))
                        valid_target.append(expr.index(ex))
                        
                continue
            for ex in expr:
                for file in glob.glob(f"newdata/faceonly/{name}/{ex}/*.jpg"):
                    face_image.append(load_and_preprocess_image(file,image_height=164,image_width=164,channels=1))
                    target.append(expr.index(ex))
        train_value = tf.data.Dataset.from_tensor_slices(face_image)
        train_label = tf.data.Dataset.from_tensor_slices(target)
        train_dataset = tf.data.Dataset.zip((train_value, train_label))
        valid_value = tf.data.Dataset.from_tensor_slices(valid_face_image)
        valid_label = tf.data.Dataset.from_tensor_slices(valid_target)
        valid_dataset = tf.data.Dataset.zip((valid_value, valid_label))
        train_count = len(face_image)
        valid_count = len(valid_face_image)
        train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
        valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)
        return train_dataset,valid_dataset,train_count,valid_count
    if mode == 1:
        face_image = []
        valid_face_image = []
        target =[]
        valid_target = []
        for name in dataname:
            if name in validname:
                for ex in expr:
                    for file in glob.glob(f"newdata/landmark/{name}/{ex}/*.jpg"):
                        valid_face_image.append(load_and_preprocess_image(file,image_height=164,image_width=164,channels=1))
                        valid_target.append(expr.index(ex))
                continue
            for ex in expr:
                for file in glob.glob(f"newdata/landmark/{name}/{ex}/*.jpg"):
                    face_image.append(load_and_preprocess_image(file,image_height=164,image_width=164,channels=1))
                    target.append(expr.index(ex))
        train_value = tf.data.Dataset.from_tensor_slices(face_image)
        train_label = tf.data.Dataset.from_tensor_slices(target)
        train_dataset = tf.data.Dataset.zip((train_value, train_label))
        valid_value = tf.data.Dataset.from_tensor_slices(valid_face_image)
        valid_label = tf.data.Dataset.from_tensor_slices(valid_target)
        valid_dataset = tf.data.Dataset.zip((valid_value, valid_label))
        train_count = len(face_image)
        valid_count = len(valid_face_image)
        train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
        valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)
        return train_dataset,valid_dataset,train_count,valid_count
    if mode == 2:
        #arr1 = np.random.uniform(low =0.0 , high =1.0,size = (7746,10000))
        #arr2 = np.random.randn(512,2500)
        valid_data =np.empty([0,956])
        v_targets = np.empty(0)
        datasets = np.empty([0,956])
        targets = np.empty(0)
        for name in dataname:
            file =f"newdata/postive-{name}-landmarks.npz"
            if name in validname:
                data =  np.load(file)
                valid_data = np.concatenate([valid_data,data["arr_0"]],axis=0)
                v_targets= np.concatenate([v_targets,data["arr_1"]])
                continue
            data =  np.load(file)
            datasets=np.concatenate([datasets,data["arr_0"]],axis=0)
            targets=np.concatenate([targets,data["arr_1"]])
        #datasets = np.dot(datasets,arr1)
        #valid_data = np.dot(valid_data,arr1)
        '''for name  in nnames:
            file =f"Actor_video/{name}-landmarks.npz"
            data =  np.load(file)
            datasets=np.concatenate([datasets,data["arr_0"]],axis=0)
            targets=np.concatenate([targets,data["arr_1"]])
        
        file ="newdata/postive-kai-landmarks.npz"
        data =  np.load(file)
        datasets=np.concatenate([datasets,data["arr_0"]],axis=0)
        targets=np.concatenate([targets,data["arr_1"]])'''
        datasets = datasets.reshape((len(datasets),478,2,1))
        valid_data =valid_data.reshape((len(valid_data),478,2,1))
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
        return train_dataset, valid_dataset,train_count,valid_count
    if mode == 3:
        valid_data =np.empty([0,956])
        v_targets = np.empty(0)
        datasets = np.empty([0,956])
        targets = np.empty(0)
        face_image = []
        valid_face_image = []
        for name in dataname:
            file =f"newdata/postive-{name}-landmarks.npz"#f"Actor_video/{name}-landmarks.npz"
            if name in validname:
                data =  np.load(file)
                valid_data = np.concatenate([valid_data,data["arr_0"]/np.max(data["arr_0"],axis=1)],axis=0)
                v_targets= np.concatenate([v_targets,data["arr_1"]])
                continue
            data =  np.load(file)
            datasets=np.concatenate([datasets,data["arr_0"]/np.max(data["arr_0"],axis=1)],axis=0)
            targets=np.concatenate([targets,data["arr_1"]])
        datasets = datasets.reshape((len(datasets),478,2,1))
        valid_data =valid_data.reshape((len(valid_data),478,2,1))
        for name in dataname:
            if name in validname:
                for ex in expr:
                    for file in glob.glob(f"newdata/faceonly/{name}/{ex}/*.jpg"):
                        valid_face_image.append(load_and_preprocess_image(file,image_height=164,image_width=164,channels=1))
                continue
            for ex in expr:
                for file in glob.glob(f"newdata/faceonly/{name}/{ex}/*.jpg"):
                    face_image.append(load_and_preprocess_image(file,image_height=164,image_width=164,channels=1))
        train_value = tf.data.Dataset.from_tensor_slices(datasets)
        train_label = tf.data.Dataset.from_tensor_slices(targets)
        face_train = tf.data.Dataset.from_tensor_slices(face_image)
        train_dataset = tf.data.Dataset.zip((train_value, train_label,face_train))
        valid_value = tf.data.Dataset.from_tensor_slices(valid_data)
        valid_label = tf.data.Dataset.from_tensor_slices(v_targets)
        face_valid = tf.data.Dataset.from_tensor_slices(valid_face_image)
        valid_dataset = tf.data.Dataset.zip((valid_value, valid_label,face_valid))
        train_count = len(datasets)
        valid_count = len(valid_data)
        train_dataset = train_dataset.shuffle(buffer_size=train_count).batch(batch_size=config.BATCH_SIZE)
        valid_dataset = valid_dataset.batch(batch_size=config.BATCH_SIZE)
        return train_dataset,valid_dataset,train_count,valid_count

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
        #self.layer1 = make_basic_block_layer(filter_num=512,blocks = 2,firstLayerKernelsize=1)
        '''self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0],firstLayerKernelsize=1)
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2,firstLayerKernelsize=1)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2,firstLayerKernelsize=1)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2,firstLayerKernelsize=1)'''

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)
    def call(self, inputs,training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        #x = self.layer2(x, training=training)
       # x = self.layer3(x, training=training)
        #x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output
class face_only_resnet_50(tf.keras.Model):
    def __init__(self,layer_params=[3,4,6,3]):
        super(face_only_resnet_50, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(5, 5),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(3, 3),
                                                strides=2,
                                                padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)
    def call(self, inputs,training=None, mask=None):
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
class l_only_resnet_50(tf.keras.Model):
    def __init__(self,layer_params=[3,4,6,3]):
        super(l_only_resnet_50, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters=64,
                                            kernel_size=(3, 1),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 1),
                                                strides=2,
                                                padding="same")

        self.layer1 = make_bottleneck_layer(filter_num=64,
                                            blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(filter_num=128,
                                            blocks=layer_params[1],
                                            stride=2)
        self.layer3 = make_bottleneck_layer(filter_num=256,
                                            blocks=layer_params[2],
                                            stride=2)
        self.layer4 = make_bottleneck_layer(filter_num=512,
                                            blocks=layer_params[3],
                                            stride=2)

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)
    def call(self, inputs,training=None, mask=None):
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
    
model_face_only = face_only_resnet_50()
model_landmark_only = l_only_resnet_50()
model_landmark_transision = landmark_only_resnet_50()
model_landmark_plus_face = multimodalnn()


loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(0.000001)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

valid_loss = tf.keras.metrics.Mean(name='valid_loss')
valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

@tf.function
def train_step(landmarks, labels,images):
    with tf.GradientTape() as tape:
        predictions = model(images,landmarks, training=True)
        loss = loss_object(y_true=labels, y_pred=predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(grads_and_vars=zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def valid_step(landmarks, labels,images):
    predictions = model(images,landmarks, training=False)
    v_loss = loss_object(labels, predictions)

    valid_loss(v_loss)
    valid_accuracy(labels, predictions)
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

valid_names = ["iwata"]
result_train_accuracy = []
result_all_accuracy = []
modes = [2]
for mode in modes:
    if mode == 0:
        model = model_face_only
        train_dataset,valid_dataset,train_count,valid_count = dataprepare(names,valid_names,expr,mode)
        for epoch in range(config.EPOCHS):
            train_loss.reset_states()
            train_accuracy.reset_states()
            valid_loss.reset_states()
            valid_accuracy.reset_states()
            step = 0
            for image, labels in train_dataset:
                step += 1
                train_step_image(image, labels)
                print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                        config.EPOCHS,
                                                                                        step,
                                                                                        math.ceil(train_count / config.BATCH_SIZE),
                                                                                        train_loss.result(),
                                                                                        train_accuracy.result()))

            for image, valid_labels in valid_dataset:
                valid_step_image(image, valid_labels)

            print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                    config.EPOCHS,
                                                                    train_loss.result(),
                                                                    train_accuracy.result(),
                                                                    valid_loss.result(),
                                                                    valid_accuracy.result()))
            result_all_accuracy.append(valid_accuracy.result().numpy())
            result_train_accuracy.append(train_accuracy.result().numpy())
        fig = plt.figure()
        plt.plot(range(config.EPOCHS), result_train_accuracy,label="train_accuracy")
        plt.plot(range(config.EPOCHS), result_all_accuracy,label="valid_accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")

        plt.legend()
        fig.savefig(f"{valid_names}{mode}res18.png")
        print(result_all_accuracy)
        f = open('test.txt', 'a', encoding='UTF-8')

        f.write(str(np.max(result_all_accuracy))+"\n")
        f.close()

    if mode == 1:
        model = model_landmark_only
        train_dataset,valid_dataset,train_count,valid_count = dataprepare(names,valid_names,expr,mode)
        for epoch in range(config.EPOCHS):
            train_loss.reset_states()
            train_accuracy.reset_states()
            valid_loss.reset_states()
            valid_accuracy.reset_states()
            step = 0
            for image, labels in train_dataset:
                step += 1
                train_step_image(image, labels)
                print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                        config.EPOCHS,
                                                                                        step,
                                                                                        math.ceil(train_count / config.BATCH_SIZE),
                                                                                        train_loss.result(),
                                                                                        train_accuracy.result()))

            for image, valid_labels in valid_dataset:
                valid_step_image(image, valid_labels)

            print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                    config.EPOCHS,
                                                                    train_loss.result(),
                                                                    train_accuracy.result(),
                                                                    valid_loss.result(),
                                                                    valid_accuracy.result()))
            result_all_accuracy.append(valid_accuracy.result().numpy())
            result_train_accuracy.append(train_accuracy.result().numpy())
        fig = plt.figure()
        plt.plot(range(config.EPOCHS), result_train_accuracy,label="train_accuracy")
        plt.plot(range(config.EPOCHS), result_all_accuracy,label="valid_accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")

        plt.legend()
        fig.savefig(f"{valid_names}{mode}res18.png")
        print(result_all_accuracy)
        f = open('test.txt', 'a', encoding='UTF-8')

        f.write(str(np.max(result_all_accuracy))+"\n")
        f.close()
    if mode == 2:
        model = model_landmark_transision
        train_dataset,valid_dataset,train_count,valid_count = dataprepare(names,valid_names,expr,mode)
        for epoch in range(50):
            train_loss.reset_states()
            train_accuracy.reset_states()
            valid_loss.reset_states()
            valid_accuracy.reset_states()
            step = 0
            for image, labels in train_dataset:
                step += 1
                train_step_image(image, labels)
                print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                        50,
                                                                                        step,
                                                                                        math.ceil(train_count / config.BATCH_SIZE),
                                                                                        train_loss.result(),
                                                                                        train_accuracy.result()))

            for image, valid_labels in valid_dataset:
                valid_step_image(image, valid_labels)

            print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                    config.EPOCHS,
                                                                    train_loss.result(),
                                                                    train_accuracy.result(),
                                                                    valid_loss.result(),
                                                                    valid_accuracy.result()))
            result_all_accuracy.append(valid_accuracy.result().numpy())
            result_train_accuracy.append(train_accuracy.result().numpy())
        model.save_weights(filepath="loss_train/target/model_3", save_format='tf')
    if mode ==3:
        
        model = model_landmark_plus_face
        train_dataset,valid_dataset,train_count,valid_count = dataprepare(names,valid_names,expr,mode)
        for epoch in range(config.EPOCHS):
            train_loss.reset_states()
            train_accuracy.reset_states()
            valid_loss.reset_states()
            valid_accuracy.reset_states()
            step = 0
            for landmarks, labels,face_image in train_dataset:
                step += 1
                train_step(landmarks, labels,face_image)
                print("Epoch: {}/{}, step: {}/{}, loss: {:.5f}, accuracy: {:.5f}".format(epoch + 1,
                                                                                        config.EPOCHS,
                                                                                        step,
                                                                                        math.ceil(train_count / config.BATCH_SIZE),
                                                                                        train_loss.result(),
                                                                                        train_accuracy.result()))

            for landmarks, valid_labels,face_valid in valid_dataset:
                valid_step(landmarks, valid_labels, face_valid)

            print("Epoch: {}/{}, train loss: {:.5f}, train accuracy: {:.5f}, "
                "valid loss: {:.5f}, valid accuracy: {:.5f}".format(epoch + 1,
                                                                    config.EPOCHS,
                                                                    train_loss.result(),
                                                                    train_accuracy.result(),
                                                                    valid_loss.result(),
                                                                    valid_accuracy.result()))
            result_all_accuracy.append(valid_accuracy.result().numpy())
            result_train_accuracy.append(train_accuracy.result().numpy())
        #fig = plt.figure()
        plt.plot(range(config.EPOCHS), result_train_accuracy,label="train_accuracy")
        plt.plot(range(config.EPOCHS), result_all_accuracy,label="valid_accuracy")
        plt.xlabel("epoch")
        plt.ylabel("accuracy")

        plt.legend()
        fig.savefig(f"{valid_names}{mode}res18.png")
        plt.show()
        print(result_all_accuracy)
        f = open('test.txt', 'a', encoding='UTF-8')

        f.write(str(np.max(result_all_accuracy))+"\n"+f"{valid_names}")
        f.close()
    if mode == 4:
        from exception import Xception
        model = Xception()
        opt = tf.keras.optimizers.Adam(learning_rate=0.000001)
        model.compile(optimizer=opt,loss ="sparse_categorical_crossentropy" ,metrics=['accuracy'])
        batch_size = 32
        epochs = 50
        valid_data =np.empty([0,7746])
        v_targets = np.empty(0)
        datasets = np.empty([0,7746])
        targets = np.empty(0)
        for name in names:
            file =f"newdata/postive-{name}-distance-landmarks.npz"
            if name in valid_names:
                data =  np.load(file)
                valid_data = np.concatenate([valid_data,data["arr_0"]/1.5],axis=0)
                v_targets= np.concatenate([v_targets,data["arr_1"]])
                continue
            data =  np.load(file)
            datasets=np.concatenate([datasets,data["arr_0"]/1.5],axis=0)
            targets=np.concatenate([targets,data["arr_1"]])
        datasets = datasets.reshape((len(datasets),1291,6,1))
        valid_data =valid_data.reshape((len(valid_data),1291,6,1))
        h = model.fit(datasets,targets,batch_size=batch_size,epochs=epochs,validation_data=(valid_data,v_targets))
        plt.plot(h.history['acc'])
        plt.plot(h.history['val_acc'])
        plt.xlabel('Epoch')
        plt.ylabel("Accuracy")
        plt.legend(["Train","Test"])
        plt.show()
