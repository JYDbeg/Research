import tensorflow as tf
import config
import numpy as np
import time
import matplotlib.pyplot as plt
import math
import glob
from multimodal import multimodalnn
from prepare_data import load_and_preprocess_image
from sklearn.model_selection import train_test_split
if __name__ == '__main__':
    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    names = ["iwata","nisio","tanaka","miymoto"]
    expr = ["negative","postive","normal"]
    #valid_names = ["07","09"]
    valid_names = ["miymoto"]
    valid_data =np.empty([0,956])
    v_targets = np.empty(0)
    datasets = np.empty([0,956])
    targets = np.empty(0)
    face_image = []
    valid_face_image = []
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
    for name in names:
        if name in valid_names:
            for ex in expr:
                for file in glob.glob(f"newdata/faceonly/{name}/{ex}/*.jpg"):
                    valid_face_image.append(load_and_preprocess_image(file,image_height=164,image_width=164,channels=1))
            continue
        for ex in expr:
            for file in glob.glob(f"newdata/faceonly/{name}/{ex}/*.jpg"):
                face_image.append(load_and_preprocess_image(file,image_height=164,image_width=164,channels=1))
        
    datasets = datasets.reshape((len(datasets),478,2,1))
    valid_data =valid_data.reshape((len(valid_data),478,2,1))
    #x_train,x_valid,y_train,y_valid,z_train,z_valid = train_test_split(datasets,targets,face_image,test_size =0.5)
    
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

    # create model
    model = multimodalnn()
    #model.load_weights(filepath = config.save_model_dir)
    # define loss and optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    optimizer = tf.keras.optimizers.SGD(0.0001,momentum=0.9)

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
    resul =[]
    # start training
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
        resul.append(valid_accuracy.result())

    print(resul)
    model.save_weights(filepath=config.save_model_dir, save_format='tf')
