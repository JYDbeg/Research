import tensorflow as tf
import config
from prepare_data import generate_datasets
from train import get_model
import time
from nnnnnn import class_dnn
import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':

    # GPU settings
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    # get the original_dataset
    train_dataset, valid_dataset, test_dataset, train_count, valid_count, test_count = generate_datasets()

    datasets = np.empty([0,956])
    targets = np.empty(0)
    file ="0829/landmarks.npz"
    data =  np.load(file)
    datasets=np.concatenate([datasets,data["arr_0"]],axis=0)
    targets=np.concatenate([targets,data["arr_1"]])
    datasets = datasets.reshape((len(datasets),478,2,1))
    #valid_data =valid_data.reshape((len(valid_data),478,2,1))


    test_value = tf.data.Dataset.from_tensor_slices(datasets)
    test_label = tf.data.Dataset.from_tensor_slices(targets)
    test_dataset = tf.data.Dataset.zip((test_value, test_label))

    test_count = len(datasets)

    test_dataset = test_dataset.batch(batch_size=config.BATCH_SIZE)
    # print(train_dataset)
    # load the model
    model = get_model()
    model.load_weights(filepath=config.save_model_dir)
    # Get the accuracy on the test set
    loss_object = tf.keras.metrics.SparseCategoricalCrossentropy()
    test_loss = tf.keras.metrics.Mean()
    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    @tf.function
    def test_step(images, labels):
        predictions = model(images, training=False)
        t_loss = loss_object(labels, predictions)
        test_loss(t_loss)
        test_accuracy(labels, predictions)
    for test_images, test_labels in test_dataset:
        predictions = model(test_images, training=False)
  

        t_loss = loss_object(test_labels, predictions)
        test_loss(t_loss)
        test_accuracy(test_labels, predictions)
        #test_step(test_images, test_labels)
        print("loss: {:.5f}, test accuracy: {:.5f}".format(test_loss.result(),
                                                           test_accuracy.result()))

    print("The accuracy on test set is: {:.3f}%".format(test_accuracy.result()*100))