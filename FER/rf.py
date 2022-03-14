
import tensorflow as tf
import config
import numpy as np
import matplotlib.pyplot as plt
import math
import glob
from models.residual_block import make_basic_block_layer, make_bottleneck_layer
from config import NUM_CLASSES

names = ["nisio","tanaka","miymoto"]
valid_names = ["tanaka"]
valid_data =np.empty([0,956])
v_targets = np.empty(0)
datasets = np.empty([0,956])
targets = np.empty(0)
#595 1522 2001
class landmark_only_resnet_50(tf.keras.Model):
    def __init__(self,layer_params=[3,4,6,3]):
        super(landmark_only_resnet_50, self).__init__()
        
        self.conv1 = tf.keras.layers.Conv2D(filters=256,
                                            kernel_size=(2, 1),
                                            strides=2,
                                            padding="same")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(1, 1),
                                                strides=2,
                                                padding="same")
        self.layer1 = make_basic_block_layer(filter_num=512,
                                             blocks=3,
                                             stride=2,firstLayerKernelsize=1)
 
        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(units=NUM_CLASSES, activation=tf.keras.activations.softmax)
    def call(self, inputs,training=None, mask=None,hidden=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.avgpool(x)
        if hidden == True:
            return x
        output = self.fc(x)
        return output
#model = landmark_only_resnet_50()
#model.load_weights(filepath=config.save_model_dir)
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
file_f =f"newdata/postive-tanaka-distance-landmarks.npz"
file_s =f"newdata/postive-iwata-distance-landmarks.npz"
data_f = np.load(file_f)
data_s = np.load(file_s)
d_f  = data_f["arr_0"]
d_s = data_s["arr_0"]
t_f =data_f["arr_1"]
t_s = data_s["arr_1"]
a_f = d_f[35]
b_f = d_f[np.where(t_f ==1)[0][1]]
c_f = d_f[np.where(t_f == 2)[0][6]]
a_s = d_s[36]
b_s = d_s[np.where(t_s ==1)[0][1]]
c_s = d_s[np.where(t_s == 2)[0][1]]
print(cos_sim(a_f,a_s))
print(cos_sim(b_f,b_s))
print(cos_sim(c_f,c_s))
print(cos_sim(a_f,b_s))
print(cos_sim(b_f,c_s))
print(cos_sim(c_f,a_s))
print(cos_sim(a_f,c_s))
print(cos_sim(b_f,a_s))
print(cos_sim(c_f,b_s))
print(cos_sim(a_f,b_f))
print(cos_sim(a_f,c_f))
#0=2 1=2 2=2
'''for name in names:
    file =f"newdata/postive-{name}-landmarks.npz"
    data =  np.load(file)
    datasets=np.concatenate([datasets,data["arr_0"]],axis=0)
    targets=np.concatenate([targets,data["arr_1"]])
datasets = datasets.reshape((len(datasets),478,2,1))
train_value = tf.data.Dataset.from_tensor_slices(datasets)
train_label = tf.data.Dataset.from_tensor_slices(targets)
train_value = train_value.batch(batch_size=1)
datasd =[]
l = len(train_value)
count = 0
for image in train_value:
    datasd.append(model(image,hidden=True)[0])
    count +=1 
    print(count/l)

file = f"newdata/postive-iwata-landmarks.npz"
data =  np.load(file)
x = data["arr_0"].reshape((len(data["arr_0"]),478,2,1))
y = data["arr_1"]
test = []
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score
C = 1.
kernel = 'rbf'
gamma  = 0.01
l = len(x)
count = 0
for i in x :
    test.append(model(i.reshape((1,478,2,1)),hidden=True)[0])
    count+=1
    print(count/l)
estimator = SVC(C=C, kernel=kernel, gamma=gamma)
classifier = OneVsRestClassifier(estimator)
classifier.fit(datasd, targets)
pred_y = classifier.predict(test)

classifier2 = SVC(C=C, kernel=kernel, gamma=gamma)
classifier2.fit(datasd, targets)
pred_y2 = classifier2.predict(test)

print('One-versus-the-rest: {:.5f}'.format(accuracy_score(y, pred_y)))
print('One-versus-one: {:.5f}'.format(accuracy_score(y, pred_y2)))'''