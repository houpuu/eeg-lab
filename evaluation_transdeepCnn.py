import scipy.io as scio
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

import sys
sys.path.append(r'.')
from scipy.io import savemat
import keras
from keras.models import Sequential
from keras.models import save_model
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Conv2D
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling1D
from keras.utils import plot_model
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras import Model
from keras.models import load_model

import matplotlib.pyplot as plt
from scipy import signal

import numpy as np

#def precision(y_true, y_pred):
      #Calculates the precision
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#     precision = true_positives / (predicted_positives + K.epsilon())
#     return precision
#
# def recall(y_true, y_pred):
#     # Calculates the recall
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     recall = true_positives / (possible_positives + K.epsilon())
#     return recall

model_1 = load_model('C:\PycharmProjects\EEG_CNN\model\music_p1.h5')
model_2 = load_model('C:\PycharmProjects\EEG_CNN\model\music_p3_new1.h5')

dataFile = 'C:\EEG\music_feature\code_ver2\dSampleData_cnn_p4.mat'
num_class = 5
data = scio.loadmat(dataFile)
train_input = data['train_input']
train_output = data['train_output']


print('train_input的维度：', train_input.shape)
print('train_output的维度：', train_output.shape)

height, width, length = train_input.shape
# 输入数据归一化
train_input = np.reshape(train_input, (-1, length))
print('train_input的维度：', train_input.shape)
ss = StandardScaler()
scaler = ss.fit(train_input)
print(scaler)
print(scaler.mean_)
scaled_train = scaler.transform(train_input)
print('scaled_train的维度：', scaled_train.shape)
train_input = np.reshape(scaled_train, (-1, width, length))
print('train_input的维度：', train_input.shape)


# 打乱数据
permutation = np.random.permutation(train_input.shape[0])
train_input = train_input[permutation, :, :]
train_output = train_output[permutation]

# 标签one hot化
lb = LabelBinarizer()
# train_output = lb.fit_transform(train_output)  # transfer label to binary value
train_output = to_categorical(train_output)  # transfer binary label to one-hot. IMPORTANT

print('train_input的维度：', train_input.shape)
print('train_output的维度：', train_output.shape)

layer_model = Model(model_1.input, model_1.layers[9].output)
layer_model.summary()
feature1 = layer_model.predict(train_input, batch_size=10)

### 评估模型,输出预测结果
score = model_2.evaluate(feature1, train_output,
                         batch_size=10,
                         verbose=2,
                         sample_weight=None)
print('Test score:', score[0])
print("test accuracy:", score[1])

'''
test_output = model_2.predict(feature1, batch_size=1)
train_output = np.argmax(test_output, axis=1)
print('y_pre:', train_output)
print('train_input的维度：', train_input.shape)
print('train_output的维度：', train_output.shape)

#file_name = 'C:\EEG\music_feature\code_ver2\dSampleData_cnn_p2_new.mat'
#savemat(file_name, {'train_input': train_input, 'train_output': train_output})
'''