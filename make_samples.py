import scipy.io as scio
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import sys
sys.path.append(r'.')
from tkinter import filedialog
from tkinter import *
import tkinter as tk
import tkinter.messagebox
from numpy import *

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



window = tk.Tk()
window.title('Welcome to the EEG World')
window.geometry('1200x500')

info_text = tk.Text(window, width=80, height=15)
info_text.pack()
num_class = 5

def open_file():
    global train_input

    file_path = filedialog.askopenfilename(filetypes=[("MAT", ".mat"),("EDF", ".edf")])
    data = scio.loadmat(file_path)
    train_input = data['train_input']
    train_output = data['train_output']

    height, width, length = train_input.shape

    info_text.insert('insert', file_path, train_input.shape)
    info_text.insert('insert', '\n')

def save_file():

    global train_input
    global train_output

    new_file = filedialog.asksaveasfilename()
    scio.savemat(new_file, {'train_input': train_input, 'train_output': train_output})
    info_text.insert('insert', new_file)
    info_text.insert('insert', train_output)
    info_text.insert('insert', '\n')
    info_text.insert('insert', '\n')

def input_coeff():

    global train_input
    global train_output
    global new_width
    global new_label
    global scaled_train
    global length
    global w1


    new_width = int(new_width.get())
    train_input = np.reshape(scaled_train, (-1, new_width, length))
    new_label = int(new_label.get())

    height, width, length = train_input.shape

    hang = int(height)

    train_output = np.ones((hang, 1), dtype=int) * new_label

    info_text.insert('insert', train_input.shape)
    info_text.insert('insert', train_output.shape)
    info_text.insert('insert', '\n')


def single_sample():

    global train_input
    global new_width
    global new_label
    global scaled_train
    global length
    global w1

    height, width, length = train_input.shape
    # 输入数据归一化
    train_input = np.reshape(train_input, (-1, length))
    ss = StandardScaler()
    scaler = ss.fit(train_input)
    scaled_train = scaler.transform(train_input)

    w1 = tk.Tk()
    w1.title('Input the EEG coeffition')
    w1.geometry('400x200')

    tk.Label(w1, text="Input the width of sample:").place(x=20, y=50)
    tk.Label(w1, text="Input the label of sample:").place(x=20, y=90)

    var1 = tk.StringVar()
    var1.set("250")
    new_width = tk.Entry(w1, textvariable=var1)
    new_width.place(x=200, y=50)

    var2 = tk.StringVar()
    new_label = tk.Entry(w1, textvariable=var2)
    new_label.place(x=200, y=90)

    btn_ok = tk.Button(w1, text="OK", command=input_coeff)
    btn_ok.place(x=230, y=130)
    btn_close = tk.Button(w1, text="Close", command=w1.destroy)
    btn_close.place(x=300, y=130)

def add_file():
    global train_input
    global train_output

    file_path = filedialog.askopenfilename(filetypes=[("MAT", ".mat")])

    data = scio.loadmat(file_path)
    x_input = data['train_input']
    x_output = data['train_output']

    height, width, length = train_input.shape
    train_input = np.reshape(train_input, (-1, length))
    height, width, length = x_input.shape
    x_input = np.reshape(x_input, (-1, length))

    print(x_output.shape)

    train_input = vstack((train_input, x_input))
    train_output = vstack((train_output, x_output))

    train_input = np.reshape(train_input, (-1, width, length))

    info_text.insert('insert', file_path)
    info_text.insert('insert', '\n')


def endadd_file():
    global train_input
    global train_output

    new_file = filedialog.asksaveasfilename()
    scio.savemat(new_file, {'train_input': train_input, 'train_output': train_output})
    info_text.insert('insert', train_input.shape)
    info_text.insert('insert', '\n')
    info_text.insert('insert', train_output.shape)
    info_text.insert('insert', '\n')
    info_text.insert('insert', '\n')

def combine_sample():

    global train_input
    global train_output

    file_path = filedialog.askopenfilename(filetypes=[("MAT", ".mat"), ("EDF", ".edf")])

    data = scio.loadmat(file_path)
    train_input = data['train_input']
    train_output = data['train_output']

    info_text.insert('insert', file_path)
    info_text.insert('insert', '\n')

    com_B1 = tk.Button(window, text="ADD MORE", command=add_file)
    com_B1.place(x=30, y=100)

    com_B2 = tk.Button(window, text="NO MORE", command=endadd_file)
    com_B2.place(x=30, y=150)

def save_h5():

    global model_m
    global h5_model

    h5_model = h5_model.get()
    # model_m.save('C:\PycharmProjects\eeg\data\music\music_model_20210706.h5')
    model_m.save(h5_model)

def deepcnn_go():
    global model_m
    global h5_model

    file_path = filedialog.askopenfilename(filetypes=[("MAT", ".mat"), ("EDF", ".edf")])

    data = scio.loadmat(file_path)
    train_input = data['train_input']
    train_output = data['train_output']



    height, width, length = train_input.shape
    # 打乱数据
    permutation = np.random.permutation(train_input.shape[0])
    train_input = train_input[permutation, :, :]
    train_output = train_output[permutation]
    # 设置测试集

    # 标签one hot化
    lb = LabelBinarizer()
    # train_output = lb.fit_transform(train_output)  # transfer label to binary value
    train_output = to_categorical(train_output)  # transfer binary label to one-hot. IMPORTANT

    kk = math.ceil(height * 0.8)
    print('kk=:', kk)
    x_test = train_input[kk:, :, :]
    y_test = train_output[kk:]
    train_input = train_input[0:kk, :, :]
    train_output = train_output[0:kk]

    model_m = Sequential()
    model_m.add(Conv1D(25, 5, activation='relu', input_shape=(width, length)))
    model_m.add(MaxPooling1D(2))
    model_m.add(Dropout(0.4))
    model_m.add(Conv1D(50, 5, activation='relu'))
    model_m.add(MaxPooling1D(2))
    model_m.add(Dropout(0.4))
    model_m.add(Conv1D(100, 5, activation='relu'))
    model_m.add(MaxPooling1D(2))
    model_m.add(Dropout(0.2))
    model_m.add(Conv1D(200, 5, activation='relu'))
    model_m.add(MaxPooling1D(2))
    model_m.add(Dropout(0.2))
    model_m.add(GlobalAveragePooling1D())
    model_m.add(Dense(num_class, activation='softmax'))
    print(model_m.summary())

    # 回调函数Callbacks
    callbacks_list = [
        # keras.callbacks.TensorBoard(log_dir='./logs'),
        # keras.callbacks.ModelCheckpoint(
        #     filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        #     monitor='val_loss', verbose=0,  save_best_only=True,
        #     # save_weights_only=True,
        #     mode='auto', period=1),
        # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
        #                                   min_delta=0.0001, cooldown=0, min_lr=0),
        # keras.callbacks.EarlyStopping(monitor='val_loss', patience=0)
    ]

    model_m.compile(loss='categorical_crossentropy',
                    optimizer='adam', metrics=['accuracy'],
                    )

    BATCH_SIZE = 100
    EPOCHS = 100

    # 训练模型
    history = model_m.fit(train_input, train_output,
                          batch_size=BATCH_SIZE,
                          epochs=EPOCHS,
                          validation_data=(x_test, y_test),
                          verbose=2, shuffle=True,
                          callbacks=callbacks_list,
                          validation_split=0.2)

    print(history.history.keys())
    info_text.insert('insert', history.history.keys())
    info_text.insert('insert', '\n')

    '''
    # 绘制训练 & 验证的准确率值
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    ##
    ##
    ### 绘制训练 & 验证的损失值
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()
    '''

    score = model_m.evaluate(x_test, y_test,
                             batch_size=10,
                             verbose=2,
                             sample_weight=None)

    info_text.insert('insert', score[0])
    info_text.insert('insert', '\n')

    info_text.insert('insert', score[1])
    info_text.insert('insert', '\n')

    w2 = tk.Tk()
    w2.title('Save the cnn model as *.h5')
    w2.geometry('400x200')

    tk.Label(w2, text="Input the name of model.h5:").place(x=20, y=50)

    var2 = tk.StringVar()
    var2.set("C:\PycharmProjects\eeg\data\music\music_data\*.h5")

    h5_model = tk.Entry(w2, textvariable=var2)
    h5_model.place(x=200, y=50)

    btn_ok = tk.Button(w2, text="OK", command=save_h5)
    btn_ok.place(x=230, y=130)
    btn_close = tk.Button(w2, text="Close", command=w2.destroy)
    btn_close.place(x=300, y=130)





menubar = tk.Menu(window, fg='green')

filemenu = tk.Menu(menubar, tearoff=0, fg='green')
menubar.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="Open", command=open_file)
filemenu.add_command(label="Save", command=save_file)
filemenu.add_separator()
filemenu.add_command(label='Exit', command=window.quit)

make_sample = tk.Menu(menubar, tearoff=0, fg='blue')
menubar.add_cascade(label="MAKE_Sample", menu=make_sample)
make_sample.add_command(label="Single", command=single_sample)
make_sample.add_command(label="Combine", command=combine_sample)

cnn_go = tk.Menu(menubar, tearoff=0, fg='red')
menubar.add_cascade(label="DEEP_CNN", menu=cnn_go)
cnn_go.add_command(label="GO", command=deepcnn_go)

window.config(menu=menubar)
window.mainloop()
