import sys
import matplotlib.pyplot as plt
import mne
from mne.io import concatenate_raws, read_raw_edf
from tkinter import *
from tkinter.messagebox import *
from tkinter import filedialog
import tkinter as tk
import os
from keras.models import load_model
import scipy.io as scio
from numpy import *
sys.path.append(r'.')

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
from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import math

window = tk.Tk()
window.title('Welcome to the EEG World')
window.geometry('1200x600')

sfreq = 1000

# make training set
info_text = tk.Text(window, width=40, height=10, relief='flat')
info_text.pack()
info_text.place(x=800, y=120)

def edf_file():
    global raw
    global evoked
    global montage
    global epochs
    global x_test
    global ch_names
    global edf0

    edf0 = 1

    var = tk.StringVar()

    file_path = filedialog.askopenfilename()
    raw = read_raw_edf(file_path, preload=False)

    mapping = {'Fp1-A1': 'Fp1', 'Fp2-A2': 'Fp2', 'F3-A1': 'F3', 'F4-A2': 'F4', 'C3-A1': 'C3', 'C4-A2': 'C4',
               'P3-A1': 'P3', 'P4-A2': 'P4', 'O1-A1': 'O1', 'O2-A2': 'O2', 'F7-A1': 'F7', 'F8-A2': 'F8',
               'T3-A1': 'T3', 'T4-A2': 'T4', 'T5-A1': 'T5', 'T6-A2': 'T6'}
    raw.rename_channels(mapping)
    montage = mne.channels.read_custom_montage("C:\PycharmProjects\eeg\data\edf\my_location.locs")
    epochs = raw.set_montage(montage)
    sfreq = raw.info['sfreq']
    ch_names = raw.info['ch_names']

    info_text = tk.Text(window, width=40, height=15, relief='flat')
    info_text.pack()
    info_text.place(x=480, y=50)
    info_text.insert('insert', raw.info)

def mat_file():

    global raw
    global edf0

    edf0 = 0

    file_path = filedialog.askopenfilename(filetypes=[("MAT", ".mat")])
    data = scio.loadmat(file_path)
    raw = data['train_input']

    raw = np.reshape(raw, (-1, 16))
    raw = np.transpose(raw)

    #height, width, length = raw.shape


    info_text = tk.Text(window, width=40, height=15, relief='flat')
    info_text.pack()
    info_text.place(x=480, y=50)
    info_text.insert('insert', file_path)
    info_text.insert('insert', '\n')
    info_text.insert('insert', raw.shape)


def MAKE_SAMPLES():

    global new_time0
    global new_time1
    global new_width
    global new_label

    tk.Label(window, text="Input the LABEL of sample:").place(x=50, y=410)
    tk.Label(window, text="STAR time of sample(s):").place(x=50, y=440)
    tk.Label(window, text="END  time of sample(s):").place(x=50, y=460)
    tk.Label(window, text="EPOCH LENGTH of sample:").place(x=50, y=490)

    var1 = tk.StringVar()
    var1.set('250')
    new_width = tk.Entry(window, textvariable=var1, width=3, relief='flat')
    new_width.place(x=280, y=490)

    var2 = tk.StringVar()
    var2.set('1')
    new_label = tk.Entry(window, textvariable=var2, width=3, relief='flat')
    new_label.place(x=280, y=410)

    var3 = tk.StringVar()
    var3.set('0')
    new_time0 = tk.Entry(window, textvariable=var3, width=3, relief='flat')
    new_time0.place(x=280, y=440)

    var4 = tk.StringVar()
    var4.set('10')
    new_time1 = tk.Entry(window, textvariable=var4, width=3, relief='flat')
    new_time1.place(x=280, y=460)


def make_OK():

    global x_test
    global data
    global train_input
    global train_output

    if edf0 == 1:
        data, times = raw[:, int(sfreq * int(new_time0.get())):int(sfreq * int(new_time1.get()))]
    else:
        data = raw[:, int(sfreq * int(new_time0.get())):int(sfreq * int(new_time1.get()))]

    ss = StandardScaler()
    scaler = ss.fit(data)
    scaled_train = scaler.transform(data)
    m_data = np.transpose(scaled_train)

    #new_width = int(new_width.get())
    train_input = np.reshape(m_data, (-1, int(new_width.get()), 16))
    #new_label = int(new_label.get())

    height, width, length = train_input.shape
    hang = int(height)

    train_output = np.ones((hang, 1), dtype=int) * int(new_label.get())

    info_text1 = tk.Text(window, width=20, height=2, relief='flat')
    info_text1.pack()
    info_text1.place(x=480, y=260)
    info_text1.insert('insert', train_input.shape)


    plt.plot(m_data)
    plt.title("Sample channels")
    plt.show()

def save_file():

    #global train_input
    #global train_output

    new_file = filedialog.asksaveasfilename()
    scio.savemat(new_file, {'train_input': train_input, 'train_output': train_output})

    info_text1 = tk.Text(window, width=40, height=2, relief='flat')
    info_text1.pack()
    info_text1.place(x=480, y=280)
    info_text1.insert('insert', new_file)


def data_plot():

    plt.plot(data.T)
    plt.title("Sample channels")
    plt.show()

def RAW_PLOT():
    raw.plot()
    plt.show()

def PSD():
    raw.plot_psd()
    plt.show()

def TOPO():
    raw.plot_psd_topo()
    plt.show()

def Music_model():

    global model_m
    model_m = load_model('C:\PycharmProjects\EEG_GUI\music_data\music_model.h5')

def Recognize():

    feature = model_m.predict(x_test)
    recognition_text = tk.Text(window, width=50, height=4, relief='flat')
    recognition_text.pack()
    recognition_text.place(x=700, y=410)
    recognition_text.insert('insert', np.argmax(feature, axis=1))

def topo_plot():
    # 创建evokeds对象
    evoked = mne.EvokedArray(data, raw.info)
    # evokeds设置通道
    evoked.set_montage(montage)
    # 画图-脑地形图
    mne.viz.plot_topomap(evoked.data[:, 10], evoked.info, show=False)
    plt.show()

def psd_topomap():
    new_events = mne.make_fixed_length_events(raw, duration=5.)
    epochs = mne.Epochs(raw, new_events)
    epochs.set_montage(montage)

    epochs.plot_psd_topomap()
    plt.show()

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

    '''


    info_text = tk.Text(window, width=40, height=5, relief='flat')
    info_text.pack()
    info_text.place(x=800, y=200)
    '''

    info_text.insert('insert', file_path)
    info_text.insert('insert', '\n')



def save_h5():

    global model_m
    global h5_model

    h5_model = h5_model.get()
    # model_m.save('C:\PycharmProjects\eeg\data\music\music_model_20210706.h5')
    model_m.save(h5_model)

def deepcnn_go():
    global model_m
    global h5_model

    num_class = int(classnumber.get())

    file_path = filedialog.askopenfilename(filetypes=[("MAT", ".mat")])

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


    info_text = tk.Text(window, width=40, height=10, relief='flat')
    info_text.pack()
    info_text.place(x=800, y=400)

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


def about_ver():
    statistic_window = tk.Toplevel()
    statistic_window.title('Welcome to the EEG World')
    statistic_window.geometry('700x200')
    word_1 = '''这个软件可以实现EEG采集信号的单样本和样本集的生成:
                                    1，导入采集的原始信号*.edf或者*.mat；
                                    2，生成单样本信号，给定样本长度，和标签；
                                    3，生成数据集，将多个不同标签的信号的数据样本合成一个样本集，生成*.mat文件;
                                    4，导入一个样本集文件，实现CNN训练。
                                    '''
    tk.Label(statistic_window, text=word_1, justify='left').pack()


#--------------------------------------------------------------------------------------------------------
# fist level
btn_sample = tk.Button(window, text="1. EDF FILE", bg='darkseagreen', fg='black', font=5, width=18, command=edf_file)
btn_sample.place(x=50, y=50)
btn_sample = tk.Button(window, text="1. MAT FILE", bg='darkseagreen', fg='black', font=5, width=18, command=mat_file)
btn_sample.place(x=50, y=90)

btn_sample = tk.Button(window, text="2. ONE LABEL", bg='greenyellow', fg='black', font=5, width=18, command=MAKE_SAMPLES)
btn_sample.place(x=50, y=230)

btn_sample = tk.Button(window, text="3. DATA SET", bg='greenyellow', fg='black', font=5, width=18, command=combine_sample)
btn_sample.place(x=800, y=50)
com_B1 = tk.Button(window, text="ADD MORE", bg='burlywood', fg='black', command=add_file)
com_B1.place(x=1000, y=50)
com_B2 = tk.Button(window, text="NO MORE", bg='burlywood', fg='black', command=endadd_file)
com_B2.place(x=1000, y=80)

btn_sample = tk.Button(window, text="4. TRAINING_CNN", bg='lightblue', fg='black', font=5, width=18, command=deepcnn_go)
btn_sample.place(x=800, y=300)

tk.Label(window, text="CLASS NUMBER:").place(x=800, y=350)
var = tk.StringVar()
var.set('5')
classnumber = tk.Entry(window, textvariable=var, width=3, relief='flat')
classnumber.place(x=920, y=350)





#second level

#btn_sample = tk.Button(window, text="RAW INFORMATION", bg='darkturquoise', fg='black', width=17,  command=RAW_INFO)
#btn_sample.place(x=300, y=50)
btn_sample = tk.Button(window, text="RAW PLOT", bg='burlywood', fg='black', width=17, command=RAW_PLOT)
btn_sample.place(x=300, y=50)
btn_sample = tk.Button(window, text="RAW PSD", bg='burlywood', fg='black', width=17, command=PSD)
btn_sample.place(x=300, y=90)
btn_sample = tk.Button(window, text="RAW TOPO", bg='burlywood', fg='black', width=17, command=TOPO)
btn_sample.place(x=300, y=130)

btn_sample = tk.Button(window, text="MAKE OK", bg='greenyellow', fg='black', width=17, command=make_OK)
btn_sample.place(x=300, y=230)
btn_sample = tk.Button(window, text="SAVE FILE", bg='greenyellow', fg='black', width=17, command=save_file)
btn_sample.place(x=300, y=270)
btn_sample = tk.Button(window, text="DATA PLOT", bg='burlywood', fg='black', width=17, command=data_plot)
btn_sample.place(x=300, y=310)


#---------------------------------------------------------------------------------------------------

menubar = tk.Menu(window, fg='green')

filemenu = tk.Menu(menubar, tearoff=0, fg='green')
menubar.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="EDF_FILE", command=edf_file)
filemenu.add_command(label="MAT_FILE", command=mat_file)
filemenu.add_separator()
filemenu.add_command(label='Exit', command=window.quit)

plotmenu = tk.Menu(menubar, tearoff=0, fg='greenyellow')
menubar.add_cascade(label="MAKE DATA SET", menu=plotmenu)
plotmenu.add_command(label="ONE LABEL SAMPLE", command=MAKE_SAMPLES)
plotmenu.add_command(label='DATA SET', command=combine_sample)

emotionmenu = tk.Menu(menubar, tearoff=0, fg='lightblue')
menubar.add_cascade(label="Deep CNN ", menu=emotionmenu)
emotionmenu.add_command(label="TRAINING", command=deepcnn_go)

helpmenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='HELP', menu=helpmenu)
helpmenu.add_command(label='ABOUT', command=about_ver)

window.config(menu=menubar)
window.mainloop()

