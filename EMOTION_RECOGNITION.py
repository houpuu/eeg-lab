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

sys.path.append(r'.')

from keras.utils import to_categorical
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import math

window = tk.Tk()
window.title('Welcome to the EEG World')
window.geometry('1200x600')

tk.Label(window, text="WELCOME TO EEG WORLD!", font=25).pack()

def open_file():
    global raw
    global evoked
    global montage
    global epochs
    global x_test
    global sfreq
    global ch_names


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


def MAKE_SAMPLES():

    global new_time0
    global new_time1
    global new_width
    global new_label

    tk.Label(window, text="Input the width of sample:").place(x=700, y=230)
    tk.Label(window, text="Input the label of sample:").place(x=700, y=290)
    tk.Label(window, text="Star time of sample(s):").place(x=700, y=250)
    tk.Label(window, text="End  time of sample(s):").place(x=700, y=270)

    var1 = tk.StringVar()
    var1.set('250')
    new_width = tk.Entry(window, textvariable=var1, width=3, relief='flat')
    new_width.place(x=890, y=230)

    var2 = tk.StringVar()
    var2.set('1')
    new_label = tk.Entry(window, textvariable=var2, width=3, relief='flat')
    new_label.place(x=890, y=290)

    var3 = tk.StringVar()
    var3.set('0')
    new_time0 = tk.Entry(window, textvariable=var3, width=3, relief='flat')
    new_time0.place(x=890, y=250)

    var4 = tk.StringVar()
    var4.set('10')
    new_time1 = tk.Entry(window, textvariable=var4, width=3, relief='flat')
    new_time1.place(x=890, y=270)


def make_OK():

    global x_test
    global data

    data, times = raw[:, int(sfreq * int(new_time0.get())):int(sfreq * int(new_time1.get()))]
    ss = StandardScaler()
    scaler = ss.fit(data)
    scaled_train = scaler.transform(data)
    m_data = np.transpose(scaled_train)
    new_width1 = int(new_width.get())
    x_test = np.reshape(m_data, (-1, new_width1, 16))
    y_test = int(new_label.get())
    info_text1 = tk.Text(window, width=20, height=1, relief='flat')
    info_text1.pack()
    info_text1.place(x=930, y=290)
    info_text1.insert('insert', x_test.shape)
    info_text1.insert('insert', data.shape)

    plt.plot(m_data)
    plt.title("Sample channels")
    plt.show()

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

def RAW_INFO():
    info_text = tk.Text(window, width=40, height=8, relief='flat')
    info_text.pack()
    info_text.place(x=700, y=50)
    info_text.insert('insert', raw.info)

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

    """""
    epochs.plot()
    plt.show()
    """""
    epochs.plot_psd_topomap()
    plt.show()


def about_ver():
    statistic_window = tk.Toplevel()
    statistic_window.title('Welcome to the EEG World')
    statistic_window.geometry('600x200')
    word_1 = '''这个软件可以实现EEG采集信号类别的识别（*.edf版本）:
                                    1，导入采集数据；
                                    2，对数据进行样本的制作；
                                    3，选取CNN训练好的模型（目前有关于情绪的模型）；
                                    4，识别采集数据的（情绪类别）;
                                    '''
    tk.Label(statistic_window, text=word_1, justify='left').pack()


#--------------------------------------------------------------------------------------------------------
# fist level
btn_sample = tk.Button(window, text="1. OPEN_FILE", bg='darkseagreen', fg='black', font=5, width=18, command=open_file)
btn_sample.place(x=50, y=50)
btn_sample = tk.Button(window, text="2. MAKE_SAMPLES", bg='darkseagreen', fg='black', font=5, width=18, command=MAKE_SAMPLES)
btn_sample.place(x=50, y=230)
btn_sample = tk.Button(window, text="CHOOSE_CNN", bg='darkgray', fg='black', font=5, width=18)
btn_sample.place(x=50, y=410)



#second level
btn_sample = tk.Button(window, text="RAW INFORMATION", bg='darkturquoise', fg='black', width=17,  command=RAW_INFO)
btn_sample.place(x=300, y=50)
btn_sample = tk.Button(window, text="RAW PLOT", bg='burlywood', fg='black', width=17, command=RAW_PLOT)
btn_sample.place(x=500, y=50)
btn_sample = tk.Button(window, text="RAW PSD", bg='burlywood', fg='black', width=17, command=PSD)
btn_sample.place(x=500, y=90)
btn_sample = tk.Button(window, text="RAW TOPO", bg='burlywood', fg='black', width=17, command=TOPO)
btn_sample.place(x=500, y=130)

btn_sample = tk.Button(window, text="3. MAKE_OK", bg='darkseagreen', fg='black', width=17, command=make_OK)
btn_sample.place(x=300, y=230)
btn_sample = tk.Button(window, text="DATA PLOT", bg='burlywood', fg='black', width=17, command=data_plot)
btn_sample.place(x=500, y=310)
btn_sample = tk.Button(window, text="BRAIN TOPOMAP", bg='burlywood', fg='black', width=17, command=topo_plot)
btn_sample.place(x=500, y=230)
btn_sample = tk.Button(window, text="BOUND TOPOMAP", bg='burlywood', fg='black', width=17, command=psd_topomap)
btn_sample.place(x=500, y=270)


btn_sample = tk.Button(window, text="4. MUSIC EMOTION", bg='darkseagreen', fg='black', width=17,  command=Music_model)
btn_sample.place(x=300, y=410)
btn_sample = tk.Button(window, text="PICTURE EMOTION", bg='darkgray', fg='black', width=17,  command=Music_model)
btn_sample.place(x=300, y=450)
btn_sample = tk.Button(window, text="COLOR EMOTION", bg='darkgray', fg='black', width=17,  command=Music_model)
btn_sample.place(x=300, y=490)
btn_sample = tk.Button(window, text="5. RECOGNIZE", bg='darkseagreen', fg='black', width=17, command=Recognize)
btn_sample.place(x=500, y=410)
#---------------------------------------------------------------------------------------------------

menubar = tk.Menu(window, fg='green')

filemune = tk.Menu(menubar, tearoff=0, fg='green')
menubar.add_cascade(label="File", menu=filemune)
filemune.add_command(label="Open", command=open_file)
filemune.add_command(label="Information", command=RAW_INFO)
filemune.add_separator()
filemune.add_command(label='Exit', command=window.quit)

plotmenu = tk.Menu(menubar, tearoff=0,fg='blue')
menubar.add_cascade(label="Plot", menu=plotmenu)
plotmenu.add_command(label="PSD", command=PSD)
plotmenu.add_command(label='TOPO', command=TOPO)

emotionmune = tk.Menu(menubar, tearoff=0, fg='red')
menubar.add_cascade(label="Emotion", menu=emotionmune)
emotionmune.add_command(label="Recognize", command=Recognize)

helpmenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='HELP', menu=helpmenu)
helpmenu.add_command(label='ABOUT', command=about_ver)

window.config(menu=menubar)
window.mainloop()

