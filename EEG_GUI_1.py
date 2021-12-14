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

tk.Label(window, text="Input the width of sample:").place(x=350, y=120)
tk.Label(window, text="Input the label of sample:").place(x=350, y=160)
tk.Label(window, text="Input the times of sample(s):").place(x=350, y=200)


def open_file():
    global raw
    global evoked
    global montage
    global epochs
    global x_test
    global sfreq


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

def MAKE_SAMPLES():

    global new_time
    global new_width
    global new_label



    var1 = tk.StringVar()
    var1.set('250')
    new_width = tk.Entry(window, textvariable=var1, width=3)
    new_width.place(x=550, y=120)

    var2 = tk.StringVar()
    var2.set('1')
    new_label = tk.Entry(window, textvariable=var2, width=3)
    new_label.place(x=550, y=160)

    var3 = tk.StringVar()
    var3.set('10')
    new_time = tk.Entry(window, textvariable=var3, width=3)
    new_time.place(x=550, y=200)


def make_OK():

    global x_test

    data, times = raw[:, 0:int(sfreq * int(new_time.get()))]



    ss = StandardScaler()
    scaler = ss.fit(data)
    scaled_train = scaler.transform(data)
    m_data = np.transpose(scaled_train)

    #m_data = np.transpose(data)

    new_width1 = int(new_width.get())
    x_test = np.reshape(m_data, (-1, new_width1, 16))

    plt.plot(m_data)
    plt.title("Sample channels")
    plt.show()


    y_test = int(new_label.get())

    info_text1 = tk.Text(window, width=10, height=3)
    info_text1.pack()
    info_text1.place(x=550, y=240)
    info_text1.insert('insert', x_test.shape)
    info_text1.insert('insert', '\n')
    info_text1.insert('insert', m_data.shape)
    info_text1.insert('insert', '\n')
    info_text1.insert('insert', y_test)
    info_text1.insert('insert', '\n')

    # 创建evokeds对象
    evoked = mne.EvokedArray(data, raw.info)
    # evokeds设置通道
    evoked.set_montage(montage)


def sample_plot():

    plt.plot(x_test[int(new_label.get()), :, :])
    plt.title("Sample channels")
    plt.show()

def data_plot():

    data, times = raw[:, 0:int(sfreq * int(new_time.get()))]
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


    #train_input = x_test

    #model_m = load_model('C:\PycharmProjects\EEG_GUI\music_data\music_model.h5')
    feature = model_m.predict(x_test)
    #print('predict feature:', feature, 'y_true', y_test)

    recognition_text = tk.Text(window, width=20, height=10)
    recognition_text.pack()
    recognition_text.place(x=950, y=100)
    #recognition_text.insert('insert', feature)
    recognition_text.insert('insert', np.argmax(feature, axis=1))

def RAW_INFO():
    info_text = tk.Text(window, width=40, height=15)
    info_text.pack()
    info_text.place(x=50, y=320)
    info_text.insert('insert', raw.info)



#--------------------------------------------------------------------------------------------------------
# fist level
btn_sample = tk.Button(window, text="NEW_FILE", bg='green', fg='yellow', font=5, width=18, command=open_file)
btn_sample.place(x=50, y=50)
btn_sample = tk.Button(window, text="MAKE_SAMPLES", bg='green', fg='yellow', font=5, width=18, command=MAKE_SAMPLES)
btn_sample.place(x=350, y=50)
btn_sample = tk.Button(window, text="CHOOSE_CNN", bg='green', fg='yellow', font=5, width=18, command=open_file)
btn_sample.place(x=650, y=50)
btn_sample = tk.Button(window, text="RECOGNIZE", bg='green', fg='yellow', font=5, width=18, command=Recognize)
btn_sample.place(x=950, y=50)

#second level
btn_sample = tk.Button(window, text="RAW INFORMATION", bg='blue', fg='yellow', width=17,  command=RAW_INFO)
btn_sample.place(x=50, y=120)
btn_sample = tk.Button(window, text="RAW STATISTIC", bg='blue', fg='yellow', width=17, command=RAW_INFO)
btn_sample.place(x=50, y=160)
btn_sample = tk.Button(window, text="RAW PLOT", bg='blue', fg='yellow', width=17, command=RAW_PLOT)
btn_sample.place(x=50, y=200)
btn_sample = tk.Button(window, text="RAW PSD", bg='blue', fg='yellow', width=17, command=PSD)
btn_sample.place(x=50, y=240)
btn_sample = tk.Button(window, text="RAW TOPO", bg='blue', fg='yellow', width=17, command=TOPO)
btn_sample.place(x=50, y=280)

btn_sample = tk.Button(window, text="MAKE_OK", bg='blue', fg='yellow', width=17, command=make_OK)
btn_sample.place(x=350, y=240)
btn_sample = tk.Button(window, text="SAMPLE PLOT", bg='blue', fg='yellow', width=17, command=sample_plot)
btn_sample.place(x=350, y=280)
btn_sample = tk.Button(window, text="DATA PLOT", bg='blue', fg='yellow', width=17, command=data_plot)
btn_sample.place(x=350, y=320)

btn_sample = tk.Button(window, text="MUSIC EMOTION", bg='blue', fg='yellow', width=17,  command=Music_model)
btn_sample.place(x=650, y=120)

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

window.config(menu=menubar)
window.mainloop()

