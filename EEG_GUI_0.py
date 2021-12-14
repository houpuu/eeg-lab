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
window.geometry('800x400')

tk.Label(window, text="WELCOME TO EEG WORLD!", font=25).pack()


def open_file():
    global raw
    global evoked
    global montage
    global epochs
    global x_test


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


    data, times = raw[:, 0:int(sfreq * 10)]

    ss = StandardScaler()
    scaler = ss.fit(data)
    data = scaler.transform(data)

    x_test = data[:, 0:250]
    x_test = np.reshape(x_test, (-1, 250, 16))

    # 创建evokeds对象
    evoked = mne.EvokedArray(data, raw.info)
    # evokeds设置通道
    evoked.set_montage(montage)


    raw.plot()
    plt.show()

def PSD():
    raw.plot_psd()
    plt.show()

def TOPO():
    raw.plot_psd_topo()
    plt.show()

def Recognize():

    #train_input = np.zeros([1, 250, 16])
    train_input = x_test

    #file_path = filedialog.askopenfilename()
    #raw = read_raw_edf(file_path, preload=False)

    model_m = load_model('C:\PycharmProjects\EEG_GUI\music_data\music_model.h5')
    feature = model_m.predict(train_input)
    #print('predict feature:', feature, 'y_true', y_test)

    recognition_text = tk.Text(window, width=120, height=4)
    recognition_text.pack()
    #recognition_text.insert('insert', feature)
    recognition_text.insert('insert', np.argmax(feature))

def RAW_INFO():
    info_text = tk.Text(window, width=80, height=15)
    info_text.pack()
    info_text.insert('insert', raw.info)



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

