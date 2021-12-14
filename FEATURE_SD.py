import sys
import matplotlib.pyplot as plt
import mne
from mne.io import concatenate_raws, read_raw_edf
from tkinter import *
from tkinter.messagebox import *
from tkinter import filedialog
import tkinter as tk
import os
import scipy.io as scio
from numpy import *
sys.path.append(r'.')

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
import numpy as np
import math
import pywt


# 需要分析的四个频段
iter_freqs = [
    {'name': 'Delta', 'fmin': 0, 'fmax': 4},
    {'name': 'Theta', 'fmin': 4, 'fmax': 8},
    {'name': 'Alpha', 'fmin': 8, 'fmax': 13},
    {'name': 'Beta', 'fmin': 13, 'fmax': 35},
]

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
mne.set_log_level(False)

window = tk.Tk()
window.title('Welcome to the EEG World')
window.geometry('1200x600')

info_text = tk.Text(window, width=40, height=5, relief='flat')
info_text.pack()
info_text.place(x=600, y=120)

sfreq = 1000


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

    info_text = tk.Text(window, width=30, height=8, relief='flat')
    info_text.pack()
    info_text.place(x=50, y=150)
    info_text.insert('insert', raw.info)



def mat_file():

    global mat_raw
    global edf0

    edf0 = 0

    file_path = filedialog.askopenfilename(filetypes=[("MAT", ".mat")])
    data = scio.loadmat(file_path)
    mat_raw = data['EEGdata']

    '''mat_raw = np.reshape(mat_raw, (-1, 16))
    mat_raw = np.transpose(mat_raw)
'''
    info_text = tk.Text(window, width=30, height=5, relief='flat')
    info_text.pack()
    info_text.place(x=50, y=150)
    info_text.insert('insert', file_path)
    info_text.insert('insert', '\n')
    info_text.insert('insert', mat_raw.shape)


def PARAMETERS():

    global new_time0
    global new_time1
    global new_width
    global new_channel

    tk.Label(window, text="Number of Channel (0~15):").place(x=50, y=410)
    tk.Label(window, text="STAR time of sample(s):").place(x=50, y=440)
    tk.Label(window, text="END  time of sample(s):").place(x=50, y=460)
    tk.Label(window, text="EPOCH LENGTH of sample:").place(x=50, y=490)

    var2 = tk.StringVar()
    var2.set('1')
    new_channel = tk.Entry(window, textvariable=var2, width=3, relief='flat')
    new_channel.place(x=280, y=410)

    var3 = tk.StringVar()
    var3.set('0')
    new_time0 = tk.Entry(window, textvariable=var3, width=3, relief='flat')
    new_time0.place(x=280, y=440)

    var4 = tk.StringVar()
    var4.set('10')
    new_time1 = tk.Entry(window, textvariable=var4, width=3, relief='flat')
    new_time1.place(x=280, y=460)

    var1 = tk.StringVar()
    var1.set('250')
    new_width = tk.Entry(window, textvariable=var1, width=5, relief='flat')
    new_width.place(x=280, y=490)

def TimeFrequencyWP(data, fs, wavelet, maxlevel):

    global new_data

    time_L = int(sfreq * (int(new_time1.get()) - int(new_time0.get())))

    new_data = np.random.randn(4, time_L)
    print(new_data.shape)


    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    freqBand = fs / (2 ** maxlevel)
    fig, axes = plt.subplots(len(iter_freqs) + 1, 1, figsize=(12, 6), sharex=True, sharey=False)
    axes[0].plot(data)
    axes[0].set_title('原始数据')

    for iter in range(len(iter_freqs)):
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)

        for i in range(len(freqTree)):
            bandMin = i * freqBand
            bandMax = bandMin + freqBand

            if (iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax):
                new_wp[freqTree[i]] = wp[freqTree[i]].data

        wavelet_data = new_wp.reconstruct(update=True)
        axes[iter + 1].plot(wavelet_data[0:time_L])
        axes[iter + 1].set_title(iter_freqs[iter]['name'])
        new_data[iter][0:time_L] = wavelet_data[0:time_L]

    plt.show()


def WAVELET_TRANS():

    global data

    if edf0 == 1:
        data, times = raw[int(new_channel.get()), int(sfreq * int(new_time0.get())):int(sfreq * int(new_time1.get()))]
    else:
        data = mat_raw[int(new_channel.get()), int(sfreq * int(new_time0.get())):int(sfreq * int(new_time1.get()))]

    TimeFrequencyWP(data, 1000, wavelet='db4', maxlevel=10)

def FEATURES_BAR():

    global features_wavelet

    features_raw = np.reshape(new_data, (4, -1, int(new_width.get())))
    features_wavelet = np.random.randn(int(features_raw.shape[0]), int(features_raw.shape[1]))
    print(features_wavelet.shape)

    fig, axes = plt.subplots(4, 1, figsize=(12, 6), sharex=True, sharey=False)

    for i in range(features_raw.shape[0]):
        for j in range(features_raw.shape[1]):
            features_wavelet[i, j] = sum(abs(features_raw[i][j]))/(features_raw.shape[2])

        axes[i].bar(range(features_raw.shape[1]), features_wavelet[i, ])
        axes[i].set_title(iter_freqs[i]['name'])
    plt.show()


def SAVE_FEATURES():

    new_file = filedialog.asksaveasfilename()
    scio.savemat(new_file, {'features': features_wavelet})
'''
    info_text1 = tk.Text(window, width=40, height=2, relief='flat')
    info_text1.pack()
    info_text1.place(x=680, y=280)
    info_text1.insert('insert', new_file)
'''

def data_plot():

    global data

    if edf0 == 1:
        data, times = raw[int(new_channel.get()), int(sfreq * int(new_time0.get())):int(sfreq * int(new_time1.get()))]
    else:
        data = mat_raw[int(new_channel.get()), int(sfreq * int(new_time0.get())):int(sfreq * int(new_time1.get()))]

    plt.plot(data.T)
    plt.title("Sample channels")
    plt.show()

def RAW_PLOT():

    if edf0 == 1:
        raw.plot()
        plt.show()
    else:
        m_data = np.transpose(mat_raw)
        plt.plot(m_data)
        plt.show()


def PSD():
    raw.plot_psd()
    plt.show()

def TOPO():
    raw.plot_psd_topo()
    plt.show()

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

def FEATRUE_GENERATION():

    # make training set
    global features


    file_path = filedialog.askopenfilename(filetypes=[("MAT", ".mat"), ("EDF", ".edf")])

    data = scio.loadmat(file_path)
    features = data['features']

    info_text.insert('insert', file_path)
    info_text.insert('insert', '\n')

def add_file():

    global features

    file_path = filedialog.askopenfilename(filetypes=[("MAT", ".mat")])

    data = scio.loadmat(file_path)
    features0 = data['features']

    width, length = features.shape
    features = np.reshape(features, (-1, length))
    print(features.shape)

    width, length = features0.shape
    features0 = np.reshape(features0, (-1, length))
    print(features0.shape)

    features = np.hstack((features, features0))
    #features = np.reshape(features, (-1, width, length))

    info_text.insert('insert', file_path)
    info_text.insert('insert', '\n')

def endadd_file():

    new_file = filedialog.asksaveasfilename()
    scio.savemat(new_file, {'features': features})
    info_text.insert('insert', features.shape)
    info_text.insert('insert', '\n')
    info_text.insert('insert', '\n')



def T_TEST():

    global new_band
    global T_features

    file_path = filedialog.askopenfilename(filetypes=[("MAT", ".mat")])

    data = scio.loadmat(file_path)
    features = data['features']

    width, length = features.shape
    info_text.insert('insert', features.shape)
    info_text.insert('insert', '\n')
    info_text.insert('insert', '\n')

    tk.Label(window, text="Number of Band (1~4):").place(x=600, y=310)
    tk.Label(window, text="Significant Differences ?").place(x=600, y=350)

    varT = tk.StringVar()
    varT.set('1')
    new_band = tk.Entry(window, textvariable=varT, width=3, relief='flat')
    new_band.place(x=900, y=310)

    T_features = np.random.randn(1, int(features.shape[1]))
    T_features = features[new_band - 1, :]

def T_Test_GO():


    info_text.insert('insert', T_features.shape)
    info_text.insert('insert', '\n')
    info_text.insert('insert', '\n')

    info_text2 = tk.Text(window, width=2, height=1, relief='flat', font='10', fg='red')
    info_text2.pack()
    info_text2.place(x=900, y=350)

    info_text2.insert('insert', 'T')

def about_ver():
    statistic_window = tk.Toplevel()
    statistic_window.title('Welcome to the EEG World')
    statistic_window.geometry('1100x400')
    word_1 = '''这个软件可以实现EEG采集信号的单样本和样本集的生成:
    
                                    1，导入采集的原始信号*.edf或者*.mat；
                                    
                                    2，对给定长度和通道的数据进行小波变换，得到四个波段的小波重构信号， 
                                    
                                       根据给定的时常，计算每个波段的信号的
                                       
                                       均值，方差，最大值，偏度，能量等特征，
                                       
                                       构成４波段＊５特征＊样本个数的特征向量。
                                       
                                    3，合并多个文件，构造特征样本集，生成*.mat文件;
                                    
                                    4，进行T检验。
                                    '''
    tk.Label(statistic_window, text=word_1, justify='left', font='华文行楷，6').pack()


#--------------------------------------------------------------------------------------------------------
# fist level
btn_sample = tk.Button(window, text="1. EDF FILE", bg='darkseagreen', fg='black', font=5, width=18, command=edf_file)
btn_sample.place(x=50, y=50)
btn_sample = tk.Button(window, text="1. MAT FILE", bg='darkseagreen', fg='black', font=5, width=18, command=mat_file)
btn_sample.place(x=50, y=90)

btn_sample = tk.Button(window, text="2. PARAMETER", bg='greenyellow', fg='black', font=5, width=18, command=PARAMETERS)
btn_sample.place(x=50, y=260)

btn_sample = tk.Button(window, text="3. FEATRURES", bg='greenyellow', fg='black', font=5, width=18, command=FEATRUE_GENERATION)
btn_sample.place(x=600, y=50)
com_B1 = tk.Button(window, text="ADD MORE", bg='burlywood', fg='black', command=add_file)
com_B1.place(x=800, y=50)
com_B2 = tk.Button(window, text="NO MORE", bg='burlywood', fg='black', command=endadd_file)
com_B2.place(x=800, y=80)

btn_sample = tk.Button(window, text="4. T-TEST", bg='lightblue', fg='black', font=5, width=18, command=T_TEST)
btn_sample.place(x=600, y=260)


#second level

#btn_sample = tk.Button(window, text="RAW INFORMATION", bg='darkturquoise', fg='black', width=17,  command=RAW_INFO)
#btn_sample.place(x=300, y=50)
btn_sample = tk.Button(window, text="RAW PLOT", bg='burlywood', fg='black', width=17, command=RAW_PLOT)
btn_sample.place(x=300, y=50)
btn_sample = tk.Button(window, text="RAW PSD", bg='burlywood', fg='black', width=17, command=PSD)
btn_sample.place(x=300, y=90)
btn_sample = tk.Button(window, text="RAW TOPO", bg='burlywood', fg='black', width=17, command=TOPO)
btn_sample.place(x=300, y=130)

btn_sample = tk.Button(window, text="WAVELET TRANS", bg='greenyellow', fg='black', width=17, command=WAVELET_TRANS)
btn_sample.place(x=300, y=260)
btn_sample = tk.Button(window, text="FEATURES BAR", bg='greenyellow', fg='black', width=17, command=FEATURES_BAR)
btn_sample.place(x=300, y=300)
btn_sample = tk.Button(window, text="SAVE FEATURES", bg='greenyellow', fg='black', width=17, command=SAVE_FEATURES)
btn_sample.place(x=300, y=340)
btn_sample = tk.Button(window, text="DATA PLOT", bg='burlywood', fg='black', width=17, command=data_plot)
btn_sample.place(x=300, y=380)

btn_sample = tk.Button(window, text="GO", bg='burlywood', fg='black', width=17, command=T_Test_GO)
btn_sample.place(x=800, y=260)

#---------------------------------------------------------------------------------------------------

menubar = tk.Menu(window, fg='green')

filemenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="File", menu=filemenu)
filemenu.add_command(label="EDF_FILE", command=edf_file)
filemenu.add_command(label="MAT_FILE", command=mat_file)
filemenu.add_separator()
filemenu.add_command(label='Exit', command=window.quit)

plotmenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="WAVELET", menu=plotmenu)
plotmenu.add_command(label="PARAMETERS", command=PARAMETERS)
plotmenu.add_command(label="WAVELET TRANS", command=WAVELET_TRANS)
plotmenu.add_command(label='FEATURES BAR', command=FEATURES_BAR)

T_TESTmenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label="T-TEST ", menu=T_TESTmenu)
T_TESTmenu.add_command(label="T-TEST", command=T_TEST)

helpmenu = tk.Menu(menubar, tearoff=0)
menubar.add_cascade(label='HELP', menu=helpmenu)
helpmenu.add_command(label='ABOUT', command=about_ver)

window.config(menu=menubar)
window.mainloop()

