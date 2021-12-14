import numpy as np
import matplotlib.pyplot as plt
import pywt
import mne
import scipy.io as scio
import math
from numpy import *

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

def TimeFrequencyWP(data, fs, wavelet, maxlevel):

    wavelet_data = []
    new_data = array([[0 for i in range(0, 4000)]for j in range(0, 4)])
    print(new_data.shape)

    # 小波包变换这里的采样频率为250，如果maxlevel太小部分波段分析不到
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    #print(len(freqTree))
    # 计算maxlevel最小频段的带宽
    freqBand = fs / (2 ** maxlevel)
    #print(freqBand)
    #######################根据实际情况计算频谱对应关系，这里要注意系数的顺序
    # 绘图显示
    fig, axes = plt.subplots(len(iter_freqs) + 1, 1, figsize=(12, 6), sharex=True, sharey=False)
    # 绘制原始数据
    axes[0].plot(data)
    axes[0].set_title('原始数据')
    for iter in range(len(iter_freqs)):
        # 构造空的小波包
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
        #print(iter)
        for i in range(len(freqTree)):
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand


            # 判断第i个频段是否在要分析的范围内
            if (iter_freqs[iter]['fmin'] <= bandMin and iter_freqs[iter]['fmax'] >= bandMax):
                # 给新构造的小波包参数赋值
                new_wp[freqTree[i]] = wp[freqTree[i]].data

        # 绘制对应频率的数据
        wavelet_data = new_wp.reconstruct(update=True)
        axes[iter + 1].plot(wavelet_data[0:4000])
        #axes[iter + 1].plot(new_wp.reconstruct(update=True))
        # 设置图名
        axes[iter + 1].set_title(iter_freqs[iter]['name'])
        #保存文件
        #print(wavelet_data)

        new_data[iter][0:4000] = wavelet_data[0:4000]


    plt.show()
    plt.plot(new_data[1][0:4000])
    plt.show()
    #print(new_data)


if __name__ == '__main__':
    # 读取筛选好的epoch数据
    dataFile = 'C:\eeg_project\eeg_DATA\music\data\孙雪映数据\孙雪莹悲伤清洗数据\c_2019041001001.mat'
    data = scio.loadmat(dataFile)
    data_input = data['EEGdata']
    #print(data_input.shape)
    '''
    train_input = data['train_input']
    train_output = data['train_output']
    print(train_input.shape)
'''
    dataCom = (data_input[0, 0:4000])

    
    TimeFrequencyWP(dataCom, 1000, wavelet='db4', maxlevel=10)

