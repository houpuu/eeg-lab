import numpy as np
import matplotlib.pyplot as plt
import pywt
import mne
import scipy.io as scio
import math


mne.set_log_level(False)


######################################################连续小波变换##########
# totalscal小波的尺度，对应频谱分析结果也就是分析几个（totalscal-1）频谱
def TimeFrequencyCWT(data, fs, totalscal, wavelet='cgau8'):
    # 采样数据的时间维度
    t = np.arange(data.shape[0]) / fs
    # 中心频率
    wcf = pywt.central_frequency(wavelet=wavelet)
    # 计算对应频率的小波尺度
    cparam = 2 * wcf * totalscal
    scales = cparam / np.arange(totalscal, 1, -1)
    # 连续小波变换
    [cwtmatr, frequencies] = pywt.cwt(data, scales, wavelet, 1.0 / fs)
    # 绘图
    plt.figure(figsize=(8, 4))
    plt.subplot(211)
    plt.plot(t, data)
    plt.xlabel(u"time(s)")
    plt.title(u"Time spectrum")
    plt.subplot(212)
    plt.contourf(t, frequencies, abs(cwtmatr))
    plt.ylabel(u"freq(Hz)")
    plt.xlabel(u"time(s)")
    plt.subplots_adjust(hspace=0.04)
    plt.show()


if __name__ == '__main__':
    # 读取筛选好的epoch数据
    dataFile = 'C:\PycharmProjects\eeg\data\music\music_data\happy_3.mat'
    data = scio.loadmat(dataFile)
    train_input = data['train_input']
    train_output = data['train_output']
    print(train_input.shape)

    dataCom = (train_input[1, 0:250, 0])

    TimeFrequencyCWT(dataCom, fs=250, totalscal=8, wavelet='cgau8')