# 短时傅里叶变换和FIR滤波效果对比

import mne
import matplotlib.pyplot as plt
from scipy import signal, fft
import numpy as np
import pywt
from mne.io import concatenate_raws, read_raw_edf

# 设置MNE库打印Log的级别
mne.set_log_level(False)

raw = read_raw_edf("C:\PycharmProjects\eeg\data\edf\mne_test3.edf",preload=False)
'''
mapping ={'Fp1-A1':'Fp1', 'Fp2-A2':'Fp2', 'F3-A1':'F3', 'F4-A2':'F4', 'C3-A1':'C3', 'C4-A2':'C4',
          'P3-A1':'P3', 'P4-A2':'P4', 'O1-A1':'O1', 'O2-A2':'O2', 'F7-A1':'F7', 'F8-A2':'F8',
          'T3-A1':'T3', 'T4-A2':'T4', 'T5-A1':'T5', 'T6-A2':'T6'}
raw.rename_channels(mapping)
montage = mne.channels.read_custom_montage("C:\PycharmProjects\eeg\data\edf\my_location.locs")
raw.set_montage(montage)

#print(raw.info)
'''
sfreq = raw.info['sfreq']
data, times = raw[:, 1:int(sfreq*10)]

# 需要分析的频带及其范围
bandFreqs = [
    {'name': 'Delta', 'fmin': 1, 'fmax': 3},
    {'name': 'Theta', 'fmin': 4, 'fmax': 7},
    {'name': 'Alpha', 'fmin': 8, 'fmax': 13},
    {'name': 'Beta', 'fmin': 14, 'fmax': 31},
    {'name': 'Gamma', 'fmin': 31, 'fmax': 40}
]


def __CalcWP(data, sfreq, wavelet, maxlevel, band):
    # 如果maxlevel太小部分波段分析不到
    wp = pywt.WaveletPacket(data=data, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
    # 频谱由低到高的对应关系，这里需要注意小波变换的频带排列默认并不是顺序排列，所以这里需要使用’freq‘排序。
    freqTree = [node.path for node in wp.get_level(maxlevel, 'freq')]
    # 计算maxlevel最小频段的带宽，采样频率的一半
    freqBand = (sfreq / 2) / (2 ** maxlevel)
    bandResult = []
    #######################根据实际情况计算频谱对应关系，这里要注意系数的顺序
    for iter_freq in band:
        # 构造空的小波包
        new_wp = pywt.WaveletPacket(data=None, wavelet=wavelet, mode='symmetric', maxlevel=maxlevel)
        for i in range(len(freqTree)):
            # 第i个频段的最小频率
            bandMin = i * freqBand
            # 第i个频段的最大频率
            bandMax = bandMin + freqBand
            # 判断第i个频段是否在要分析的范围内
            if (iter_freq['fmin'] <= bandMin and iter_freq['fmax'] >= bandMax):
                # 给新构造的小波包参数赋值
                # print('freq',bandMin, bandMax,'fmin',iter_freq['fmin'],'fmax',iter_freq['fmax'])
                new_wp[freqTree[i]] = wp[freqTree[i]].data
        # 计算对应频率的数据
        bandResult.append(new_wp.reconstruct(update=True))
    return bandResult


########################################小波包变换-重构造分析不同频段的特征(注意maxlevel，如果太小可能会导致部分频段分析不到)#########################
# 定义WP函数
# epochsData:epochs的数据（mumpy格式）
# sfreq:采样频率
# wavelet:小波类型
# maxlevel:小波层数
# band:频带类型

def WP(epochsData, sfreq, wavelet='db4', maxlevel=8, band=bandFreqs):
    # 输出的维度顺序为 频率->epoch->channel->timeData
    result = []
    for epochData in epochsData:
        channel = []
        for channelData in epochData:
            # print('channel:')
            channel.append(__CalcWP(channelData, sfreq, wavelet=wavelet, maxlevel=maxlevel, band=band))
        result.append(channel)
    return np.array(result).transpose((2, 0, 1, 3))


if __name__ == '__main__':
    # 加载fif格式的数据
    epochs = mne.read_epochs(r'F:\BaiduNetdiskDownload\BCICompetition\BCICIV_2a_gdf\Train\Fif\A02T_epo.fif')
    # 绘图验证结果
    plt.figure(figsize=(15, 10))
    # 获取采样频率
    sfreq = epochs.info['sfreq']
    # 想要分析的目标频带
    bandIndex = 0
    # 想要分析的channel
    channelIndex = 0
    # 想要分析的epoch
    epochIndex = 0
    # 绘制原始数据
    plt.plot(epochs.get_data()[epochIndex][channelIndex], label='Raw')
    # 计算FIR滤波后的数据并绘图（注意这里要使用copy方法，否则会改变原始数据）
    firFilter = epochs.copy().filter(bandFreqs[bandIndex]['fmin'], bandFreqs[bandIndex]['fmax'])
    plt.plot(firFilter.get_data()[epochIndex][channelIndex], c=(1, 0, 0), label='FIR_Filter')
    # 计算小波包滤波后的数据并绘图
    wpFilter = WP(epochs.get_data(), sfreq)
    plt.plot(wpFilter[bandIndex][epochIndex][channelIndex], c=(0, 1, 0), label='WP_Filter')
    # 绘制图例和图名
    plt.legend()
    plt.title(bandFreqs[bandIndex]['name'])

    ####################################FFT对比两种方法的频谱分布
    plt.figure(figsize=(15, 10))
    # 对FIR滤波后的数据进行FFT变换
    mneFIRFreq = np.abs(fft.fft(firFilter.get_data()[epochIndex][channelIndex]))
    # 对小波包滤波后的数据进行FFT变换，需要注意小波包变换后数据的点数可能会发生变化，这里截取数据保持一致性
    pointNum = epochs.get_data()[epochIndex][channelIndex].shape[0]
    wpFreq = np.abs(fft.fft(wpFilter[bandIndex][epochIndex][channelIndex][:pointNum]))
    # 想要绘制的点数
    pointPlot = 300
    # FIR滤波后x轴对应的频率幅值范围
    FIR_X = np.linspace(0, sfreq / 2, int(mneFIRFreq.shape[0] / 2))
    # 小波包滤波后x轴对应的频率幅值范围
    WP_X = np.linspace(0, sfreq / 2, int(wpFreq.shape[0] / 2))
    # 绘制FIR滤波后的频谱分布
    plt.plot(FIR_X[:pointPlot], mneFIRFreq[:pointPlot], c=(1, 0, 0), label='FIR_Filter')
    # 绘制小波包滤波后的频谱分布
    plt.plot(WP_X[:pointPlot], wpFreq[:pointPlot], c=(0, 1, 0), label='WP_FIlter')
    # 绘制图例和图名
    plt.legend()
    plt.title(bandFreqs[bandIndex]['name'])
    plt.show()