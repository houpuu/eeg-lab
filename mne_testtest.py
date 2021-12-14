from mne.io import concatenate_raws, read_raw_edf
import matplotlib.pyplot as plt
import mne


raw = read_raw_edf("C:\PycharmProjects\eeg\data\edf\mne_test3.edf",preload=False)
mapping ={'Fp1-A1':'Fp1', 'Fp2-A2':'Fp2', 'F3-A1':'F3', 'F4-A2':'F4', 'C3-A1':'C3', 'C4-A2':'C4',
          'P3-A1':'P3', 'P4-A2':'P4', 'O1-A1':'O1', 'O2-A2':'O2', 'F7-A1':'F7', 'F8-A2':'F8',
          'T3-A1':'T3', 'T4-A2':'T4', 'T5-A1':'T5', 'T6-A2':'T6'}
raw.rename_channels(mapping)
montage = mne.channels.read_custom_montage("C:\PycharmProjects\eeg\data\edf\my_location.locs")
raw.set_montage(montage)

print(raw.info)
sfreq = raw.info['sfreq']
data, times = raw[:, 1:int(sfreq*10)]

#创建evokeds对象
evoked = mne.EvokedArray(data, raw.info)
#evokeds设置通道
evoked.set_montage(montage)
#画图
mne.viz.plot_topomap(evoked.data[:, 1], evoked.info, show=False)


"""
绘制功率谱密度
"""
raw.plot_psd()
plt.show()
