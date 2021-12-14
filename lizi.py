# 导入工具包
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne

"""
第一步：
从文件中读取诱发对象
"""
# 获取数据文件默认春芳地址
data_path = mne.datasets.sample.data_path()
# 构建文件存放的具体路径
fname = op.join(data_path, 'MEG', 'sample', 'sample_audvis-ave.fif')
# 根据文件存放的具体路径
evoked = mne.read_evokeds(fname, baseline=(None, 0), proj=True)

evoked_l_aud = evoked[0]
evoked_r_aud = evoked[1]
evoked_l_vis = evoked[2]
evoked_r_vis = evoked[3]



ts_args = dict(gfp=True, time_unit='s')
topomap_args = dict(sensors=False, time_unit='s')
evoked_r_aud.plot_joint(title='right auditory', times=[.09, .20],
                        ts_args=ts_args, topomap_args=topomap_args)