
import matplotlib.pyplot as plt
import os
from sklearn import preprocessing
from scipy import signal
import numpy as np
import torch
import pandas as pd


# 预处理函数
def de_extreme_t(in_t, exclude_max=65534, exclude_min=5000):  #################
    '''
    :param in_t:输入要抽去极端线条的tensor
    :param exclude_max:摸到这个高度的谱线被抽走
    :param exclude_min: 没过这个高度的谱线被抽走
    :return: 抽去极端谱线后的tensor
    '''
    mask = (torch.max(in_t, dim=0).values <= exclude_max) & (torch.max(in_t, dim=0).values >= exclude_min)
    out_t = in_t[:, mask]
    return out_t

def preprocess(items, name, xaxis):

    de_extremed = de_extreme_t(items)
    print(de_extremed.shape)
    sged1 = signal.savgol_filter(de_extremed, 7, 2, deriv=0, axis=0)
    scaled2 = preprocessing.scale(sged1)
    samples = scaled2

    ##################to show the effect of preprocess, use below

    plt.subplot(2, 2, 1)
    plt.plot(xaxis, items)
    plt.title('0-original {} specs'.format(items.shape[-1]))
    plt.subplot(2, 2, 2)
    plt.plot(xaxis, de_extremed)
    plt.title('1-de_extremed')
    plt.subplot(2, 2, 3)
    plt.plot(xaxis, sged1)
    plt.title('2-SG-filtered')
    plt.subplot(2, 2, 4)
    plt.plot(xaxis, scaled2)
    plt.title('3-scaled {} specs'.format(scaled2.shape[-1]))
    plotslocation = '/home/jzq/PycharmProjects/Br/Question-塑料分析/plots/plotsBr3'
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    plt.savefig(os.path.join(plotslocation, name), dpi=150)

    # if plot others, plot location here:
    # plotslocation63k = 'D:\\22\\1211_IR_twj\\xlsx数据思考区\\6-25总实验池\\7-3预筛选后的测试集\\63k_plots\\'
    # plt.savefig(os.path.join(plotslocation63k, name), dpi=150)


    plt.clf()
    return samples


path = '/home/jzq/PycharmProjects/Br/Question-塑料分析'
xaxis = np.linspace(887.1, 1715.45, 255)
# xpath = 'F:/PythonFiles/NIR-classify'
# nir_location = os.path.join(xpath, 'wavelength.csv')
# nir = pd.read_csv(nir_location)
# xaxis = nir.to_numpy()

files = os.path.join(path, 'Data3/br')

# if plot others, xlsx location here:
# clxlsxpath = 'D:\\22\\1211_IR_twj\\xlsx数据思考区\\6-25总实验池\\7-3预筛选后的测试集\\实验6-3-k\\normal\\train\\'
# files = clxlsxpath
for file in os.listdir(files):

    df = pd.read_excel(os.path.join(files, file))
    data = df.to_numpy()
    data = torch.tensor(data, dtype=torch.float32)

name = file.split('.')[0]
data2 = preprocess(data, name, xaxis)
print(data2.shape, file)

