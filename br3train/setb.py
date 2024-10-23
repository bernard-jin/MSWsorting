from torch.utils.data import Dataset
import os
import os
import warnings
from sklearn import preprocessing
import pandas as pd
import torch
from scipy import signal
from torch.utils.data import Dataset
from torch.utils.data import random_split

warnings.filterwarnings("ignore", category=UserWarning)

'''
# br-item 一对一，item-specs一对多
1. br 细分到每一个spec
2. item打乱
3. item分TV
4. T/V的specs分别跨item打乱得到T' V'
5. T' V'训练

要预测还是要分类？
后续再处理，先吐出Br（kppm）与spec的对子
'''


class Brset(Dataset):
    def __init__(self, specroot, labelroot, mode):
        super(Brset, self).__init__()
        self.mode = mode
        self.specs, self.labels = self.load_xlsx(specroot, labelroot)

        a = int(0.7 * len(self.specs))
        b = int(0.85 * len(self.specs))

        if self.mode == 'train':
            self.specs = self.specs[:a]
            self.labels = self.labels[:a]
        elif self.mode == 'valid':
            self.specs = self.specs[a:b]
            self.labels = self.labels[a:b]
        elif self.mode == 'end':
            self.specs = self.specs[b:]
            self.labels = self.labels[b:]
        elif self.mode == 'kf-train':
            self.specs = self.specs[:b]
            self.labels = self.labels[:b]
        else:
            self.specs = self.specs
            self.labels = self.labels

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, idx):
        return self.specs[idx], self.labels[idx]

    def load_xlsx(self, fileroot, labelroot):
        '''
        1. dict the Br label list
        2. for xlsx in fileroot:
                xlsx conversions;
                for one spec in xlsx:
                    label of this spec = br value by dict[xlsx]
                end, by finishing all spec in this xlsx
            end, by finishing all xlsx

        '''
        df = pd.read_excel(labelroot)
        brdict = df.set_index('No')['Br(kppm)'].to_dict()
        speclist, labellist = [], []

        for file in os.listdir(fileroot):
            noitem = int(file.split('.')[0])  # 0023, like this
            file = os.path.join(fileroot, file)
            specs = pd.read_excel(file)
            specs = specs.to_numpy()
            specs = torch.tensor(specs, dtype=torch.float32)
            specs = self.preprocess(specs)

            for spec in specs.T:
                label = brdict[noitem]
                labellist.append(label)
                speclist.append(spec)

            print('file-{}  label-{}  data-{}  total-{} '.format(noitem, label, specs.shape, len(speclist)))
        return speclist, labellist

    # 预处理函数
    def de_extreme_t(self, in_t, exclude_max=65534, exclude_min=0.1):  #################
        '''
        :param in_t:输入要抽去极端线条的tensor
        :param exclude_max:摸到这个高度的谱线被抽走
        :param exclude_min: 没过这个高度的谱线被抽走
        :return: 抽去极端谱线后的tensor
        '''
        mask = (torch.max(in_t, dim=0).values <= exclude_max) & (torch.max(in_t, dim=0).values >= exclude_min)
        out_t = in_t[:, mask]
        return out_t

    def preprocess(self, items):

        items = self.de_extreme_t(items)
        items = signal.savgol_filter(items, 7, 2, deriv=0, axis=0)
        # items = preprocessing.scale(items)
        # items = preprocessing.minmax_scale(items)
        samples = items



        # # de_extremed = self.de_extreme_t(items)
        # sged1 = signal.savgol_filter(items, 7, 2, deriv=0, axis=0)
        # # scaled2 = preprocessing.scale(sged1.T)
        # scaled2 = preprocessing.minmax_scale(sged1)
        # samples = sged2

        ##################to show the effect of preprocess, use below

        # plt.subplot(2, 2, 1)
        # plt.plot(items)
        # plt.title('0-origin')
        # plt.subplot(2, 2, 2)
        # plt.plot(de_extremed)
        # plt.title('1-de_extremed')
        # plt.subplot(2, 2, 3)
        # plt.plot(sged1)
        # plt.title('2-SG-filtered')
        # plt.subplot(2, 2, 4)
        # plt.plot(scaled2)
        # plt.title('3-scaled')
        # plt.show()

        return samples


if __name__ == '__main__':
    path = 'D:\\23SeniorSemester2\\小论文\\Br\\Question-塑料分析'

    br = os.path.join(path, 'BR.xlsx')
    files = os.path.join(path, 'Data3\\br')
    brdata = Brset(specroot=files, labelroot=br, mode='')
    trainset = brdata
    total_samples = len(trainset)
    test_size = int(0.2 * total_samples)
    train_size = total_samples - test_size

    train_dataset, test_dataset = random_split(trainset, [train_size, test_size])
    print(brdata.__len__())
    print(train_dataset.__len__())
    print(test_dataset.__len__())
