import os

import numpy as np
import torch.nn as nn
import torch.optim
import visdom
from torch.utils.data import DataLoader, random_split

from modelclasses.dualnet import MLCP
from setb import Brset

# dataset configure

path = 'D:\\23SeniorSemester2\\小论文\\Br\\Question-塑料分析'

br = os.path.join(path, 'BR.xlsx')
# files = os.path.join(path, 'Data3\\br')
files = os.path.join(path, 'Data3\\br')
trainset = Brset(specroot=files, labelroot=br, mode='')
# testset = Brset(specroot=files, labelroot=br, mode='test')
# 假设你想将数据集划分为训练集和测试集，测试集占比为 20%，剩余部分为训练集
# 计算测试集和训练集的样本数量
total_samples = len(trainset)
test_size = int(0.2 * total_samples)
train_size = total_samples - test_size

# 使用 random_split 函数划分数据集
train_dataset, test_dataset = random_split(trainset, [train_size, test_size])

# 分别创建训练集和测试集的 DataLoader 对象
trainer = DataLoader(train_dataset, shuffle=True, batch_size=500)
tester = DataLoader(test_dataset, shuffle=False, batch_size=500)

# activate visdom, cuda
device = torch.device('cuda:0')
viz = visdom.Visdom()
viz.close()

epochs = 200
headsize = [1, 128, 128, 64, 32]
lr = 1e-4

network = MLCP(sizes=headsize, drops=[0.2, 0.2, 0.2, 0.2], out=1)
network.to(device)
optimizer = torch.optim.Adam(network.parameters(), lr=lr)

# loss function
# Br ~ [0, 10.8],so make label / 11 as criteria label, evaluate through
# say batch is 12, then half-batch is 6, then label is 12*1, net output is 6*2,
# then resize label into [6*2] will do the work
criterion = nn.MSELoss()
global_step = 0  # 无视这个，纯粹为了viz当横坐标的
width, height = 550, 400
label_rate = 1e3
tol = 200

for epoch in range(epochs):
    for idx, [x, label] in enumerate(trainer):
        # 1 inputting
        x = x.to(torch.float32).to(device)
        label = label.to(torch.float32).to(device)
        # network.train()
        # outputting
        y = network(x)

        # 2 punishing and learning
        loss = criterion(y, label / label_rate)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 4 validating
        with torch.no_grad():
            # x_valid, label_valid = next(iter(tester))
            for x_valid, label_valid in tester:
                # 4.1 inputting
                x_valid = x_valid.to(torch.float32).to(device)
                label_valid = label_valid.to(torch.float32).to(device)
                # network.eval()
                y_valid = network(x_valid)

                # 4.2 label processing
                loss_valid = criterion(y_valid, label_valid / label_rate)

        # with loss.cpu(), loss_valid.cpu():
        visloss = loss.cpu().detach().numpy()
        visloss_valid = loss_valid.cpu().detach().numpy()
        global_step = global_step + 1
        viz.line([visloss], [global_step],
                 update='append',
                 win='loss',
                 name='Train',
                 opts={'title': 'losses',
                       'xlabel': 'global step',
                       'ylabel': 'loss',
                       'showlegend': True,
                       'width': width,
                       'height': height,
                       })
        viz.line([visloss_valid], [global_step],
                 update='append',
                 win='loss',
                 name='Valid',
                 opts={'title': 'losses',
                       'xlabel': 'global step',
                       'ylabel': 'accuracy',
                       'showlegend': True})

        acc = np.log((y[0].item() + 1e-5) / ((label / label_rate)[0].item() + 1e-5))
        viz.line([acc], [global_step],
                 update='append',
                 win='log rate: y/label',
                 name='Valid',
                 opts={'title': 'log rate: y/label',
                       'xlabel': 'global step',
                       'ylabel': 'log rate: y/label',
                       'showlegend': True})

        # train accuracy
        # y * label_rate vs label;  y vs label / label rate, tol == label
        y_class = y * label_rate > tol
        label_class = label > tol
        y_class = y_class.view(y_class.shape[0])
        acc_class = (y_class == label_class).sum().item() / y_class.shape[0]
        viz.line([acc_class], [global_step],
                 update='append',
                 win='accuracy',
                 name='Train',
                 opts={'title': 'accuracy',
                       'xlabel': 'global step',
                       'ylabel': 'accuracy',
                       'showlegend': True})

        y_valid_class = y_valid * label_rate > tol
        label_valid_class = label_valid > tol
        y_valid_class = y_valid_class.view(y_valid_class.shape[0])
        acc_valid_class = (y_valid_class == label_valid_class).sum().item() / y_valid_class.shape[0]
        viz.line([acc_valid_class], [global_step],
                 update='append',
                 win='accuracy',
                 name='Valid',
                 opts={'xlabel': 'global step',
                       'ylabel': 'accuracy',
                       'showlegend': True})

        # reporting
        print(x.shape, label.shape, label_valid.shape,
              "idx {} of epoch {}, train loss={}, valid loss={}".format(idx, epoch, loss, loss_valid))
