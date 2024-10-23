import contextlib
import copy
import os
import random
import time

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import visdom
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data import Subset

import losses
from modelclasses.dualnet import Siam_net
from setb import Brset

# if ubuntu, ubuntu path
# path = '/home/jzq/PycharmProjects/Br/Question-塑料分析'
# core_path = os.path.join(path, 'pkls/siam')
# br = os.path.join(path, 'BR.xlsx')  # 溴含量.xlsx的位置
# files = os.path.join(path, 'Data3/br')  # 所有光谱文件的位置
# # files = os.path.join(path, 'Data3/br4code')
# # files = os.path.join(path, 'Data3/br4kernel')
# set = Brset(specroot=files, labelroot=br, mode='kf-train')


# if hefei

path = '/home/jzq/PycharmProjects/Br/Question-塑料分析'
core_path = os.path.join(path, 'pkls/hefei/cnn3')
files = os.path.join(path, 'Data3/HeFei/xlsx/NB_notcol')

br = os.path.join(path, 'BR.xlsx')
set = Brset(specroot=files, labelroot=br, mode='kf-train')


device = torch.device('cuda:0')
viz = visdom.Visdom()
viz.close()

# 训练的参数配置
num_folds = 5  # 设置k-折参数
EPOCHS = 20  # 训练轮
batchsize = 2048
valsize = 1000
subepochs = 40
criterion = nn.MSELoss()
lr, l1_lambda = 1e-3, 1e-4

# 网络结构的参数配置
headsize = [1, 256, 256, 512]
headout = 10000
tailsize, tol = [headout, 2], 50
drops = [0.1, 0.1, 0.1, 0.1]

# 交叉验证大循环
kf = KFold(n_splits=num_folds, shuffle=False)  # 创建k折交叉验证对象
for fold, (train_indices, val_indices) in enumerate(kf.split(set)):

    trainer = DataLoader(Subset(set, train_indices), batch_size=batchsize, shuffle=True, num_workers=8)
    tester = DataLoader(Subset(set, val_indices), batch_size=valsize, shuffle=True, num_workers=8)

    network = Siam_net(heads_size=headsize, drops=drops, heads_out=headout, tail_size=tailsize).to(device)  # 实例化网络
    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=l1_lambda)  # 装载优化器

    mynetname = type(network).__name__  # 记录名字这次跑用的什么网络，后面写结果要用
    myprocessname = '{}_fold{}_decay{}'.format(mynetname, fold, l1_lambda)  # 记录名字这次跑用的什么网络，后面写结果要用

    # 这是为了给即将到来的pkl们申请文件夹
    timetitle = str(list(time.localtime())[:6])
    net_location1 = core_path + '/'
    net_location = net_location1 + timetitle + myprocessname + '/'
    if not os.path.exists(net_location1):
        os.mkdir(net_location1)
    if not os.path.exists(net_location):
        os.mkdir(net_location)

    # 迭代过程中随时监控表现，这些是各种实时指标的初始化
    accuracy_valid_max, trainacc_max, accuracy_train, bestepoch, best_batchidx, peak_flag = 0, 0, 0, 0, 0, False  # 无视这个
    bestnet = network  # 无视这个
    global_step = 0  # 无视这个，纯粹为了viz当横坐标的

    train_confusion = np.zeros([2, 2])  # 初始化全局混淆矩阵，全局 = 所有局部之和
    # net训练过程
    for epoch in range(EPOCHS):  # range轮学习
        global_confusion = np.zeros([2, 2])  # 在本epoch内，初始化全局混淆矩阵，全局 = 这一epoch当中所有batchidx局部之和
        # for batchidx, [x, label] in enumerate(trainer):  # 每一轮都过一遍全部训练集，一个batch一个batch地过[x=>batch]

        for sub_epoch in range(subepochs):
            batchidx = sub_epoch
            dataset = trainer.dataset
            rand_indices = random.sample(range(len(dataset)), batchsize)
            x = [torch.tensor(dataset[i][0]) for i in rand_indices]
            x = torch.stack(x)
            label = torch.tensor([dataset[i][1] for i in rand_indices])

            if x.shape[0] % 2:  # 万一某个batch从中间劈不开，就舍去尾巴上的一个样本
                x = x[:-1]
                label = label[:-1]

            # 搬运到显卡上
            x = x.to(torch.float32).to(device)  # 测试的时候发现x必须转成float32格式才能进入网络
            # label = losses.labelreshape(label).to(torch.float32).to(device)
            label = losses.labeldist(label).to(torch.float32).to(device)

            # 实验3 label 需要one_hot
            label_tf = (label > tol * torch.ones_like(label)).to(torch.int64)  # 根据输出的概率比拼，胜出的当作【0/1】预测结果
            label_1hot = F.one_hot(label_tf, num_classes=2).to(torch.float32)
            label_1hot = label_1hot.view(label_1hot.shape[0], -1)

            # 进行预测行为，计算预测与label的loss
            network.train()
            y = network(x)
            loss = criterion(y, label)

            # 回传
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 训练集上的表现？ # 训练集上的表现，比氯元素大小
            # prediction_train = y.argmax(dim=1)  # 根据输出的概率比拼，胜出的当作【0/1】预测结果
            # label_pred = label.argmax(dim=1)  # 根据输出的概率比拼，胜出的当作【0/1】预测结果
            # acca = int(sum(prediction_train == label_pred))
            # temp_mat = (y.reshape(y.shape[0]) >= label - tol) & (y.reshape(y.shape[0]) < label + tol)

            # # 训练集上的表现，单个元素输出预测
            # y = y.reshape(y.shape[0])
            # ratio = y+1e-5 / label+1e-5
            # temp_mat = torch.abs(ratio - 1) <= tol*1e-2

            # 训练集上的表现，距离大小分类判别
            prediction_train = y.argmax(dim=1)  # 根据输出的概率比拼，胜出的当作【0/1】预测结果
            label_pred = label_1hot.argmax(dim=1)
            acca = int(sum(prediction_train == label_pred))
            accb = int(label.shape[0])
            accuracy_train = acca / accb

            a = prediction_train.cpu()
            b = label_pred.cpu()
            train_confusion = confusion_matrix(b, a, labels=[0, 1])

            # 测试集上的表现？
            local_confusion = np.zeros([2, 2])
            with torch.no_grad():
                # [x_valid, label_valid] = next(iter(tester))  # pop出一对test样本
                for _, [x_valid, label_valid] in enumerate(tester):
                    if x_valid.shape[0] % 2:
                        x_valid = x_valid[:-1]
                        label_valid = label_valid[:-1]

                    # 搬运到显卡上
                    x_valid = x_valid.to(torch.float32).to(device)  # 测试的时候发现x必须转成float32格式才能进入网络
                    # label_valid = losses.labelreshape(label_valid).to(torch.float32).to(device)
                    label_valid = losses.labeldist(label_valid).to(torch.float32).to(device)
                    # label_valid = losses.labelnorm(label_valid).to(device)
                    # 实验3 label 需要one_hot
                    label_tf = label_valid > tol * torch.ones_like(label_valid)  # 根据输出的概率比拼，胜出的当作【0/1】预测结果
                    label_1hot = nn.functional.one_hot(label_tf.to(torch.int64), num_classes=2).to(torch.float32)
                    label_1hot = label_1hot.view(label_1hot.shape[0], -1)

                    # 输出预测，loss
                    network.eval()
                    y_valid = network(x_valid)  # 模型做测试
                    loss_valid = criterion(y_valid, label_1hot)

                    # 训练集上的表现，距离大小分类判别
                    prediction_valid = y_valid.argmax(dim=1)  # 根据输出的概率比拼，胜出的当作【0/1】预测结果
                    label_valid_pred = label_1hot.argmax(dim=1)
                    acca_valid = int(sum(prediction_valid == label_valid_pred))
                    accb_valid = int(label_valid_pred.shape[0])
                    accuracy_valid = acca_valid / accb_valid
                    # acc_valid_for_rec.append(accuracy_valid)

                    # 用全体 tester做一次随堂考试
                    a = prediction_valid.cpu()
                    b = label_valid_pred.cpu()
                    local_confusion = local_confusion + confusion_matrix(b, a, labels=[0, 1])
                    '''
                          | 预测为正类 | 预测为负类 |
                实际为正类 |    TP     |    FN     |
                实际为负类 |    FP     |    TN     |
                    '''

                    # 验证集表现————正在结算分数
                acca_valid = local_confusion[0, 0] + local_confusion[1, 1]
                accb_valid = local_confusion.sum()
                accuracy_valid = acca_valid / accb_valid

            # 在 viz上画图，分别是训练准确率，val准确率（全体val验证求平均），训练loss，val的loss
            global_step = global_step + 1
            viz.line([accuracy_train], [global_step],
                     update='append',
                     win='accuracy train',
                     name='fold{}'.format(fold),
                     opts={'title': 'trainingAcc',
                           'xlabel': 'global step',
                           'ylabel': 'accuracy',
                           'ylim': [0, 1],
                           'showlegend': True,
                           })
            viz.line([accuracy_valid], [global_step],
                     update='append',
                     win='accuracy valid',
                     name='fold{}'.format(fold),
                     opts={'title': 'ValidationAcc',
                           'xlabel': 'global step',
                           'ylabel': 'accuracy',
                           'ylim': [0, 1],
                           'showlegend': True})
            viz.line([loss.item()], [global_step],
                     update='append',
                     win='train loss',
                     name='fold{}'.format(fold),
                     opts={'title': 'TrainLoss',
                           'xlabel': 'global step',
                           'ylabel': 'loss',
                           'showlegend': True})
            viz.line([loss_valid.item()], [global_step],
                     update='append',
                     win='valid loss',
                     name='fold{}'.format(fold),
                     opts={'title': 'ValLoss',
                           'xlabel': 'global step',
                           'ylabel': 'loss',
                           'showlegend': True})

            # # if you want confusion matrix
            # tnfn = local_confusion[1, 1] + local_confusion[1, 0]
            # if tnfn == 0:
            #     tnr = 0
            # else:
            #     tnr = local_confusion[1, 1] / tnfn
            # fnr = local_confusion[0, 1] / (local_confusion[0, 1] + local_confusion[0, 0])
            # if (epoch >= 0.6 * EPOCHS):
            #     viz.scatter([fnr], [tnr],  # roc curve (for high)
            #              update='append',
            #              win='val roc',
            #              name='fold{}'.format(fold),
            #              opts={'title': 'val roc',
            #                    'showlegend': True,
            #                    'xlabel': 'fnr',
            #                    'ylabel': 'tnr',
            #                    'markersize': 4})

            # # 如果要plot tnr
            # viz.line([tnr], [global_step],
            #          update='append',
            #          win='val tnr',
            #          name='fold{}'.format(fold),
            #          opts={'title': 'val tnr',
            #                'showlegend': True})

            # 输出各项数据具体值
            print("fold:{},epoch:{},batch:{},loss:{}".format(fold, epoch, batchidx, loss))
            # print(train_confusion)
            # print("local confusion matrix of this batch:\n", local_confusion)  # 每过一个valid batch，吐出一个混淆矩阵
            print("train:{}/{}={}".format(acca, accb, accuracy_train),
                  "valid:{}/{}={}".format(acca_valid, accb_valid, accuracy_valid),
                  # "val_recall:{}".format(recall)
                  )
            print("\n")
            # global_confusion = global_confusion + local_confusion

            if (epoch >= 0.8 * EPOCHS) & (accuracy_valid > accuracy_valid_max):
                # 如果创下新纪录则保存网络
                peak_flag = True
                accuracy_valid_max = accuracy_valid
                trainacc_max = accuracy_train
                bestnet = copy.deepcopy(network)
                bestepoch = epoch
                best_batchidx = batchidx

        if epoch >= 0.6 * EPOCHS:
            # 从某个epoch开始，每隔一些个epoch，保存本轮训练最终结果
            epochnet = copy.deepcopy(network)
            net_item = net_location + 'V{}T{}_F{}Ep{}Idx{}.pkl'.format(round(accuracy_valid, 3),
                                                                       round(accuracy_train, 3), fold, epoch, -1)
            # torch.save(epochnet, net_item)

        # 每过1个epoch，吐出一个混淆矩阵
        print("global confusion matrix for now:\n", global_confusion)

        if peak_flag:
            # 如果新纪录则将网络导出来
            bestnet_item = net_location + 'V{}T{}_F{}Ep{}Idx{}.pkl'.format(round(accuracy_valid_max, 3),
                                                                           round(trainacc_max, 3), fold, bestepoch,
                                                                           best_batchidx)
            # torch.save(bestnet, bestnet_item)
            peak_flag = False

    # 如果启用本行以及最下面那一行，需要把中间的内容tab+
    # # 计划把训练过程中的所有sysprint输出到一个txt文件当中以方便后续研究
    output_filename = 'print.txt'
    with open(os.path.join(net_location, output_filename), 'w') as output_file:
        with contextlib.redirect_stdout(output_file):
            params = {}
            params['num_folds'] = 5  # 设置k-折参数
            params['EPOCHS'] = 40
            # params[headsize] = [1, 256, 256, 512]
            params['headout'] = 10000
            # tailsize = [headout, 2]
            # tailsize, tol = [headout, 1], 10
            params['divtol'] = 2
            # params[drops] = [0.1, 0.1, 0.1, 0.1]
            params["batchsize"] = 1024
            params['valsize'] = 400
            params['subepochs'] = 30

            print(params)
            print(network)
            print(criterion)  # mse
            print('fuzz:\n')
            # print(mius)
            # print(crosspoint)
            print("global confusion:\n", global_confusion)

# output_file.close()
