import os
import time

import numpy as np
import torch.optim
import visdom
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from modelclasses.dualnet import Siam_net
from setb import Brset

# if ubuntu, ubuntu path
path = '/home/jzq/PycharmProjects/Br/Question-塑料分析'
core_path = os.path.join(path, 'pkls/pca-cnn3')
br = os.path.join(path, 'BR.xlsx')
# files = os.path.join(path, 'Data3/br')
files = os.path.join(path, 'Data3/br4code')
# files = os.path.join(path, 'Data3/br4kernel')
set = Brset(specroot=files, labelroot=br, mode='kf-train')

'''
# activate visdom, cuda
device = torch.device('cuda:0')
viz = visdom.Visdom()
width, height = 600, 400
label_rate = 1e3

def tfone_hot(t_tf):
    t = t_tf.long()
    t2 = F.one_hot(t, num_classes=2).to(torch.float32)
    return t2

def comparinglabels(label, label_rate):
    x1, x2 = torch.chunk(label / label_rate, 2, dim=0)
    label_output = torch.cat([x1.view(-1, 1), x2.view(-1, 1)], dim=-1)
    return label_output

def labeldist(label):
    label = label.reshape([label.shape[0], 1])
    l1, l2 = torch.chunk(label, 2, dim=0)
    new_label = torch.abs(l1 - l2)
    # new_label = torch.cat([l1, l2], dim=1)
    # print(new_label.shape)
    return new_label

# tol = 200  # 200kppm below/above is low/high
for tol in [250]:

    viz.close()
    kf = KFold(n_splits=7, shuffle=False)

    for fold, (train_indices, val_indices) in enumerate(kf.split(set)):
        train_dataset = Subset(set, train_indices)
        test_dataset = Subset(set, val_indices)

        # 分别创建训练集和测试集的 DataLoader 对象
        trainer = DataLoader(train_dataset, shuffle=True, batch_size=1024)
        tester = DataLoader(test_dataset, shuffle=True, batch_size=2000)

        epochs = 200
        headsize = [1, 32, 64, 128]
        headout = 10
        tailsize = [headout, 2]
        lr = 1e-4

        network = Siam_net(heads_size=headsize, heads_out=headout, tail_size=tailsize)
        network.to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)

        # loss function
        # Br ~ [0, 10.8],so make label / 11 as criteria label, evaluate through
        # say batch is 12, then half-batch is 6, then label is 12*1, net output is 6*2,
        # then resize label into [6*2] will do the work
        criterion = nn.MSELoss()

        for epoch in range(epochs):
            for idx, [x, label] in enumerate(trainer):

                # 每个batch对半开，但如果奇数
                if x.shape[0] % 2:
                    x = x[:-1]
                    label = label[:-1]

                # 1 inputting
                x = x.to(torch.float32).to(device)
                label = label.to(torch.float32).to(device)
                network.train()
                y = network(x)

                # 2 label processing
                label_contrast = comparinglabels(label, label_rate)

                # 3 punishing and learning
                loss = criterion(y, label_contrast)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 4 validating
                vconfu = np.zeros([2, 2])
                with (torch.no_grad()):
                    # if valid through all v set
                    total_vscore, vscore = 0+1e-7, 0
                    for x_valid, label_valid in tester:

                        # 每个batch对半开，但如果奇数
                        if x_valid.shape[0] % 2:
                            x_valid = x_valid[:-1]
                            label_valid = label_valid[:-1]

                        # 4.1 inputting
                        x_valid = x_valid.to(torch.float32).to(device)
                        label_valid = label_valid.to(torch.float32).to(device)
                        network.eval()
                        y_valid = network(x_valid)

                        # 4.2 label processing
                        label_contrast_valid = comparinglabels(label_valid, label_rate)
                        loss_valid = criterion(y_valid, label_contrast_valid)  # loss valid

'''

device = torch.device('cuda:0')
viz = visdom.Visdom()
viz.close()

# k-fold
num_folds = 5  # 设置k-折参数
EPOCHS = 25
headsize = [1, 64, 128, 128]
headout = 336
# tailsize = [headout, 2]
tailsize, tol = [headout, 1], 1
drops = [0.2, 0.2, 0.2, 0.2]
batchsize = 1024
valsize = 1000
subepochs = 40

# 交叉验证大循环
kf = KFold(n_splits=num_folds, shuffle=False)  # 创建k折交叉验证对象
for fold, (train_indices, val_indices) in enumerate(kf.split(set)):
    train_data = Subset(set, train_indices)
    val_data = Subset(set, val_indices)

    trainer = DataLoader(train_data, batch_size=batchsize, shuffle=True, num_workers=8)
    # trainer = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0)
    # trainer = DataLoader(train_data, batch_size=256, shuffle=True, num_workers=0)
    tester = DataLoader(val_data, batch_size=valsize, shuffle=True, num_workers=8)

    network = Siam_net(heads_size=headsize, drops_prob=drops, heads_out=headout, tail_size=tailsize)  # 实例化resnet网络
    # l1_lambda = 1e-1
    # l1_lambda = 1e-2

    l1_lambda = 1e-4
    # l1_lambda = 0
    optimizer = torch.optim.Adam(network.parameters(), lr=1e-3, weight_decay=l1_lambda)  # 装载优化器

    mynetname = type(network).__name__  # 记录名字这次跑用的什么网络，后面写结果要用
    myprocessname = '{}_fold{}_decay{}'.format(
        mynetname, fold,
        l1_lambda)  # 记录名字这次跑用的什么网络，后面写结果要用

    network.to(device)  # 搬运到显卡上

    # 这是为了给即将到来的pkl们申请文件夹
    timetitle = str(list(time.localtime())[:6])
    net_location1 = '/home/jzq/PycharmProjects/Br/Question-塑料分析/pkls/siam' + '/'
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
    # resnet训练过程
    for epoch in range(EPOCHS):  # range轮学习
        global_confusion = np.zeros([2, 2])  # 在本epoch内，初始化全局混淆矩阵，全局 = 这一epoch当中所有batchidx局部之和
        # for batchidx, [x, label] in enumerate(trainer):  # 每一轮都过一遍全部训练集，一个batch一个batch地过[x=>batch]
        for sub_epoch in range(subepochs):
            batchidx = sub_epoch
            dataset = trainer.dataset
            rand_indices = random.sample(range(len(dataset)), batchsize)

            x = torch.stack([dataset[i][0] for i in rand_indices])
            label = torch.tensor([dataset[i][1] for i in rand_indices])

            # 多模态，原始数据并联pca
            x1 = pca.transform(x.reshape([x.shape[0], x.shape[-1]]))  # 输出的是array
            x1 = torch.tensor(x1).reshape([x1.shape[0], 1, x1.shape[-1]])  # 把格式转回tensor
            x1 = x1.to(device)
            # 多模态，原始数据并联enc
            with torch.no_grad():
                x = x.to(torch.float32).to(device)  # 测试的时候发现x必须转成float32格式才能进入网络
                x2 = encoder1(x)
            # 并联
            x2 = x2.reshape(x2.shape[0], 1, x2.shape[-1])
            x = torch.cat([x, x1, x2], dim=-1)

            # 搬运到显卡上
            x = x.to(torch.float32).to(device)  # 测试的时候发现x必须转成float32格式才能进入网络
            # label = losses.labelreshape(label).to(torch.float32).to(device)  # 测试的时候发现x必须转成float32格式才能进入网络
            label = losses.labeldist(label).to(torch.float32).to(device)

            # 进行预测行为，计算预测与label的loss
            network.train()
            y = network(x)
            label_fuzz = label
            loss = criterion(y, label_fuzz)

            # 回传
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 训练集上的表现？
            # prediction_train = y.argmax(dim=1)  # 根据输出的概率比拼，胜出的当作【0/1】预测结果
            # label_fuzz_pred = label_fuzz.argmax(dim=1)  # 根据输出的概率比拼，胜出的当作【0/1】预测结果
            # acca = int(sum(prediction_train == label_fuzz_pred))
            prediction_train = y  # 根据输出的概率比拼，胜出的当作【0/1】预测结果
            label_fuzz_pred = label_fuzz
            temp_mat = (y >= label - tol) & (y < label + tol)
            # temp_mat = (y + 1e-5) / (label + 1e-5)
            acca = int(temp_mat.sum())

            accb = int(label_fuzz_pred.shape[0])
            accuracy_train = acca / accb

            # a = prediction_train.cpu()
            # b = label_fuzz_pred.cpu()
            # train_confusion = confusion_matrix(b, a, labels=[0, 1])

            # 测试集上的表现？
            local_confusion = np.zeros([2, 2])
            with torch.no_grad():
                # [x_valid, label_valid] = next(iter(tester))  # pop出一对test样本
                for _, [x_valid, label_valid] in enumerate(tester):
                    if x_valid.shape[0] % 2:
                        x_valid = x_valid[:-1]
                        label_valid = label_valid[:-1]
                    # 多模态，原始数据并联pca
                    x1_valid = pca.transform(x_valid.reshape([x_valid.shape[0], x_valid.shape[-1]]))  # 输出的是array
                    x1_valid = torch.tensor(x1_valid).reshape([x1_valid.shape[0], 1, x1_valid.shape[-1]])  # 把格式转回tensor
                    x1_valid = x1_valid.to(device)
                    # 多模态，原始数据并联enc
                    with torch.no_grad():
                        x_valid = x_valid.to(torch.float32).to(device)  # 测试的时候发现x必须转成float32格式才能进入网络
                        x2_valid = encoder1(x_valid)
                    # 并联
                    x2_valid = x2_valid.reshape(x2_valid.shape[0], 1, x2_valid.shape[-1])
                    x_valid = torch.cat([x_valid, x1_valid, x2_valid], dim=-1)
                    # x_valid = torch.cat([x1_valid, x2_valid], dim=-1)

                    # 搬运到显卡上
                    x_valid = x_valid.to(torch.float32)  # 测试的时候发现x必须转成float32格式才能进入网络
                    # label_valid = losses.labelreshape(label_valid).to(torch.float32).to(device)
                    label_valid = losses.labeldist(label_valid).to(torch.float32).to(device)
                    label_valid_fuzz = label_valid

                    # 输出预测，loss
                    network.eval()
                    y_valid = network(x_valid)  # 模型做测试
                    loss_valid = criterion(y_valid, label_valid_fuzz)

                    # 测试集上的表现？
                    # prediction_valid = y_valid.argmax(dim=1)  # 根据输出的概率比拼，胜出的当作【0/1】预测结果
                    # label_valid_fuzz_pred = label_valid_fuzz.argmax(dim=1)

                    prediction_valid = y_valid  # 根据输出的概率比拼，胜出的当作【0/1】预测结果
                    label_valid_fuzz_pred = label_valid_fuzz
                    temp_mat_valid = (y_valid >= label_valid - tol) & (y_valid < label_valid + tol)
                    # acca_valid = int(sum(prediction_valid == label_valid_fuzz_pred))
                    # accb_valid = int(label_valid_fuzz_pred.shape[0])
                    # accuracy_valid = acca_valid / accb_valid
                    # acc_valid_for_rec.append(accuracy_valid)

                    # 用全体 tester做一次随堂考试
                    # a = prediction_valid.cpu()
                    # b = label_valid_fuzz_pred.cpu()
                    # local_confusion = local_confusion + confusion_matrix(b, a, labels=[0, 1])
                    '''
                          | 预测为正类 | 预测为负类 |
                实际为正类 |    TP     |    FN     |
                实际为负类 |    FP     |    TN     |
                    '''

                # 验证集表现————正在结算分数
                # acca_valid = local_confusion[0, 0] + local_confusion[1, 1]
                # accb_valid = local_confusion.sum()
                # accuracy_valid = acca_valid / accb_valid  # 计算随堂考试的总成绩
                acca_valid = int(temp_mat_valid.sum())
                accb_valid = int(label_valid_fuzz_pred.shape[0])
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

            if (epoch >= 0.3 * EPOCHS) & (accuracy_valid > accuracy_valid_max):
                # 如果创下新纪录则保存网络
                peak_flag = True
                accuracy_valid_max = accuracy_valid
                trainacc_max = accuracy_train
                bestnet = copy.deepcopy(network)
                bestepoch = epoch
                best_batchidx = batchidx

        # if (epoch >= 0.7 * EPOCHS) & (epoch % 2):

        if (epoch >= 0.6 * EPOCHS):
            # 从某个epoch开始，每隔一些个epoch，保存本轮训练最终结果
            epochnet = copy.deepcopy(network)
            net_item = net_location + 'V{}T{}_F{}Ep{}Idx{}.pkl'.format(round(accuracy_valid, 3),
                                                                       round(accuracy_train, 3), fold, epoch, -1)
            torch.save(epochnet, net_item)

        # 每过1个epoch，吐出一个混淆矩阵
        print("global confusion matrix for now:\n", global_confusion)

        if peak_flag:
            # 如果新纪录则将网络导出来
            bestnet_item = net_location + 'V{}T{}_F{}Ep{}Idx{}.pkl'.format(round(accuracy_valid_max, 3),
                                                                           round(trainacc_max, 3), fold, bestepoch,
                                                                           best_batchidx)
            torch.save(bestnet, bestnet_item)
            peak_flag = False

    # 如果启用本行以及最下面那一行，需要把中间的内容tab+
    # # 计划把训练过程中的所有sysprint输出到一个txt文件当中以方便后续研究
    output_filename = 'print.txt'
    with open(os.path.join(net_location, output_filename), 'w') as output_file:
        with contextlib.redirect_stdout(output_file):
            print('encoder:   ', encoder1)
            print('pca    ', pca)
            print(network)
            print(criterion)  # mse
            print(" with weight", label_weight)  # mse
            # print(criterion, label_weight)  # ce with weight
            # print(criterion, "gamma={},alpha={}".format(gamma, alpha))      # focal
            print('fuzz:\n')
            # print(mius)
            # print(crosspoint)
            print("global confusion:\n", global_confusion)

# output_file.close()
