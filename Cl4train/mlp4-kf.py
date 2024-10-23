import torch.optim
import visdom
from torch.utils.data import DataLoader
from torch.utils.data import Subset, ConcatDataset
import os
from setc import Clset
from modelclasses.dualnet import MLP
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
import numpy as np
import torch.nn.functional as F
import time
import contextlib

# dataset configure

# if omen, path
# path = 'D:\\23SeniorSemester2\\小论文\\Br\\Question-塑料分析'
# br = os.path.join(path, 'BR.xlsx')
# files = os.path.join(path, 'Data3\\br')
# set = Brset(specroot=files, labelroot=br, mode='kf-train')

# if ubuntu, ubuntu path
# path = '/home/jzq/PycharmProjects/Br/Question-塑料分析'
# core_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/pkls/cnn3'
# files = os.path.join(path, 'Data3/br')
# files = os.path.join(path, 'Data3/br4code')
# files = os.path.join(path, 'Data3/br4kernel')

# if hefei cl
path = '/home/jzq/PycharmProjects/Br/Question-塑料分析'
core_path = os.path.join(path, 'pkls/cnn3/cl')
files = os.path.join(path, 'Data3/HeFei/TXxlsx/TXNB_processed')
cl = os.path.join(path, 'Cl4train/Cls.xlsx')
set1 = Clset(specroot=files, labelroot=cl, mode='kf-train')
# # combined_set = set1

# '''
# 24-07-02:实例化得到的数据集才1362条光谱，显然不够。要把830件当中有Cl的拎进来
# '''
path = '/home/jzq/PycharmProjects/Br/Question-塑料分析'
files = os.path.join(path, 'Data3/HeFei/xlsx/NB_notcol')
br = os.path.join(path, 'BR.xlsx')
set2 = Clset(specroot=files, labelroot=br, mode='kf-train')
# combined_set = set2
combined_set = ConcatDataset([set1, set2])


# activate visdom, cuda
device = torch.device('cuda:0')
viz = visdom.Visdom()
width, height = 600, 400
label_rate = 1e3
out=2

def tfone_hot(t_tf):
    t = t_tf.long()
    t2 = F.one_hot(t, num_classes=2).to(torch.float32)
    return t2

for tol in [0.05]:
    core_path = os.path.join(path, 'pkls/hefei/Cl/mlp4')
    viz.close()
    # tol = 200  # 200kppm below/above is low/high
    kf = KFold(n_splits=7, shuffle=False)

    # 24-07-21-13-48: 这是为了给即将到来的pkl们申请文件夹
    timetitle = str(list(time.localtime())[:6])+'{}'.format(tol)
    if not os.path.exists(core_path):
        os.mkdir(core_path)
    core_path = os.path.join(core_path, timetitle)
    if not os.path.exists(core_path):
        os.mkdir(core_path)

    for fold, (train_indices, val_indices) in enumerate(kf.split(combined_set)):
        train_dataset = Subset(combined_set, train_indices)
        test_dataset = Subset(combined_set, val_indices)

        # 分别创建训练集和测试集的 DataLoader 对象
        # trainer = DataLoader(train_dataset, shuffle=True, batch_size=784) # got 70%

        trainer = DataLoader(train_dataset, shuffle=True, batch_size=128)
        tester = DataLoader(test_dataset, shuffle=True, batch_size=2000)
        #
        # # original mlp configs
        # headsize = [255, 1024]
        # network1 = MLP(sizes=headsize, drops=[0.2, 0.2])

        # original mlcp configs
        # headsize = [1, 256, 128, 64]
        # network2 = MLCP(sizes=headsize, drops=[0.1, 0.1, 0.1], out=2)

        # mlp configs
        headsize = [195, 2048, 2048, 512, 2]
        network1 = MLP(sizes=headsize, drops=[0.2, 0.2])
        #
        vacc_max = 0.5
        global_step = 0  # 无视这个，纯粹为了viz当横坐标的
        epochs = 80
        lr = 2e-4
        # network = nn.Sequential(network1,
        #                         network2)
        network = network1
        network.to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=lr)

        # loss function
        # Br ~ [0, 10.8],so make label / 11 as criteria label, evaluate through
        # say batch is 12, then half-batch is 6, then label is 12*1, net output is 6*2,
        # then resize label into [6*2] will do the work
        # w = torch.tensor([1, 1]).to(device)
        criterion = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss(weight=w)

        for epoch in range(epochs):
            for idx, [x, label] in enumerate(trainer):
                # 1 inputting
                x = x.to(torch.float32).to(device)
                label = label.to(torch.float32).to(device)  # (kppm)
                label = label / label_rate  # (*100%)
                network.train()

                # 2 outputting

                # #2.1, if y is ,1)
                if out == 1:
                    y = network(x).view(-1)
                    # y = y*label_rate
                    y_class = y > tol     # y ,1) to TF
                    y_class = y_class.view(y_class.shape[0])

                elif out == 2:
                    # 2.2, if y is ,2)
                    y = network(x)
                    y_class = y.argmax(axis=-1) > 0.5  # y ,2) to TF

                label_class = label > tol  # label ,1) to TF
                score, length = (y_class == label_class).sum().item(), y_class.shape[0]
                acc_class = score / length  # = score / total
                confu = confusion_matrix(label_class.cpu(), y_class.cpu(), labels=[False, True])

                # 3 punishing and learning
                if out == 2:
                    loss = criterion(y, tfone_hot(label_class))  # 3.1
                elif out == 1:
                    loss = criterion(y, label)  # 3.2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 4 validating
                vconfu = np.zeros([2, 2])
                with (torch.no_grad()):
                    # x_valid, label_valid = next(iter(tester))

                    # if valid through all v set
                    total_vscore, vscore = 0+1e-7, 0
                    for x_valid, label_valid in tester:
                        # 4.1 inputting
                        x_valid = x_valid.to(torch.float32).to(device)
                        label_valid = label_valid.to(torch.float32).to(device)
                        network.eval()

                        label_valid_class = label_valid > tol*label_rate  # label to T/F

                        # #4.2.1 if y is ,1)
                        if out == 1:
                            y_valid = network(x_valid).view(-1)   # outputting

                            # y_valid = y_valid*label_rate
                            label_valid = label_valid / label_rate
                            y_valid_class = y_valid > tol     # y ,1) to TF
                            y_valid_class = y_valid_class.view(y_valid_class.shape[0])
                            loss_valid = criterion(y_valid, label_valid)

                        # 4.2.2 if y is ,2)
                        elif out == 2:
                            y_valid = network(x_valid)   # outputting
                            y_valid_class = y_valid.argmax(axis=-1) > 0.5   # y ,2) to TF
                            loss_valid = criterion(y_valid, tfone_hot(label_valid_class))

                        # acc of this v_batch
                        vscore += (y_valid_class == label_valid_class).sum().item()  # num of rights
                        total_vscore += y_valid_class.shape[0]  # num of totals
                        vconfu += confusion_matrix(label_valid_class.cpu(), y_valid_class.cpu(),
                                                   labels=[False, True])   # y_TF confuse with label_TF

                acc_valid_class = vscore / total_vscore
                # with loss.cpu(), loss_valid.cpu():
                global_step = global_step + 1
                visloss = loss.cpu().detach().numpy()
                viz.line([visloss], [global_step],
                         update='append',
                         win='loss_train',
                         name='fold{}'.format(fold),
                         opts={'title': 'train losses',
                               'xlabel': 'global step',
                               'ylabel': 'loss',
                               'showlegend': True,
                               'width': width,
                               'height': height,
                               })
                visloss_valid = loss_valid.cpu().detach().numpy()
                viz.line([visloss_valid], [global_step],
                         update='append',
                         win='loss_valid',
                         name='fold{}'.format(fold),
                         opts={'title': 'valid losses',
                               'xlabel': 'global step',
                               'ylabel': 'accuracy',
                               'showlegend': True,
                               'width': width,
                               'height': height
                               })

                viz.line([acc_class], [global_step],
                         update='append',
                         win='Train Accuracy',
                         name='fold{}'.format(fold),
                         opts={'title': 'Train Accuracy',
                               'xlabel': 'global step',
                               'ylabel': 'accuracy',
                               'showlegend': True,
                               'width': width,
                               'height': height,
                               })
                viz.line([acc_valid_class], [global_step],
                         update='append',
                         win='Valid Accuracy',
                         name='fold{}'.format(fold),
                         opts={'title': 'Valid Accuracy',
                             'xlabel': 'global step',
                               'ylabel': 'accuracy',
                               'showlegend': True,
                               'width': width,
                               'height': height,
                               })

                # # 2024-07-21-14-06: if F1 score, refracted
                precision = vconfu[0, 0] / (vconfu[0, 1] + vconfu[0, 0])
                recall = vconfu[1, 1] / (vconfu[1, 0] + vconfu[1, 1])
                F1 = 2 * (precision * recall) / (precision + recall)
                viz.line([F1], [global_step],
                         update='append',
                         win='F1 score',
                         name='fold{}'.format(fold),
                         opts={'title': 'F1 score',
                               'xlabel': 'global step',
                               'ylabel': 'F1 score',
                               'showlegend': True})

                # reporting
                print('tol {} fold {} epoch {} idx {}, loss = {}'.format(tol, fold, epoch, idx, round(loss.item(), 4)))
                print(confu)
                print('valid confusion for this batch:\n', vconfu)
                print('train accuracy = {}/{} = {}'.format(round(score, 4), length, round(acc_class, 4)))
                print('valid accuracy = {}/{} = {}'.format(round(vscore, 4), round(total_vscore, 4),
                                                           round(acc_valid_class, 4)))
                print('\n')
                # if idx % 10 == 0:
                #     flag = input('continue epoch?')
                #     if flag == 'n':
                #         epoch = epochs
                # save some of the pkls
                if epoch >= int(0.9 * epochs):
                    if vacc_max < acc_valid_class:
                        vacc_max = acc_valid_class
                        torch.save(network, os.path.join(core_path,
                                                         'tol{}_F{}ep{}idx{}_T{}V{}.pkl'.format(tol, fold, epoch,
                                                                                                idx,
                                                                                                round(acc_class, 4),
                                                                                                round(vacc_max, 4))))
                        print('saving..')
                    else:
                        continue

    # #  24-07-02-21-40:  计划把训练过程中的所有sysprint输出到一个txt文件当中以方便后续研究
    output_filename = 'parameters.txt'
    with open(os.path.join(core_path, output_filename), 'w') as output_file:
        with contextlib.redirect_stdout(output_file):
            params = {}
            params['datascale'] = combined_set.__len__()
            params['tol'] = tol
            params['positive'] = sum(a >= tol * label_rate for a in set1.labels) + sum(
                a >= tol * label_rate for a in set2.labels)
            params['negative'] = sum(a < tol * label_rate for a in set1.labels) + sum(
                a < tol * label_rate for a in set2.labels)
            print(kf)
            print(params)
            print(network)
            print(criterion)  # mse

    # read this code file
    py_path = __file__
    with open(py_path, 'r', encoding='utf-8') as f:
        scripts_train = f.read()
    # save it to txt
    py_txtpath = os.path.join(core_path, 'train.txt')
    with open(py_txtpath, 'w', encoding='utf-8') as f:
        f.write(scripts_train)

    # read setb code file
    pyset_path = os.path.join(path, 'Cl4train/setc.py')
    with open(pyset_path, 'r', encoding='utf-8') as f:
        scripts_set = f.read()
    # save it to txt
    pyset_txtpath = os.path.join(core_path, 'set.txt')
    with open(pyset_txtpath, 'w', encoding='utf-8') as f:
        f.write(scripts_set)

    print(timetitle)
    input('tol {}, awaiting..'.format(tol))



