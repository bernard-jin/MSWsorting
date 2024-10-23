import os
import pickle
import warnings

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import visdom
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from br3train.setb import Brset
from modelclasses.dualnet import MLCP

warnings.filterwarnings("ignore", category=UserWarning)

# dataset configure

# if omen, path
# path = 'D:\\23SeniorSemester2\\小论文\\Br\\Question-塑料分析'
# br = os.path.join(path, 'BR.xlsx')
# files = os.path.join(path, 'Data3\\br')
# set = Brset(specroot=files, labelroot=br, mode='kf-train')

# if ubuntu, ubuntu path
path = '/home/jzq/PycharmProjects/Br/Question-塑料分析'
core_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/pkls/pecnn'
br = os.path.join(path, 'BR.xlsx')
files = os.path.join(path, 'Data3/br')
# files = os.path.join(path, 'Data3/br4code')
set = Brset(specroot=files, labelroot=br, mode='kf-train')

# activate visdom, cuda
device = torch.device('cuda:0')
viz = visdom.Visdom()


def tfone_hot(t_tf):
    t = t_tf.long()
    t2 = F.one_hot(t, num_classes=2).to(torch.float32)
    return t2


def normalize(x):
    y = scale(x, axis=1)
    return y


# get a pca module
n_compo = 80
pca = PCA(n_components=n_compo)
pca.fit(set.specs)
# save the pca module
pca_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/pkls/pca'
with open(os.path.join(pca_path, 'pca{}.pkl'.format(n_compo)), 'wb') as f:
    pickle.dump(pca, f)
    f.close()
    print('pca created!!')

# get a encoder module:
# encoder_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/pkls/coder/encoder3e-05.pkl'
# encoder = torch.load(encoder_path).eval().to('cpu')

# get a vae model:
encoder_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/pkls/coder/encoder183.09442138671875.pkl'
encoder = torch.load(encoder_path).eval().to('cpu')


def encode(input):
    m, logvar = encoder.encode(input)
    y = encoder.reprameterize(m, logvar)
    return y


for tol in [250]:

    viz.close()
    width, height = 600, 400
    label_rate = 1e3
    # tol = 200  # 200kppm below/above is low/high
    kf = KFold(n_splits=7, shuffle=False)

    for fold, (train_indices, val_indices) in enumerate(kf.split(set)):
        train_dataset = Subset(set, train_indices)
        test_dataset = Subset(set, val_indices)

        # 分别创建训练集和测试集的 DataLoader 对象
        trainer = DataLoader(train_dataset, shuffle=True, batch_size=1024)
        tester = DataLoader(test_dataset, shuffle=False, batch_size=2000)

        vacc_max = 0.5
        peak_flag = False
        global_step = 0  # 无视这个，纯粹为了viz当横坐标的
        epochs = 30
        headsize = [1, 256, 128, 64]
        lr = 1e-3

        network = MLCP(sizes=headsize, drops=[0.1, 0.1, 0.1, 0.1], out=2)
        network.to(device)
        optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=1e-3)

        # loss function
        # Br ~ [0, 10.8],so make label / 11 as criteria label, evaluate through
        # say batch is 12, then half-batch is 6, then label is 12*1, net output is 6*2,
        # then resize label into [6*2] will do the work
        # w = torch.tensor([2, 1]).to(device)
        # w = torch.tensor([1, 1.2]).to(device)
        criterion = nn.MSELoss()
        # criterion = nn.CrossEntropyLoss(weight=w)

        for epoch in range(epochs):
            for idx, [x, label] in enumerate(trainer):

                # 1 inputting
                x1 = pca.transform(x)  # tensor -> array
                # x2 = encoder.encode(x).detach().numpy()   # tensor -> tensor, to array
                x2 = encode(x).detach().numpy()  # tensor -> tensor, to array, vae
                x1 = torch.from_numpy(normalize(x1))  # array to tensor
                x2 = torch.from_numpy(normalize(x2))  # array to tensor
                x = torch.cat([x2, x1], dim=-1)

                x = x.to(torch.float32).to(device)
                label = label.to(torch.float32).to(device)
                label = label / label_rate

                network.train()

                # 2 outputting
                # #2.1, if y is ,1)
                # y = network(x).view(-1)
                # y_class = y*label_rate > tol     # y ,1) to TF
                # y_class = y_class.view(y_class.shape[0])

                # ??? show me the input
                # print(x.shape)
                viz.line(Y=x[2, :], win="x", name='x', opts={'showlegend': True})

                # 2.2, if y is ,2)
                # print(torch.max(x), torch.min(x))
                y = network(x)
                # print(torch.max(y), torch.min(y),'max, min')
                y_class = y.argmax(axis=-1) > 0.5  # y ,2) to TF

                label_class = label > tol / label_rate  # label ,1) to TF
                acc_class = (y_class == label_class).sum().item() / y_class.shape[0]  # = vscore / vtotal
                confu = confusion_matrix(label_class.cpu(), y_class.cpu(), labels=[False, True])

                # 3 punishing and learning
                loss = criterion(y, tfone_hot(label_class))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # 4 validating
                vconfu = np.zeros([2, 2])
                with (torch.no_grad()):
                    # x_valid, label_valid = next(iter(tester))

                    # if valid through all v set
                    total_vscore, vscore = 0 + 1e-7, 0
                    for x_valid, label_valid in tester:
                        # 4.1 inputting
                        x1_valid = pca.transform(x_valid)  # tensor -> array
                        # x2_valid = encoder.encode(x_valid).detach().numpy()   # tensor -> tensor, to array
                        x2_valid = encode(x_valid).detach().numpy()  # tensor -> tensor, to array
                        x1_valid = torch.from_numpy(normalize(x1_valid))  # array to tensor
                        x2_valid = torch.from_numpy(normalize(x2_valid))  # array to tensor
                        x_valid = torch.cat([x1_valid, x2_valid], dim=-1)

                        x_valid = x_valid.to(torch.float32).to(device)
                        label_valid = label_valid.to(torch.float32).to(device)
                        label_valid = label_valid / label_rate
                        network.eval()

                        label_valid_class = label_valid > tol / label_rate  # label to T/F

                        # #4.2.1 if y is ,1)
                        # y_valid = network(x_valid).view(-1)   # outputting
                        # y_valid_class = y_valid*label_rate > tol     # y ,1) to TF
                        # y_valid_class = y_valid_class.view(y_valid_class.shape[0])
                        # loss_valid = criterion(y_valid, label_valid)

                        # 4.2.2 if y is ,2)
                        y_valid = network(x_valid)  # outputting
                        y_valid_class = y_valid.argmax(axis=-1) > 0.5  # y ,2) to TF
                        loss_valid = criterion(y_valid, tfone_hot(label_valid_class))

                        # acc of this v_batch
                        vscore += (y_valid_class == label_valid_class).sum().item()  # num of rights
                        total_vscore += y_valid_class.shape[0]  # num of totals
                        vconfu += confusion_matrix(label_valid_class.cpu(), y_valid_class.cpu(),
                                                   labels=[False, True])  # y_TF confuse with label_TF
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
                # log accuracy of y to label

                # acc = np.log((y[0].item() + 1e-5) / ((label / label_rate)[0].item() + 1e-5))
                # viz.line([acc], [global_step],
                #          update='append',
                #          win='log rate: y/label',
                #          name='Valid',
                #          opts={'title': 'log rate: y/label',
                #                'xlabel': 'global step',
                #                'ylabel': 'log rate: y/label',
                #                'showlegend': True})
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

                # reporting
                print('tol {} fold {} epoch {} idx {}, loss = {}'.format(tol, fold, epoch, idx, round(loss.item(), 4)))
                print(confu)
                print('valid confusion for this batch:\n', vconfu)
                print('train accuracy:', round(acc_class, 4))
                print('valid accuracy = {}/{} = {}'.format(round(vscore, 4), round(total_vscore, 4),
                                                           round(acc_valid_class, 4)))
                print('\n')

                # save some of the pkls
                if (epoch >= int(0.9 * epochs)) & (files.split('/')[-1] != 'br4code'):
                    if vacc_max < acc_valid_class:
                        vacc_max = acc_valid_class
                        torch.save(network, os.path.join(core_path,
                                                         'tol{}_F{}ep{}idx{}_T{}V{}.pkl'.format(tol / 1000, fold, epoch,
                                                                                                idx,
                                                                                                round(acc_class, 4),
                                                                                                round(vacc_max, 4))))
                        print('saving..')
                    else:
                        continue

    input('tol {}, awaiting..'.format(tol))
