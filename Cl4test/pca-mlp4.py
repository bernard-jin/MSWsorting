import csv
import os
import pickle

import numpy as np
import torch.optim
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from br3train.setb import Brset
import glob
device = torch.device('cuda:0')

# if ubuntu, ubuntu path
path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/'
# files = os.path.join(path, 'Data3/br')
# files = os.path.join(path, 'Data3/br4code')
files = os.path.join(path, 'Data3/HeFei/xlsx/NB_notcol')
br = os.path.join(path, 'BR.xlsx')
set = Brset(specroot=files, labelroot=br, mode='end')
ender = DataLoader(set, shuffle=True, batch_size=1000)

paths = '/home/jzq/PycharmProjects/Br/Question-塑料分析/br3test/rnets'

dates = os.listdir(paths)
for date in dates:
    csv_path = os.path.join(paths, date)
    core_path = csv_path
    # pkls = glob.glob(os.path.join(core_path, '**/*.pkl'), recursive=True)
    pkls = [pkl for pkl in os.listdir(core_path) if pkl.endswith('.pkl')]
    label_rate = 1e3
    # tol = 80
    # for divtol in [210, 220, 230, 240, 250, 260, 270, 280, 290]:

    for divtol in np.linspace(0.05,0.35,31):
        divtol = round(divtol*label_rate)
        tol = divtol
        rnet_dict = {}

        # if pca, then load pca
        n_compo, kernel = 80, 'rbf'
        pca_path = os.path.join(path, 'pkls/hefei/pca')
        pca_path = os.path.join(pca_path, '{}pca{}.pkl'.format(kernel, n_compo))
        with open(pca_path, 'rb') as f:
            pca = pickle.load(f)
            f.close()
            print('pca read successfully')

        for pkl in pkls:
            network = torch.load(os.path.join(core_path, pkl))
            print('{}'.format(pkl.split('/')[-1]))
            vscore = 0
            total_vscore = 0
            vconfu, global_confusion = np.zeros([2, 2]), np.zeros([2, 2])

            for x_valid, label_valid in ender:
                with torch.no_grad():
                    network.eval()

                    # 4.1 inputting
                    # x_valid = x_valid.to(torch.float32).to(device)
                    # label_valid = label_valid.to(torch.float32).to(device)
                    # label_valid = label_valid / label_rate

                    # 4.1.2 inputting and use pca
                    x_valid = torch.from_numpy(pca.transform(x_valid))
                    x_valid = x_valid.to(torch.float32).to(device)
                    label_valid = label_valid.to(torch.float32).to(device)
                    label_valid = label_valid / label_rate

                    # #4.2.1 if y is ,1)
                    # y_valid = network(x_valid).view(-1)
                    # y_valid_class = y_valid*label_rate > tol     # y ,1) to TF
                    # y_valid_class = y_valid_class.view(y_valid_class.shape[0])
                    # loss_valid = criterion(y_valid, label_valid)

                    label_valid_class = label_valid > tol / label_rate  # label to T/F
                    # 4.2.2 if y is ,2)
                    y_valid = network(x_valid)
                    y_valid_class = y_valid.argmax(axis=-1) > 0.5  # y ,2) to TF

                    # acc of this v_batch
                    vscore += (y_valid_class == label_valid_class).sum().item()  # num of rights
                    total_vscore += y_valid_class.shape[0]  # num of totals
                    vconfu += confusion_matrix(label_valid_class.cpu(), y_valid_class.cpu(), labels=[False, True])

            acc_valid_class = vscore / total_vscore
            # print(acc_valid_class)
            # # reporting
            # print('valid confusion for this batch:\n', vconfu)
            # print('valid accuracy = {}/{} = {}'.format(round(vscore, 4), round(total_vscore, 4), round(acc_valid_class, 4)))
            # print('\n')

            global_confusion = global_confusion + vconfu
            global_acc = (global_confusion[0, 0] + global_confusion[1, 1]) / sum(sum(global_confusion))
            # global_acc = accuracy_valid
            rnet_dict[pkl] = [round(global_acc, 4),
                              global_confusion[0, 0],
                              global_confusion[0, 1],
                              global_confusion[1, 0],
                              global_confusion[1, 1],
                              ]
            print(global_confusion, "global_confu", global_acc, '\n')

            maximum = max(rnet_dict.items(), key=lambda x: x[1][0])
            print(maximum[-1], '= max accuracy')
            print(rnet_dict)

            # csv_name = pkl.split('/')[-2]
            csv_name = date
            with open(os.path.join(csv_path, '{}_tol{}_div{}.csv'.format(csv_name, tol, divtol)), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['pkl_name', 'v', 't', 'q', '0-0', '0-1', '1-0', '1-1'])
                for pkl in rnet_dict.keys():
                    pkl_name = pkl.split('/')[-1]
                    vt = pkl.split('_T')[-1].split('.pkl')[0]
                    t, v = vt.split('V')
                    writer.writerow([pkl_name, v, t, rnet_dict[pkl][0],
                                     rnet_dict[pkl][1],
                                     rnet_dict[pkl][2],
                                     rnet_dict[pkl][3],
                                     rnet_dict[pkl][4],
                                     ])
                f.close()