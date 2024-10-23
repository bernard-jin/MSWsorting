import csv
import os
import pickle

import numpy as np
import torch.optim
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader

from br3train.setb import Brset

# if ubuntu, ubuntu path
path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/'
files = os.path.join(path, 'Data3/br')
# files = os.path.join(path, 'Data3/br4code')
br = os.path.join(path, 'BR.xlsx')

set = Brset(specroot=files, labelroot=br, mode='end')
ender = DataLoader(set, shuffle=True, batch_size=1000)

core_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/br3test/rnets'
csv_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/br3test'
pkls = os.listdir(core_path)
device = torch.device('cuda:0')
label_rate = 1e3
tol = 250
rnet_dict = {}

# if pca, then load pca
pca_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/pkls/pca/pca80.pkl'
with open(pca_path, 'rb') as f:
    pca = pickle.load(f)
    f.close()
    print('pca read successfully')

# if vae
vae_path = ''
with open(vae_path, 'rb') as f:
    encoder = pickle.load(f)
    f.close()
    print('vae read successfully')


def encode(input):
    m, logvar = encoder.encode(input)
    y = encoder.reprameterize(m, logvar)
    return y


def normalize(x):
    y = scale(x, axis=1)
    return y


for pkl in pkls:
    network = torch.load(os.path.join(core_path, pkl))
    print('{}'.format(pkl.split('/')[-1]))
    vscore = 0
    total_vscore = 0
    vconfu, global_confusion = np.zeros([2, 2]), np.zeros([2, 2])

    for x_valid, label_valid in ender:
        with torch.no_grad():
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
    # global_acc = (global_confusion[0, 0] + global_confusion[1, 1] + global_confusion[2, 2]) / sum(sum(global_confusion))
    rnet_dict[pkl] = [round(global_acc, 4),
                      global_confusion[0, 0],
                      global_confusion[0, 1],
                      # global_confusion[0, 2],
                      global_confusion[1, 0],
                      global_confusion[1, 1],
                      # global_confusion[1, 2],
                      # global_confusion[2, 0],
                      # global_confusion[2, 1],
                      # global_confusion[2, 2]
                      ]
    print(global_confusion, "global_confu", global_acc, '\n')

    maximum = max(rnet_dict.items(), key=lambda x: x[1][0])
    print(maximum[-1], '= max accuracy')
    print(rnet_dict)

    with open(os.path.join(csv_path, 'q-br-{}%.csv'.format(tol)), 'w', newline='') as f:
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
                             # rnet_dict[pkl][5],
                             # rnet_dict[pkl][6],
                             # rnet_dict[pkl][7],
                             # rnet_dict[pkl][8],
                             # rnet_dict[pkl][9]
                             ])
        f.close()