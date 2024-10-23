import csv
import os
import pickle
import glob
import numpy as np
import torch.optim
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import scale
from torch.utils.data import DataLoader
from torch.utils.data import Subset, ConcatDataset
import contextlib
from Cl4train.setc import Clset
device = torch.device('cuda:0')
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
files = os.path.join(path, 'Data3/HeFei/TXxlsx/TXNB_processed')
cl = os.path.join(path, 'Cl4train/Cls.xlsx')
set1 = Clset(specroot=files, labelroot=cl, mode='end')
# combined_set = set1

# 24-07-02:实例化得到的数据集才1362条光谱，显然不够。要把830件当中有Cl的拎进来
path = '/home/jzq/PycharmProjects/Br/Question-塑料分析'
files = os.path.join(path, 'Data3/HeFei/xlsx/NB_notcol')
br = os.path.join(path, 'BR.xlsx')
set2 = Clset(specroot=files, labelroot=br, mode='end')
combined_set = ConcatDataset([set1, set2])

ender = DataLoader(combined_set, shuffle=True, batch_size=256)

paths = '/home/jzq/PycharmProjects/Br/Question-塑料分析/Cl4test/rnets'
out = 2

dates = os.listdir(paths)
for date in dates:
    core_path = os.path.join(paths, date)
    csv_path = core_path
    # pkls = glob.glob(os.path.join(core_path, '**/*.pkl'), recursive=True)
    pkls = [pkl for pkl in os.listdir(core_path) if pkl.endswith('.pkl')]
    label_rate = 1e3
    # for divtol in [210, 220, 230, 240, 250, 260, 270, 280, 290]:

    for divtol in np.linspace(0.01, 0.05, 21):
        divtol = round(divtol*label_rate)/label_rate
        rnet_dict = {}

        #
        # # if pca, then load pca
        # pca_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/pkls/pca/pca80.pkl'
        # with open(pca_path, 'rb') as f:
        #     pca = pickle.load(f)
        #     f.close()
        #     print('pca read successfully')
        #
        # # if vae
        # vae_path = ''
        # with open(vae_path, 'rb') as f:
        #     encoder = pickle.load(f)
        #     f.close()
        #     print('vae read successfully')
        #
        #
        # def encode(input):
        #     m, logvar = encoder.encode(input)
        #     y = encoder.reprameterize(m, logvar)
        #     return y
        #
        #
        # def normalize(x):
        #     y = scale(x, axis=1)
        #     return y

        tol = divtol  # 2024--7-9-10-58: tol在分类器之前定训练集，那么必须在分类器之后定测试集。鉴于输出的不是0～0.5～1的概率而是0~0.25~的预测，那么分divtol和tol就没什么意义了
        for pkl in pkls:
            network = torch.load(os.path.join(core_path, pkl))
            print('{} '.format(pkl.split('/')[-1]), divtol)
            vscore = 0
            total_vscore = 0
            vconfu, global_confusion = np.zeros([2, 2]), np.zeros([2, 2])

            for x_valid, label_valid in ender:
                with torch.no_grad():
                    network.eval()

                    # 4.1 inputting
                    x_valid = x_valid.to(torch.float32).to(device)
                    label_valid = label_valid.to(torch.float32).to(device)
                    label_valid_class = label_valid > tol * label_rate  # label to T/F

                    # #4.2.1 if y is ,1)
                    if out == 1:
                        y_valid = network(x_valid).view(-1)  # outputting
                        y_valid_class = y_valid > tol  # y ,1) to TF
                        y_valid_class = y_valid_class.view(y_valid_class.shape[0])

                    # 4.2.2 if y is ,2)
                    elif out == 2:
                        y_valid = network(x_valid)  # outputting
                        y_valid_class = y_valid.argmax(axis=-1) > 0.5  # y ,2) to TF

                    # acc of this v_batch
                    vscore += (y_valid_class == label_valid_class).sum().item()  # num of rights
                    total_vscore += y_valid_class.shape[0]  # num of totals
                    vconfu += confusion_matrix(label_valid_class.cpu(), y_valid_class.cpu(), labels=[False, True])

            acc_valid_class = vscore / total_vscore
            global_confusion = global_confusion + vconfu
            global_acc = (global_confusion[0, 0] + global_confusion[1, 1]) / sum(sum(global_confusion))
            # global_acc = accuracy_valid
            # global_acc = (global_confusion[0, 0] + global_confusion[1, 1] + global_confusion[2, 2]) / sum(sum(global_confusion))
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

        # #  24-07-02-21-40:  计划把训练过程中的所有sysprint输出到一个txt文件当中以方便后续研究
        output_filename = 'parameters_4test{}.txt'.format(divtol)
        with open(os.path.join(csv_path, output_filename), 'w') as output_file:
            with contextlib.redirect_stdout(output_file):
                params = {}
                params['datascale'] = combined_set.__len__()
                params['tol'] = tol
                params['positive'] = sum(a >= tol * label_rate for a in set1.labels) + sum(
                    a >= tol * label_rate for a in set2.labels)
                params['negative'] = sum(a < tol * label_rate for a in set1.labels) + sum(
                    a < tol * label_rate for a in set2.labels)
                print(params)
                print(network)
