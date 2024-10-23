import os

import numpy as np
import torch.nn as nn
import torch.optim
import visdom
from torch.utils.data import DataLoader

from modelclasses.nets import VAE, loss_function
from setb import Brset

# #### old encoder-decoder training programs, for br only 200,
'''

# dataset configure
location = '/home/jzq/PycharmProjects/Br/Question-塑料分析'
code_location = '/home/jzq/PycharmProjects/Br/Question-塑料分析/pkls/coder'
nirloc = os.path.join(location, 'NIR.csv')
brloc = os.path.join(location, 'Percentage.csv')
set = VAESet(nir_location=nirloc, br_location=brloc)
spectrum_loader = DataLoader(set, shuffle=True, batch_size=16)

# train parameter config
epochs = 80
lr = 1e-4
device = torch.device('cuda:0')
viz = visdom.Visdom()
viz.close()
loss_cache = [1]

# build vae train object
# model = VAE(512, 80, 60).to(device)
model = AE(512, 80, 60).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

for epoch in range(epochs):
    for batchidx, [x, label] in enumerate(spectrum_loader):
        x = x.to(torch.float32).to(device)

        # x_hat, miu, logvar = model(x)
        x_hat = model(x)

        x_hat.to(torch.float32)

        # loss = loss_function(x_hat, x, miu, logvar)
        loss = criterion(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            [xval, _] = next(iter(spectrum_loader))
            xval = xval.to(torch.float32).to(device)

            # xval_hat, _, _ = model(xval)
            xval_hat = model(xval)
            for i in range(2):
                yy = xval[i, 0]
                yyh = xval_hat[i, 0]
                xx = np.arange(len(yy))
                viz.line(Y=yy, X=xx, win="x", name='x', opts={'showlegend': True})
                viz.line(Y=yyh, X=xx, win="x", name='x-hat', opts={'showlegend': True}, update='append')
                print("checkin", i, "batchidx", batchidx, "epoch", epoch, "loss", loss)
                time.sleep(0.05)
            lsf = float(loss)
            if lsf < min(loss_cache):
                torch.save(model, os.path.join(code_location, 'encoder.pkl'))
            loss_cache.append(lsf)
'''
# #### old encoder-decoder training programs, for br only 200


# dataset configure
path = '/home/jzq/PycharmProjects/Br/Question-塑料分析'
core_path = '/home/jzq/PycharmProjects/Br/Question-塑料分析/pkls/coder'
files = os.path.join(path, 'Data3/br')
# files = os.path.join(path, 'Data3/br4code')
br = os.path.join(path, 'BR.xlsx')
set = Brset(specroot=files, labelroot=br, mode='')
spectrum_loader = DataLoader(set, shuffle=True, batch_size=500)

# train parameter config
epochs = 100
lr = 1e-2
device = torch.device('cuda:0')
viz = visdom.Visdom()
viz.close()
loss_cache = [1e5]

# build vae train object
model = VAE(255, 80, 60).to(device)
# model = AE(255, 128, 80).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

for epoch in range(epochs):
    for batchidx, [x, label] in enumerate(spectrum_loader):
        x = x.to(torch.float32).to(device)

        x_hat, miu, logvar = model(x)
        # x_hat = model(x)
        #
        # x_hat.to(torch.float32)
        #
        loss = loss_function(x_hat, x, miu, logvar)
        # loss = criterion(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            [xval, _] = next(iter(spectrum_loader))
            xval = xval.to(torch.float32).to(device)

            xval_hat, miu, logvar = model(x)
            for i in range(2):
                yy = xval[i, :]
                yyh = xval_hat[i, :]
                xx = np.arange(len(yy))
                viz.line(Y=yy, X=xx, win="x", name='x', opts={'showlegend': True})
                viz.line(Y=yyh, X=xx, win="x", name='x-hat', opts={'showlegend': True}, update='append')
                print("checkin", i, "batchidx", batchidx, "epoch", epoch, "loss", loss)
                # time.sleep(0.01)

        if epoch >= int(0.9 * epochs):
            lsf = float(loss)
            if (lsf < min(loss_cache)):
                torch.save(model, os.path.join(core_path, 'encoder{}.pkl'.format(lsf)))
                print('saving.....................')
            loss_cache.append(lsf)
