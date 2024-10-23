import torch
import torch.nn as nn

device = torch.device('cuda:0')


# MLP creator
class MLP(nn.Module):
    def __init__(self, sizes: list, drops: list):
        '''
        param sizes: a list like [   'from' 34, 'to' 12, 'to' 4, 'to' 2   ]
        clamped with ReLU in between, softmax at the end
        '''
        drops += [0.2] * (len(sizes) - len(drops))

        super(MLP, self).__init__()

        self.model = nn.Sequential()
        for idx, size in enumerate(sizes):
            # if used as parts
            if idx == len(sizes) - 1:
                break

            self.model.add_module('L{}'.format(idx), nn.Linear(size, sizes[idx + 1]))
            if idx < len(sizes) - 2:
                self.model.add_module('acti{}'.format(idx), nn.ReLU())
                self.model.add_module('{}dropout'.format(idx), nn.Dropout(p=drops[idx]))
            # else:
            # self.model.add_module('outact{-1}', nn.Softmax())

    def forward(self, x):
        y = self.model(x)
        return y


# MLCP creator
class MLCP(nn.Module):
    def __init__(self, sizes: list, drops: list, out: int):
        super(MLCP, self).__init__()

        drops += [0.2] * (len(sizes) - len(drops))

        self.model = nn.Sequential()
        for idx, size in enumerate(sizes):
            if idx == len(sizes) - 1:
                break
            self.model.add_module('{}Conv'.format(idx),
                                  nn.Conv1d(size, sizes[idx + 1], kernel_size=3, padding=1, stride=1))
            self.model.add_module('{}batchnorm'.format(idx), nn.BatchNorm1d(sizes[idx + 1]))
            self.model.add_module('{}activ'.format(idx), nn.ReLU())
            self.model.add_module('{}maxpool'.format(idx), nn.AvgPool1d(3))
            self.model.add_module('{}dropout'.format(idx), nn.Dropout(p=drops[idx]))

        self.outlayer = nn.Sequential()
        self.outlayer.add_module('outlayer', nn.Linear(64, out))
        self.outlayer.add_module('output', nn.Sigmoid())
        # self.outlayer.add_module('output', nn.ReLU())

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[-1])
        y = self.model(x)
        y = y.view(y.shape[0], -1)
        y2 = self.outlayer(y)
        return y2


# resblk
# class Resblk(nn.Module):
#     def __init__(self, in_channel:int, out_channel:int):

# MLRP creator
class MLRP(nn.Module):
    def __init__(self, sizes: list, drops: list, out: int):
        super(MLRP, self).__init__()

        drops += [0.2] * (len(sizes) - len(drops))

        self.model = nn.Sequential()
        for idx, size in enumerate(sizes):
            if idx == len(sizes) - 1:
                break
            self.model.add_module('{}Conv'.format(idx),
                                  nn.Conv1d(size, sizes[idx + 1], kernel_size=3, padding=1, stride=1))
            self.model.add_module('{}batchnorm'.format(idx), nn.BatchNorm1d(sizes[idx + 1]))
            self.model.add_module('{}activ'.format(idx), nn.ReLU())
            self.model.add_module('{}maxpool'.format(idx), nn.MaxPool1d(3))
            self.model.add_module('{}dropout'.format(idx), nn.Dropout(p=drops[idx]))

        self.outlayer = nn.Sequential()
        self.outlayer.add_module('outlayer', nn.Linear(192, out))
        self.outlayer.add_module('output', nn.Sigmoid())
        self.outlayer.add_module('output', nn.ReLU())

    def forward(self, x):
        x = x.view(x.shape[0], 1, x.shape[-1])
        y = self.model(x)
        y = y.view(y.shape[0], -1)
        y2 = self.outlayer(y)
        return y2


class Siam_net(nn.Module):
    def __init__(self, heads_size: list, drops: list, heads_out: int, tail_size: list):
        super(Siam_net, self).__init__()
        # drops = [0, 0, 0, 0]
        drops += [0.2] * (len(heads_size) - len(drops))

        self.heads = MLCP(heads_size, drops, heads_out)
        self.tail = MLP(sizes=tail_size, drops=[])

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=0)  # one batch cut in half
        y1, y2 = self.heads(x1), self.heads(x2)

        z = y1 - y2

        out = self.tail(z)
        return out


# def loss_function(x1, x2, d_label):
#     csizes = [1, 32, 64, 128]
#     drops = [0,0,0,0]
#     out = 100
#     mlcp = MLCP(sizes=csizes, drops=drops, out=out)
#
#     y1 = mlcp(x1)
#     y2 = mlcp(x2)


if __name__ == "__main__":
    x = torch.randn([4, 1, 512])

    size = [512, 30, 20, 2]
    mlp = MLP(sizes=size, drops=[0.2, 0.2, 0.2, 0.2])
    print(mlp)
    print(mlp(x).shape, "output")

    # csize = [1, 30, 20, 2]
    # mlcp = MLCP(sizes=csize, drops=[], out=10)
    #
    # x1 = torch.randn([4, 1, 512])
    # x2 = torch.randn([4, 1, 512])
    #
    # headsize = [1, 32, 64, 128]
    # siam = Siam_net(heads_size=headsize, heads_out=10, tail_size=[10, 1])
    # q = siam(x1)
    # print(q.shape)
    # # q.backward()
