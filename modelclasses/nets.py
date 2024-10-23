import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0')


# 定义VAE
class VAE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, latent_size)
        self.fc22 = nn.Linear(hidden_size, latent_size)
        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

    # 生成分布的参数层
    def encode(self, x):
        h1 = F.relu(
            self.fc1(x)
        )
        return self.fc21(h1), self.fc22(h1)

    # 自参数采样层
    def reprameterize(self, miu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(miu)
        return miu + eps * std

    # 自样映射输出层
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = self.fc4(h3)
        return h4

    def forward(self, x):
        miu, logvar = self.encode(x)
        z = self.reprameterize(miu, logvar)
        return self.decode(z), miu, logvar


class AE(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super(AE, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2a = nn.Linear(hidden_size, latent_size)

        self.fc2b = nn.Linear(latent_size, latent_size)

        self.fc3 = nn.Linear(latent_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, input_size)

        # self.fc1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, stride=1, padding=1)
        # self.fc2a = nn.Conv1d(hidden_size, latent_size, kernel_size=3, stride=1, padding=1)
        # self.fc2b = nn.Conv1d(latent_size, latent_size, kernel_size=3, stride=1, padding=1)
        # self.fc3 = nn.Conv1d(latent_size, hidden_size, kernel_size=3, stride=1, padding=1)
        # self.fc4 = nn.Conv1d(hidden_size, input_size, kernel_size=3, stride=1, padding=1)

    # 生成分布的参数层
    def encode(self, x):
        # x = x.view(x.shape[0], x.shape[-1], 1)
        # h1 = F.hardswish(self.fc1(x))
        # h2 = F.hardswish(self.fc2a(h1))
        # h3 = F.hardswish(self.fc2b(h2))

        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2a(h1))
        h3 = F.relu(self.fc2b(h2))
        return h3

    # 自样映射输出层
    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        h4 = self.fc4(h3)
        # h4 = h4.view(h4.shape[0], 1, h4.shape[1])
        return h4

    def forward(self, x):
        y = self.encode(x)
        z = self.decode(y)
        return z


# 损失函数
def loss_function(x_recon, x, miu, logvar):
    CE = F.mse_loss(x_recon, x, reduction='none')
    KLD = -0.5 * torch.sum(1 - miu.pow(2) + logvar - logvar.exp())
    output = torch.sum(CE + KLD)
    return output


if __name__ == '__main__':
    input = torch.randn([16, 1, 512])
    vae = VAE(512, 80, 20)
    print(input.dtype)
    m, logvar = vae.encode(input)
    y = vae.reprameterize(m, logvar)
    input_hat = vae.decode(y)

    loss = loss_function(input_hat, input, m, logvar)
    print(loss.shape)

    # ae = AE(512, 80, 20)
    # print(input.dtype)
    # m = ae.encode(input)
    # input_hat = ae.decode(m)

    print("input shape {}, output shape {}, middle shape {}".format(input.shape, input_hat.shape, m.shape))
