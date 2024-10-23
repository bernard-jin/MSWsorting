import torch

'''
自创的实验性的比大小loss：one-cross
'''


def one_cross(label, y, one_height, one_hot='need'):
    if one_hot == 'need':
        label = torch.nn.functional.one_hot(label.to(torch.int64))
    # 使用线性映射进行归一化
    y = (y - y.min()) / (y.max() - y.min())
    a = abs(label - y).sum()
    # a越大loss越大，但a>1时loss尤其增大！！
    loss = (a < 1) * a * one_height + (a >= 1) * ((a - 1) * 10 + one_height)
    return loss


'''
biased mse
'''


def mse_w(label, y, weight):
    weight_vector = (label[:, 0] >= label[:, 1]) * weight[0] + (label[:, 0] < label[:, 1]) * weight[1]
    # weight_vector = (label.argmax(dim=1) == 0) * weight[0] + (label.argmax(dim=1) == 1) * weight[1] + (label.argmax(dim=1) == 2) * weight[2]
    loss = (label - y) ** 2
    weight_vector = weight_vector.to(torch.float32)
    loss = torch.dot(loss.sum(dim=-1), weight_vector)
    return loss


def mse3_w(label, y, weight):
    # weight_vector = (label[:, 0] >= label[:, 1]) * weight[0] + (label[:, 0] < label[:, 1]) * weight[1]
    weight_vector = (label.argmax(dim=1) == 0) * weight[0] + (label.argmax(dim=1) == 1) * weight[1] + (
                label.argmax(dim=1) == 2) * weight[2]
    loss = (label - y) ** 2
    weight_vector = weight_vector.to(torch.float32)
    loss = torch.dot(loss.sum(dim=-1), weight_vector)
    return loss


def labelreshape(label):
    label = label.reshape([label.shape[0], 1])
    l1, l2 = torch.chunk(label, 2, dim=0)
    new_label = torch.cat([l1, l2], dim=1)
    # print(new_label.shape)
    return new_label


def labeldist(label):
    label = label.reshape([label.shape[0], 1])
    l1, l2 = torch.chunk(label, 2, dim=0)
    # new_label = torch.abs(l1 - l2)
    new_label = l1 - l2
    # new_label = torch.cat([l1, l2], dim=1)
    # print(new_label.shape)
    return new_label


def labelnorm(label):
    label = label.reshape([label.shape[0], 1])
    l1, l2 = torch.chunk(label, 2, dim=0)
    new_label = l1 - l2
    new_label = torch.norm(new_label, p=2, dim=1)
    # new_label = torch.cat([l1, l2], dim=1)
    # print(new_label.shape)
    return new_label


'''
focal loss
'''
#     gamma = 1.5
#     alpha = 0.25
device = torch.device('cuda:0')


def focal_loss(loss, pred, true, gamma, alpha):
    # pred_prob = torch.sigmoid(pred)  # prob from logits
    pred_prob = pred  # prob from logits
    true_f = torch.zeros(pred_prob.shape[0], 2).to(device)
    for idx, ele in enumerate(true.argmax(dim=1)):
        true_f[idx, ele] = 1
    p_t = true_f * pred_prob + (1 - true_f) * (1 - pred_prob)
    alpha_factor = true_f * alpha + (1 - true_f) * (1 - alpha)
    modulating_factor = (1.0 - p_t) ** gamma
    loss *= (alpha_factor * modulating_factor).mean()
    return loss


def main():
    # lb = torch.tensor([[0.1,0.3],[0.9,0],[0.9,0]])
    # y = torch.tensor([[0,0.9],[0.23,0.24],[0,0.9]])
    # w = torch.tensor([1,1.5])
    # print(mse_w(lb, y, weight=w))

    # lb = torch.tensor([[0.1,0.3,0.4],[0.9,0.8,0],[0.9,1,0]])
    # y = torch.tensor([[0,0.9,0.8],[0.23,0,0.24],[0,0.78,0.9]])
    # w = torch.tensor([1,1.5,2])
    # print(mse_w(lb, y, weight=w))
    label = torch.randn(128)
    # print(labelreshape(label).shape)
    print(labelnorm(label).shape)


if __name__ == '__main__':
    main()
