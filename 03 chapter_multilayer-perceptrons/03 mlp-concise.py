import torch
from torch import nn
from d2l import torch as d2l

net = nn.Sequential(
    nn.Flatten(),         # 3D 展平为二维
    nn.Linear(784, 256),  # 线性层 输入784 输出256
    nn.ReLU(),            # 激活函数
    nn.Linear(256, 10))   # 线性层 输入256 输出10

# 初始化W
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);

batch_size, lr, num_epochs = 256, 0.1, 10
loss = nn.CrossEntropyLoss()
trainer = torch.optim.SGD(net.parameters(), lr=lr)

train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
