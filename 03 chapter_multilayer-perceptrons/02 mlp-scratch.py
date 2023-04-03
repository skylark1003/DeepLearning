import torch
from torch import nn
from d2l import torch as d2l

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 初始化一个随机W1，行数为输入层大小，列数为隐藏层大小
W1 = nn.Parameter(
    torch.randn(num_inputs, num_hiddens, requires_grad=True) * 0.01)
# 初始化一个全零b1，大小为隐藏层大小
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
# 初始化一个随机W2，行数为隐藏层大小，列数为输出层大小
W2 = nn.Parameter(
    torch.randn(num_hiddens, num_outputs, requires_grad=True) * 0.01)
# 初始化一个全零b2，大小为输出层大小
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

# 所有参数
params = [W1, b1, W2, b2]

# ReLU 函数
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 模型
def net(X):
    X = X.reshape((-1, num_inputs))
    H = relu(X @ W1 + b1)
    return (H @ W2 + b2)

# 损失
loss = nn.CrossEntropyLoss()

num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)

# 测试
d2l.predict_ch3(net, test_iter)