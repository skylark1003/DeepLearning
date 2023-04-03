import torch
from torch import nn
from torch.nn import functional as F

net = nn.Sequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))

X = torch.rand(2, 20)
print(net(X))

# 任何一个层或神经网络都是一个module的子类
# 自定义一个MLP实现和上面一样的函数
# 01 自定义块
class MLP(nn.Module):          # nn.Module的子类
    def __init__(self):
        super().__init__()     # 调用父类的init函数，定义一些参数
        self.hidden = nn.Linear(20, 256)   # 隐藏层
        self.out = nn.Linear(256, 10)      # 输出层

    # 定义前向函数的运算方法
    def forward(self, X):
        return self.out(F.relu(self.hidden(X)))
    
# 实例化多层感知机的层
net = MLP()
print(net(X))

# 02 顺序块
class MySequential(nn.Module):
    # *args是收集参数，相当于把若干个参数打包成一个传入
    def __init__(self, *args):
        super().__init__()
        for block in args:
            self._modules[block] = block

    def forward(self, X):
        for block in self._modules.values():
            X = block(X)
        return X

net = MySequential(nn.Linear(20, 256), nn.ReLU(), nn.Linear(256, 10))
print(net(X))

# 可以灵活定义函数，进行灵活的模型构造
class FixedHiddenMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.rand_weight = torch.rand((20, 20), requires_grad=False)
        self.linear = nn.Linear(20, 20)

    def forward(self, X):
        X = self.linear(X)
        X = F.relu(torch.mm(X, self.rand_weight) + 1)
        X = self.linear(X)
        while X.abs().sum() > 1:
            X /= 2
        return X.sum()

net = FixedHiddenMLP()
print(net(X))

# 混合搭配各种组合块
class NestMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(20, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU())
        self.linear = nn.Linear(32, 16)

    def forward(self, X):
        return self.linear(self.net(X))

chimera = nn.Sequential(NestMLP(), nn.Linear(16, 20), FixedHiddenMLP())
print(chimera(X))