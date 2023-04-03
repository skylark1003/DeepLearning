import torch
from torch import nn

print(torch.device('cpu'), torch.cuda.device('cuda'), torch.cuda.device('cuda:1'))

print(torch.cuda.device_count())

def try_gpu(i=0):  
    """如果存在，则返回gpu(i)，否则返回cpu()。"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def try_all_gpus():  
    """返回所有可用的GPU，如果没有GPU，则返回[cpu(),]。"""
    devices = [
        torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]
    return devices if devices else [torch.device('cpu')]

print(try_gpu(), try_gpu(10), try_all_gpus())

x = torch.tensor([1, 2, 3])
# 默认在CPU上
print(x.device)

# 创建在GPU上
X = torch.ones(2, 3, device=try_gpu())
print(X)

# 没有第二个GPU则创建在CPU上
Y = torch.rand(2, 3, device=try_gpu(1))
print(Y)

# 要计算不同GPU上的数据，需在同一个GPU
Z = Y.cuda(0)
print(X)
print(Z)

print(X + Z)

print(Z.cuda(0) is Z)

# 神经网络与GPU
net = nn.Sequential(nn.Linear(3, 1))
# 复制到其他GPU
net = net.to(device=try_gpu())

print(net(X))

print(net[0].weight.data.device)