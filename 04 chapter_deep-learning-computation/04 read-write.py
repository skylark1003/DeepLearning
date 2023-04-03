import torch
from torch import nn
from torch.nn import functional as F

# 加载和保存张量
x = torch.arange(4)
torch.save(x, 'x-file')

x2 = torch.load('x-file')
print(x2)

# 存储一个张量列表
y = torch.zeros(4)
torch.save([x, y], 'x-files')
x2, y2 = torch.load('x-files')
print((x2, y2))

# 写入或读取从字符串映射到张量的字典
mydict = {'x': x, 'y': y}
torch.save(mydict, 'mydict')
mydict2 = torch.load('mydict')
print(mydict2)

# 加载和保存模型参数
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(20, 256)
        self.output = nn.Linear(256, 10)

    def forward(self, x):
        return self.output(F.relu(self.hidden(x)))

net = MLP()
X = torch.randn(size=(2, 20))
Y = net(X)

# 将模型的参数存储为一个叫做“mlp.params”的文件
# 用state_dict()得到所有参数从参数名字字符串到parameter的映射
# 存为字典
torch.save(net.state_dict(), 'mlp.params')

# 声明网络
clone = MLP()
# 用文件中的参数覆盖当前网络参数
clone.load_state_dict(torch.load('mlp.params'))
print(clone.eval())

# 对比clone和之前的Y
Y_clone = clone(X)
print(Y_clone == Y)