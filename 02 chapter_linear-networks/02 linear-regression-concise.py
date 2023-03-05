import torch
# 引入一些处理数据的模块
from torch.utils import data
from d2l import torch as d2l

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

def load_array(data_arrays, batch_size, is_train=True):
    # 构造一个PyTorch数据迭代器
    # * 是元组拆包符号
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

print(next(iter(data_iter)))

# nn是神经网络的缩写
from torch import nn

# 定义模型变量net，它是一个Sequential类的实例
# Sequential：将层放到容器里，将多个层串联在一起
# Linear：线性层（全连接层），指定输入输出维度
net = nn.Sequential(nn.Linear(2, 1))

# net第0层，使用均值为0，方差为0.01的值替换 w（权重） 的真实值
net[0].weight.data.normal_(0, 0.01)
# net第0层，使用0替换 b（偏差） 的真实值
net[0].bias.data.fill_(0)

# 计算均方误差（平方L2范数）
loss = nn.MSELoss()

# 小批量随机梯度下降，传入参数和超参数字典（学习率）
trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        # net中包含w和b，不需要传
        l = loss(net(X), y)
        trainer.zero_grad()
        l.backward()
        # 进行模型更新
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')

w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)