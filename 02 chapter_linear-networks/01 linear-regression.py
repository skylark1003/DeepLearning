import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):
    # 生成 y = Xw + b + 噪声
    # 生成 X：均值为0，方差为1的随机数，大小为 样本数 × w长度
    X = torch.normal(0, 1, (num_examples, len(w)))
    # 生成 y
    y = torch.matmul(X, w) + b
    # 为了让问题难一点，加一个均值为0，方差为0.01的随机噪音
    y += torch.normal(0, 0.01, y.shape)
    # 将 X 和 y 转为列向量返回
    return X, y.reshape((-1, 1))     # -1 表示Numpy根据维度自动计算数组的shape属性值

true_w = torch.tensor([2, -3.4])     # 真实的 w
true_b = 4.2                         # 真实的 b
# 生成特征和标注，获得训练样本
features, labels = synthetic_data(true_w, true_b, 1000)
# 第0个样本：特征为长为2的向量，标注为标量
print('features:', features[0], '\nlabel:', labels[0])
# 显示
d2l.set_figsize()                    # 设置图表大小
# 绘制散点图
# 在pytorch部分版本中需要从计算图中detach才能显示
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy())
# 显示
# d2l.plt.show()

# 输入：批量大小、矩阵特征、标签向量
# 输出：大小为batch_size的小批量
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    # 生成每个样本的indices（0-n-1），转成python的list
    indices = list(range(num_examples))
    # 随机打乱
    random.shuffle(indices)
    # 从 0 到 num_examples，每次跳 batch_size 的大小
    for i in range(0, num_examples, batch_size):
        # 每次从 i 开始，到 i+batch_size。最后一批可能出现未满情况，取 i+batch_size 和 num_examples 最小值
        # 该 batch_indices 为随机的
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        # 每次随机产生
        # python的一个生成器（generator)，不断调用不断返回，直到全部完成
        yield features[batch_indices], labels[batch_indices]

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break

# 通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
# 偏置初始化为0
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):
    # 线性回归模型
    return torch.matmul(X, w) + b     # 广播

def squared_loss(y_hat, y):
    # 均方损失
    return (y_hat - y.reshape(y_hat.shape))**2 / 2

# 输入：参数（包括w和b）、学习率、批量大小
def sgd(params, lr, bath_size):
    # 小批量随机梯度下降
    # 在该模块下，所有计算得出的tensor的requires_grad都自动设置为False。
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / bath_size    # 之前损失函数未求均值，这里求均值
            param.grad.zero_()                      # 梯度设为0

lr = 0.05              # 学习率
num_epochs = 5         # 迭代周期（整个数据扫描次数）
net = linreg           # 目的：方便换成别的模型
loss = squared_loss    # 在该模型中为均方损失

# 两层 for loop （迭代周期 批量）
for epoch in range(num_epochs):
    # 每次拿出批量大小的 X 和 y
    for X, y in data_iter(batch_size, features, labels):
        # net() 做预测  loss() 做损失
        l = loss(net(X, w, b), y)
        # 求和之后算梯度
        l.sum().backward()
        # 用 sgd 对 w 和 b 进行更新
        sgd([w, b], lr, batch_size)
    # 评价精度
    with torch.no_grad():
        train_1 = loss(net(features, w, b), labels)
        print(f' epoch {epoch + 1}, loss {float(train_1.mean()):f}')

# 我们拥有真实参数，可以进行比较
print(f'w的估计误差：{true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差：{true_b - b}')