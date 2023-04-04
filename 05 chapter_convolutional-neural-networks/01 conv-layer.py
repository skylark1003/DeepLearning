import torch
from torch import nn
from d2l import torch as d2l

# 互相关运算
def corr2d(X, K):  
    """计算二维互相关运算。"""
    h, w = K.shape
    # 输出矩阵大小
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    # 对 Y 做计算
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y

# 验证上述二维互相关运算的输出
X = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
print(corr2d(X, K))

# 实现二维卷积层
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    # 前向运算
    def forward(self, x):
        return corr2d(x, self.weight) + self.bias

# 卷积层的一个简单应用：检测图像中不同颜色的边缘
X = torch.ones((6, 8))
X[:, 2:6] = 0
print(X)

K = torch.tensor([[1.0, -1.0]])

# 1 代表从白色到黑色的边缘 -1 代表从黑色到白色的边缘
Y = corr2d(X, K)
print(Y)

# 卷积核只可以检测垂直边缘
# 转置后只可以检测横向
corr2d(X.t(), K)

# 学习由 X 生成 Y 的卷积核
# 黑白图片通道为1  彩色图片通道为3
# 参数：输入通道 输出通道 核大小 偏差
conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    Y_hat = conv2d(X)
    # 均方误差
    l = (Y_hat - Y)**2
    conv2d.zero_grad()
    l.sum().backward()
    # 梯度下降
    conv2d.weight.data[:] -= 3e-2 * conv2d.weight.grad
    if (i + 1) % 2 == 0:
        print(f'batch {i+1}, loss {l.sum():.3f}')

conv2d.weight.data.reshape((1, 2))
print(conv2d.weight.data)