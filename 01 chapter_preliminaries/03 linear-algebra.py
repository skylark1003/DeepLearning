import torch
# 1.标量
x = torch.tensor([3.0])
y = torch.tensor([2.0])
print(x + y)               # 标量运算
print(x * y)
print(x / y)
print(x ** y)

# 2.向量
x = torch.arange(4)        # 向量可以视为标量值组成的列表
print(x)
print(x[3])                # 通过张量的索引访问任一元素
print(len(x))              # 访问张量的长度
print(x.shape)             # 只有一个轴的张量，形状只有一个元素

# 3.矩阵
A = torch.arange(20).reshape(5, 4)                    # 通过两个分量m和n创建一个形状为m×n的矩阵
print(A)
print(A.T)                 # 矩阵的转置
B = torch.tensor([[1, 2, 3], [2, 0, 4], [3, 4, 5]])   # 对称矩阵
print(B)
print(B.T)
print(B == B.T)

# 4.张量（多轴数据结构）
X = torch.arange(24).reshape(2, 3, 4)                 # 三维张量
print(X)


# 1.基本运算
A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
B = A.clone()              # 通过分配新内存，将A的一个副本分配给B
print(A)
print(A + B)
print(A * B)               # 按元素乘法：哈达玛积 ⊙

a = 2
X = torch.arange(24).reshape(2, 3, 4)
print(a + X)               # 广播
print((a * X).shape)       # 广播

# 2.降维
x = torch.arange(4, dtype=torch.float32)
print(x)
print(x.sum())             # 计算元素的和

A = torch.arange(40, dtype=torch.float32).reshape(2, 5, 4)
A.reshape(2, 5, 4)
print(A)
print(A.shape)
print(A.sum())                  # 计算任意张量的元素和
A_sum_axis0 = A.sum(axis=0)     # 指定轴，通过求和降低维度
print(A_sum_axis0)
print(A_sum_axis0.shape)
A_sum_axis1 = A.sum(axis=1)
print(A_sum_axis1)
print(A_sum_axis1.shape)
A_sum_axis2 = A.sum(axis=[0, 1])
print(A_sum_axis2)
print(A_sum_axis2.shape)

print(A.mean())                 # 求平均值
print(A.sum() / A.numel())
print(A.mean(axis=0))           # 按维度求平均值
print(A.sum(axis=0) / A.shape[0])

# 3.非降维求和
sum_A = A.sum(axis=1, keepdim=True)       # 计算总和时保持轴数不变（求和维度值变为1）
print(sum_A)
print(A / sum_A)                # 通过广播将A除以A，所以须保持维度一致

print(A)                        # 某个轴计算A元素的累计总和
print(A.cumsum(axis=0))

# 4.点积
y = torch.ones(4, dtype=torch.float32)
print(x)
print(y)
print(torch.dot(x, y))          # 向量向量积：相同位置的按元素乘积的和
print(torch.sum(x * y))         # 等价于执行按元素乘法，然后进行求和

A = torch.arange(20, dtype=torch.float32).reshape(5, 4)
print(A.shape)
print(x.shape)
print(A)
print(x)
print(torch.mv(A, x))           # 矩阵向量积：A m×n x n×1

B = torch.ones(4, 3)
print(A)
print(B)
print(torch.mm(A, B))           # 矩阵矩阵积：A 5×4 B 4×3

# 5.范数
u = torch.tensor([3.0, -4.0])
print(torch.norm(u))            # 向量的L2范数（长度）
print(torch.abs(u).sum())       # 向量的L1范数（绝对值求和）

print(torch.norm(torch.ones((4, 9))))    # 矩阵的佛罗贝尼乌斯范数（Frobenius norm）：矩阵元素的平方和的平方根（将矩阵拉成向量的L2范数）