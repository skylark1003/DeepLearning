import torch
# 1.入门
x = torch.arange(12)   # 创建张量（由一个数值组成的数组）
print(x)
print(x.shape)         # 张量形状
print(x.numel())       # 张量元素总数
X = x.reshape(3, 4)    # 改变形状不改变元素数量和元素值
print(X)

a = torch.zeros((2, 3, 4))   # 全0
b = torch.ones((2, 3, 4))    # 全1
c = torch.randn(3, 4)        # 随机
print(a)
print(b)
print(c)

d = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])  # 指定内容
print(d)

# 2.运算符
# 标准算术运算符，按元素运算
x = torch.tensor([1.0, 2, 4, 8])  # 1.0表示为浮点数
y = torch.tensor([2, 2, 2, 2])
print(x + y)
print(x - y)
print(x * y)
print(x / y)
print(x ** y)
print(torch.exp(x))  # 指数运算

# 多个张量连结
X = torch.arange(12, dtype=torch.float32).reshape((3, 4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(torch.cat((X, Y), dim=0))   # 按0轴
print(torch.cat((X, Y), dim=1))   # 按1轴

print(X == Y)    # 通过逻辑运算符构建二元张量
print(X.sum())   # 元素求和

# 3.广播机制（broadcasting mechanism）
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))
print(a)
print(b)
print(a + b)     # 广播 分别按行按列复制

# 4.索引和切片
print(X)
print(X[-1])         # 读
print(X[1:3])        # 读
print(X[1:3, 0:2])   # 读
X[1, 2] = 9          # 写
print(X)
X[0:2, :] = 12       # 写
print(X)

# 5.节省内存
before = id(Y)          # id()表示该object在python中的唯一标识号
Y = Y + X
after1 = id(Y)
print(after1 == before)  # 为新结果分配内存，之前的Y被析构掉了
Y += X
after2 = id(Y)
print(after2 == after1)  # 执行原地操作（可以减少内存开销）

Z = torch.zeros_like(Y)  # zeros_like(): 数据类型和Y一样但值均为0
print("id(Z):", id(Z))
Z[:] = X + Y             # 执行原地操作
print("id(Z):", id(Z))

# 6.转换为其他Python对象
A = X.numpy()            # 转换为NumPy
B = torch.tensor(A)
print(type(A))
print(type(B))

a = torch.tensor([3.5])
print(a)
print(a.item())
print(float(a))
print(int(a))