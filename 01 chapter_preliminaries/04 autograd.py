import torch
x = torch.arange(4.0)
print(x)
# 计算之前需要一个地方存储梯度
x.requires_grad_(True)  # 等价于 x = torch.arange(4.0, requires_grad=True)
print(x.grad)           # 通过 x.grad 访问x的梯度，默认值为None
y = 2 * torch.dot(x, x)
print(y)
# 通过反向传播函数来自动计算y关于x每个分量的梯度
y.backward()            # 求导
print(x.grad)           # 访问导数
print(x.grad == 4 * x)

# 在默认情况下，PyTorch会积累梯度，需要清除之前的值
x.grad.zero_()          # 下划线表示重写内容
y = x.sum()
y.backward()
print(x.grad)

# 对非标量调用 backward() 需要传入一个 gradient 参数，指定微分函数关于 self 的梯度
x.grad.zero_()
y = x * x
y.sum().backward()      # 等价于 y.backward(torch.ones(len(x)))
print(x.grad)

x.grad.zero_()
y = x * x
u = y.detach()          # 将u移动到计算图之外，即u是一个常数而不是关于x的函数，值为x*x
z = u * x

z.sum().backward()
print(x.grad == u)

x.grad.zero_()
y.sum().backward()      # y依旧是关于x的函数
print(x.grad == 2 * x)

def f(a):
    b = a * 2
    while b.norm() < 1000:
        b = b * 2
    if b.sum() > 0:
        c = b
    else:
        c = 100 * b
    return c

a = torch.randn(size=(), requires_grad=True)
d = f(a)
d.backward()
# 对于任何a，存在某个常量标量k，使得f(a)=k*a
# 其中k的值取决于输入a，因此可以用d/a验证梯度是否正确
print(a.grad == d / a)