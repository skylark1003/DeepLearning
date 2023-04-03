# 在windows上，子进程会自动import启动它的这个文件，而在import的时候是会自动执行这些语句的。
# 如果不加__main__限制的化，就会无限递归创建子进程，进而报freeze_support()错。
if __name__ == '__main__':

    import torch
    from IPython import display
    from d2l import torch as d2l

    batch_size = 256
    # 返回训练集和测试集的迭代器
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

    # 将图片信息拉成向量 28×28=784
    num_inputs = 784
    # 10个分类类别
    num_outputs = 10

    # 初始化W和b
    W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
    b = torch.zeros(num_outputs, requires_grad=True)

    X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    # 按轴0求和
    print(X.sum(0, keepdim=True))
    # 按轴1求和
    print(X.sum(1, keepdim=True))

    # 实现softmax，对每一行做softmax
    def softmax(X):                               # 在该网络中：X行数为批量大小，列数为输入维数
        X_exp = torch.exp(X)                      # 01 对X求幂
        partition = X_exp.sum(1, keepdim=True)    # 02 对每一行求和
        return X_exp / partition                  # 03 利用广播机制获得 每个元素的指数/该行所有元素的指数和 即softmax

    # 测试softmax
    X = torch.normal(0, 1, (2, 5))
    X_prob = softmax(X)
    print(X_prob)
    print(X_prob.sum(1))

    # 实现softmax回归模型
    def net(X):
        return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

    # y_hat的简单例子
    y = torch.tensor([0, 2])
    y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
    # 把两个样本属于真实标签（0，2）的预测概率拿出来，即拿出来的是y_hat[0, 0]和y_hat[1, 2]
    print(y_hat[[0, 1], y])

    # 实现交叉熵损失函数（取真实值对应的预测概率）
    def cross_entropy(y_hat, y):
        return -torch.log(y_hat[range(len(y_hat)), y])

    print(cross_entropy(y_hat, y))

    # 将预测类和真实y元素进行比较
    def accuracy(y_hat, y):
        """计算预测正确的数量。"""
        if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
            y_hat = y_hat.argmax(axis=1)        # 最大值下标 即预测的类别
        cmp = y_hat.type(y.dtype) == y          # 比较与真实类别是否一致
        return float(cmp.type(y.dtype).sum())   # 预测正确的数量

    print(accuracy(y_hat, y) / len(y))          # 预测正确的概率

    def evaluate_accuracy(net, data_iter):
        """计算在指定数据集上模型的精度。"""
        if isinstance(net, torch.nn.Module):
            net.eval()
        metric = Accumulator(2)
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]

    class Accumulator:
        """在`n`个变量上累加。"""
        def __init__(self, n):
            self.data = [0.0] * n

        def add(self, *args):
            self.data = [a + float(b) for a, b in zip(self.data, args)]

        def reset(self):
            self.data = [0.0] * len(self.data)

        def __getitem__(self, idx):
            return self.data[idx]

    print(evaluate_accuracy(net, test_iter))

    # 训练
    def train_epoch_ch3(net, train_iter, loss, updater):
        """训练模型一个迭代周期（定义见第3章）。"""
        # 将模型设置为训练模式
        if isinstance(net, torch.nn.Module):
            net.train()
        # 训练损失综合、训练准确度总和、样本数
        metric = Accumulator(3)
        for X, y in train_iter:
            # 计算梯度并更新参数
            y_hat = net(X)
            l = loss(y_hat, y)
            if isinstance(updater, torch.optim.Optimizer):
                # 使用PyTorch内置的优化器和损失函数
                updater.zero_grad()
                l.backward()
                updater.step()
                metric.add(
                    float(l) * len(y), accuracy(y_hat, y),
                    y.size().numel())
            else:
                # 使用定制的优化器和损失函数
                l.sum().backward()
                updater(X.shape[0])
                metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        # 返回训练损失和训练精度
        return metric[0] / metric[2], metric[1] / metric[2]

    class Animator:  #@save
        """在动画中绘制数据"""
        def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                     ylim=None, xscale='linear', yscale='linear',
                     fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                     figsize=(3.5, 2.5)):
            # 增量地绘制多条线
            if legend is None:
                legend = []
            d2l.use_svg_display()
            self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
            if nrows * ncols == 1:
                self.axes = [self.axes, ]
            # 使用lambda函数捕获参数
            self.config_axes = lambda: d2l.set_axes(
                self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
            self.X, self.Y, self.fmts = None, None, fmts

        def add(self, x, y):
            # 向图表中添加多个数据点
            if not hasattr(y, "__len__"):
                y = [y]
            n = len(y)
            if not hasattr(x, "__len__"):
                x = [x] * n
            if not self.X:
                self.X = [[] for _ in range(n)]
            if not self.Y:
                self.Y = [[] for _ in range(n)]
            for i, (a, b) in enumerate(zip(x, y)):
                if a is not None and b is not None:
                    self.X[i].append(a)
                    self.Y[i].append(b)
            self.axes[0].cla()
            for x, y, fmt in zip(self.X, self.Y, self.fmts):
                self.axes[0].plot(x, y, fmt)
            self.config_axes()
            display.display(self.fig)
            d2l.plt.draw()
            d2l.plt.pause(0.001)
            display.clear_output(wait=True)

    # 训练函数
    def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
        """训练模型（定义见第3章）"""
        animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                            legend=['train loss', 'train acc', 'test acc'])
        for epoch in range(num_epochs):
            train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
            test_acc = evaluate_accuracy(net, test_iter)
            animator.add(epoch + 1, train_metrics + (test_acc,))
        train_loss, train_acc = train_metrics
        assert train_loss < 0.5, train_loss
        assert train_acc <= 1 and train_acc > 0.7, train_acc
        assert test_acc <= 1 and test_acc > 0.7, test_acc

    lr = 0.1

    def updater(batch_size):
        return d2l.sgd([W, b], lr, batch_size)

    num_epochs = 10
    train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    d2l.plt.show()

    def predict_ch3(net, test_iter, n=6):  #@save
        """预测标签（定义见第3章）"""
        for X, y in test_iter:
            break
        trues = d2l.get_fashion_mnist_labels(y)
        preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
        titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
        d2l.show_images(
            X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
        d2l.plt.show()

    predict_ch3(net, test_iter)