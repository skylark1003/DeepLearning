import os    # 常用路径操作、进程管理、环境参数等几类
# 1.读取数据集
os.makedirs(os.path.join('..', 'data'), exist_ok=True)
data_file = os.path.join('..', 'data', 'house_tiny.csv')    # 创建csv文件
with open(data_file, 'w') as f:
    f.write('NumRooms,Alley,Price\n')   # 列名
    f.write('NA,Pave,127500\n')         # 每行表示一个数据样本
    f.write('2,NA,106000\n')
    f.write('4,NA,178100\n')
    f.write('NA,NA,140000\n')

import pandas as pd
data = pd.read_csv(data_file)           # 从创建的csv文件中加载原始数据集
print(data)

# 2.处理缺失值
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean(numeric_only=True))    # 插值：空置用均值填充，非数值不操作
print(inputs)

# 对于inputs中的类别值或离散值，将NaN视为一个类别
# Alley若为Pava则Alley_Pave为1，若为NaN则Alley_nan为1
inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

# 3.转换为张量格式
# inputs和outputs中所有条目均为数值，则可以转化为张量格式。
import torch
X, y = torch.tensor(inputs.values), torch.tensor(outputs.values)
print(X)
print(y)