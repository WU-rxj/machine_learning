import pandas as pd

# 读取数据
data = pd.read_csv('../data/house_tiny.csv')
print(data)

# 分割输入和输出
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]

# 分别处理数值列和非数值列的缺失值
inputs['NumRooms'] = inputs['NumRooms'].fillna(inputs['NumRooms'].mean())  # 填充数值列
print(inputs)

inputs = pd.get_dummies(inputs, dummy_na=True)
print(inputs)

import torch

X = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(outputs.to_numpy(dtype=float))
print(X, y)