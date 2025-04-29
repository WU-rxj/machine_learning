import random
import torch
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))
# y.shape(-1,1)是把y变成列向量

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

# 有必要定义一个函数， 该函数能打乱数据集中的样本并以小批量方式获取数据

def data_iter(batch_size, features, labels):
    """
    数据批量迭代器
    参数:
        batch_size: 每批数据的大小
        features: 特征张量
        labels: 标签张量
    功能步骤:
        1. 获取样本总数
        2. 创建索引列表并随机打乱(实现数据随机化)
        3. 按批次大小循环生成批量数据
        4. 使用yield返回每个批次的特征和标签(实现惰性计算)
    """
    num_examples = len(features)  # 1. 获取样本总数
    indices = list(range(num_examples))  # 创建0到num_examples-1的索引列表
    # 2. 随机打乱索引顺序，实现数据随机化
    random.shuffle(indices)
    # 3. 按批次大小循环
    for i in range(0, num_examples, batch_size):
        # 获取当前批次的索引，防止越界
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        # 4. 使用yield返回当前批次的特征和标签
        yield features[batch_indices], labels[batch_indices]
        
batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break