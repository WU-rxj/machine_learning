import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

def synthetic_data(w, b, num_examples):
    """生成y=Xw+b+噪声"""
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))

def data_iter(batch_size, features, labels):
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

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

# nn是神经网络的缩写
from torch import nn

net = nn.Sequential(nn.Linear(2, 1))

net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)

loss = nn.MSELoss()

trainer = torch.optim.SGD(net.parameters(), lr=0.03)

num_epochs = 3
batch_size = 10
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X) ,y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l:f}')
    
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)

d2l.set_figsize()
d2l.plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
d2l.plt.plot(features[:, 1].detach().numpy(), (net[0].weight[0][0]*features[:,0] + net[0].weight[0][1]*features[:,1] + net[0].bias).detach().numpy(), 'r-')
d2l.plt.show()