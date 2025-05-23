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
batch_size = 10
# 有必要定义一个函数， 该函数能打乱数据集中的样本并以小批量方式获取数据

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
        
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def linreg(X, w, b):  
    """线性回归模型"""
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):  
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  #@save
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
            
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 因为l形状是(batch_size,1)，而不是一个标量。l中的所有元素被加到一起，
        # 并以此计算关于[w,b]的梯度
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
        
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')