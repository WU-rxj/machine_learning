import torch
x=torch.arange(4.)
x.requires_grad_(True)
# 等价于x=torch.arange(4.0,requires_grad=True)
print(x.grad)
y = 2 * torch.dot(x, x)
print(y)
# 通过调用反向传播函数来自动计算y关于x每个分量的梯度
y.backward()
print(x.grad)
print(x.grad == 4 * x)# True
# 标量可以直接调用`backward()` ，而向量需要先转换为标量或指定梯度方向。
x.grad.zero_()
y = x * x
# 等价于y.backward(torch.ones(len(x)))
y.sum().backward()
print(x.grad)