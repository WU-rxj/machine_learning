import torch
x=torch.arange(12)
print(x)
print(x.shape)
print(x.numel())
print(x.reshape(3,4))
print(torch.zeros((2,3,4)))
print(torch.ones((2,3,4)))
# 其中的每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样
print(torch.randn(3,4))
y=torch.tensor([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
print(y)
