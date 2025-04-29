import torch

x=torch.arange(12).reshape((3,4))
print(x)
print(x[0:2,1:3])
print(x[0,1]) 

x[1,2]=9
print(x)
x[0:2,:]=12
print(x)
y=torch.zeros((3,4))
before = id(y)
y = y + x
print(id(y) == before)# 不一样了

Z = torch.zeros_like(y)
print('id(Z):', id(Z))
Z[:] = x + y
print('id(Z):', id(Z))

# 如果在后续计算中没有重复使用X， 我们也可以使用X[:] = X + Y或X += Y来减少操作的内存开销

A = x.numpy()
B = torch.tensor(A)
print(type(A), type(B))
# ndarray 和 tensor 之间的转换