import torch

A=torch.arange(20,dtype=torch.float32).reshape(5,4)
# 在调用函数来计算总和或均值时保持轴数不变会很有用
sum_a=A.sum(axis=0,keepdim=True)
print(A/sum_a)

# 沿某个轴计算A元素的累积总和,函数不会沿任何轴降低输入张量的维度
print(A.cumsum(axis=0))

# 点积 是相同位置的按元素乘积的和
x=torch.arange(4,dtype=torch.float32)
y=torch.ones(4,dtype=torch.float32)
print(x,y,torch.dot(x,y))# 等价于x*y.sum()

# 在代码中使用张量表示矩阵-向量积，使用mv函数 为矩阵A和向量x调用torch.mv(A, x)时，会执行矩阵-向量积
print(torch.mv(A,x))# 等价于A@x

# 矩阵-矩阵积 使用mm函数
B = torch.ones(4, 3)
print(torch.mm(A, B))# 等价于A@B

# 范数 向量的范数是将向量映射到标量的函数
u=torch.tensor([3.0,-4.0])
print(torch.norm(u))# 向量的L2范数
print(torch.abs(u).sum())# 向量的L1范数

# 矩阵的F范数:矩阵元素平方和的平方根
torch.norm(torch.ones((4, 9)))
