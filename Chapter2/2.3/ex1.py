import torch

a=torch.tensor(1.0)
b=torch.tensor(2.0)
print(a+b,a-b,a*b,a/b)

x=torch.arange(4)
print(x[3])

X=torch.arange(20,dtype=torch.float32).reshape(5,4)
print(X.T)# 转置
print(X.shape)
Y=X.clone()

# 两个矩阵的按元素乘法称为Hadamard积，用符号⊙表示
print(X*Y)

x_sum=X.sum(axis=0)
print(x_sum,x_sum.shape)
# 同一行的元素求和

x_sum=X.sum(axis=1)
print(x_sum,x_sum.shape)
# 同一列的元素求和

print(X.mean(axis=0))# 求均值
# A.mean(axis=0) 等价于 A.sum(axis=0) / A.shape[0]