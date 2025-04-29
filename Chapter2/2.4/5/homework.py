import sympy

# 定义符号变量
x = sympy.Symbol('x')
# 定义函数
f = 3 * x ** 2 + 5 * sympy.exp(x ** 2)
# 求梯度（导数）
gradient = sympy.diff(f, x)
print(gradient)

