import sympy

# 定义符号变量
x, y = sympy.symbols('x y')
# 定义函数
f = sympy.sqrt(x ** 2 + y ** 2)
# 分别求对x和y的偏导数
gradient_x = sympy.diff(f, x)
gradient_y = sympy.diff(f, y)
print(f"对x的偏导数: {gradient_x}")
print(f"对y的偏导数: {gradient_y}")