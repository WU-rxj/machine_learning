import os

# 获取当前脚本所在目录
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
os.makedirs(data_dir, exist_ok=True)
data_file = os.path.join(script_dir, '..', 'data', 'house_tiny.csv')  # 修改路径

try:
    with open(data_file, 'w') as f:
        f.write('NumRooms,Alley,Price\n')  # 列名
        f.write('NA,Pave,127500\n')  # 每行表示一个数据样本
        f.write('2,NA,106000\n')
        f.write('4,NA,178100\n')
        f.write('NA,NA,140000\n')
    print("文件创建并写入成功！")
except Exception as e:
    print(f"出现错误: {e}")

print(os.getcwd())