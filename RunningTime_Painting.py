import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import ScalarFormatter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.sans-serif'] = ['Times New Roman']
# 读取CSV文件并计算每10个数据点的平均值
data_files = {
    'VPPO': './MAT_datas/PPO_vectorized_time.csv',
    'A2C': './MAT_datas/new_A2C_time.csv',
    'PPO': './MAT_datas/new_PPO_time.csv',
    'SAC': './MAT_datas/new_SAC_time.csv'
}

# 服务数F
F_values = [10, 20, 30, 40, 50]
mean_values = {}

# 处理每个文件的数据
for key, file_name in data_files.items():
    df = pd.read_csv(file_name, header=None)
    data = df.iloc[1].values  # 读取第二行数据
    means = np.array([data[i:i+10].mean() for i in range(0, len(data), 10)])  # 每10个数据点计算平均值
    mean_values[key] = means
    print(f"{key} average values: {means}")  # 打印每个算法每10个数据点的平均值

# 横坐标设置
x = np.arange(len(F_values))

# 纵坐标设置
width = 0.1  # 柱子的宽度

# 创建柱状图
plt.figure(figsize=(10, 6))

# 定义柱子的颜色
colors = {
    'VPPO': 'lightpink',
    'A2C': 'lightblue',
    'PPO': 'lightgreen',
    'SAC': '#D4FF19'
}

# 绘制每个算法的平均时间
for i, (key, values) in enumerate(mean_values.items()):
    plt.bar(x + i * width, values, width, label=key, color=colors[key], align='center')

# 添加图例
plt.legend(fontsize=16)

# 添加标题和标签
plt.title('', fontsize=20)
plt.xlabel('Numbers of Services (N)', fontsize=18)
plt.ylabel('Running Time (s)', fontsize=18)

plt.xticks(x + 2 * width, F_values, fontsize=18)
plt.yticks(fontsize=18)
plt.tight_layout()
# 显示网格
plt.grid(True, axis='y')

# 显示图表
plt.show()