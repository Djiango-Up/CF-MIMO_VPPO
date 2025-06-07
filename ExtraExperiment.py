import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'serif'


plt.rcParams['font.sans-serif'] = ['Times New Roman']
# 读取数据
def read_data(file_path):
    data = pd.read_csv(file_path, header=None)
    # 第一行是序号，第二行是数据
    index = data.iloc[0].values
    values = data.iloc[1].values
    return index, values

index1, values1 = read_data('D://data//Vectorized_ppo_qos_ful.csv')
index2, values2 = read_data('D://data//Vectorized_PPO_QoS_noVec.csv')
index3, values3 = read_data('D://data//Vectorized_PPO_QoS_noRet.csv')
index4, values4 = read_data('D://data//PPO_qos.csv')

# 数据采样，每隔100个点取一个值
sample_interval = 100
x = index1[::sample_interval]
y1 = values1[::sample_interval]
y2 = values2[::sample_interval]
y3 = values3[::sample_interval]
y4 = values4[::sample_interval]

# 使用指数加权移动平均进行平滑
def smooth_data(y, alpha):
    return pd.Series(y).ewm(alpha=alpha, adjust=False).mean().values

# 平滑后的数据
smoothed_y1 = smooth_data(y1, alpha=0.1)
smoothed_y2 = smooth_data(y2, alpha=0.1)
smoothed_y3 = smooth_data(y3, alpha=0.1)
smoothed_y4 = smooth_data(y4, alpha=0.1)

# 创建图表
plt.figure(figsize=(11, 6))

# 绘制平滑后的折线图
plt.plot(x, smoothed_y1, label='VPPO', color='royalblue', linestyle='-', linewidth=2, alpha=0.7)
plt.plot(x, smoothed_y2, label='VPPO_v', color='coral', linestyle='--', linewidth=2, alpha=0.7)
plt.plot(x, smoothed_y3, label='VPPO_r', color='forestgreen', linestyle='-.', linewidth=2, alpha=0.7)
plt.plot(x, smoothed_y4, label='PPO', color='purple', linestyle=':', linewidth=2, alpha=0.7)

# 添加标题和坐标轴标签
# plt.title('', fontsize=16)
plt.xlabel('Steps', fontsize=22, fontfamily='Times New Roman')
plt.ylabel('QoS ', fontsize=22, fontfamily='Times New Roman')
# plt.xlabel('Steps', fontsize=24, labelpad=15)

# 设置坐标轴刻度字体
plt.xticks(fontsize=22, fontfamily='Times New Roman')
plt.yticks(fontsize=22, fontfamily='Times New Roman')

# 添加图例和网格
plt.legend(loc='lower right', fontsize=20, frameon=False,prop={'family': 'Times New Roman'})
plt.grid(True, linestyle='--', alpha=0.5)

# 调整布局
plt.xlim(0,10000)
plt.legend(fontsize=20)
plt.tight_layout()

