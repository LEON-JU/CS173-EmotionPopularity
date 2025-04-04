import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 定义情绪煽动值公式
def emotional_impact_score(arousal, valence):
    return arousal * (4 * (valence - 0.5)**2)

# 创建x轴（Arousal）和y轴（Valence）的数据
arousal = np.linspace(0, 1, 100)  # 从0到1，100个点
valence = np.linspace(0, 1, 100)  # 从0到1，100个点
arousal, valence = np.meshgrid(arousal, valence)  # 生成网格

# 计算z轴（EIS）的数据
EIS = emotional_impact_score(arousal, valence)

# 创建3D图
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 绘制曲面
surf = ax.plot_surface(arousal, valence, EIS, cmap='viridis', edgecolor='none')

# 设置坐标轴标签
ax.set_xlabel('Arousal', fontsize=12)
ax.set_ylabel('Valence', fontsize=12)
ax.set_zlabel('EIS (Emotional Impact Score)', fontsize=12)

# 添加颜色条
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5, label='EIS Value')

# 设置视角
ax.view_init(elev=30, azim=-135)  # 调整视角

# 显示图形
plt.show()
