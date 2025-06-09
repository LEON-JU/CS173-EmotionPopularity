import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def improved_emotional_impact_score(arousal, valence):
    """
    计算改进的情绪影响分数，基于唤起度和效价
    参数:
    arousal - 情绪唤起度 (0-1): 表示情绪激活的强度
    valence - 情绪效价 (0-1): 表示情绪的正负性，0为极负面，1为极正面
    返回:
    情绪影响分数(Emotional Impact Score)
    """
    # 非线性唤起效应 - 使用正弦函数模拟唤起度的非线性影响
    # 指数0.8使曲线略微偏向高唤起状态，系数1.2用于调整整体强度
    arousal_effect = 1.2 * np.sin(np.pi * arousal**0.8)
    # 效价的U形关系 - 极端情绪（高正面或高负面）影响更强
    # 加入常数0.2确保即使在中性效价下也有基础影响
    valence_effect = 4 * (valence - 0.5)**2 + 0.2
    
    # 唤起和效价的交互作用 - 高唤起配合极端效价产生额外影响
    # 绝对值使得正负极端效价都能与高唤起产生强交互
    interaction = 0.5 * arousal * np.abs(2 * valence - 1)
    
    # 最终情绪影响分数 - 结合基础效应和交互效应
    return arousal_effect * valence_effect + interaction

# 定义情绪煽动值公式
def emotional_impact_score(arousal, valence):
    return arousal * (4 * (valence - 0.5)**2)

if __name__ == "__main__":
    # 创建x轴（Arousal）和y轴（Valence）的数据
    arousal = np.linspace(0, 1, 100)  # 从0到1，100个点
    valence = np.linspace(0, 1, 100)  # 从0到1，100个点
    arousal, valence = np.meshgrid(arousal, valence)  # 生成网格

    # 计算z轴（EIS）的数据
    # EIS = emotional_impact_score(arousal, valence)
    EIS = improved_emotional_impact_score(arousal, valence)

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
