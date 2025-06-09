# 微博话题热度预测系统

## 功能
基于微博情绪分析数据预测下一时段话题热度变化方向(上升/下降)

## 主要特征
- 情绪特征：valence/arousal的均值和标准差
- EIS特征：情绪影响力分数(Emotional Impact Score)的均值和标准差
- 时间特征：小时、星期
- 时序特征：滞后特征(1-3小时)、移动平均(3小时)
- 热度变化方向标签：1=上升，0=下降

## 数据预处理
1. 过滤1-6点的夜间数据(保留0点)
2. 计算每小时的情绪统计特征
3. 计算热度变化方向

## 消融实验
系统自动运行三种特征组合的对比实验：
1. 完整特征集
2. 去除EIS特征
3. 去除arousal_mean_ma3特征

## 使用方法
1. 安装依赖：
```bash
pip install pandas numpy xgboost scikit-learn matplotlib
```

2. 运行主程序：
```bash
python main.py
```

3. 查看结果：
- 消融实验对比结果
- 特征重要性图(feature_importance.png)
- 训练好的模型(hotness_predictor.json)

## 文件说明
- feature_engineering.py: 特征工程代码
- model_training.py: 模型训练和消融实验
- main.py: 主程序入口
- processed_data.csv: 处理后的数据集