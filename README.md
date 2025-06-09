# CS173-EmotionPopularity 项目

## 项目概述
分析微博话题中情绪特征与热度变化的关系，预测下一时段热度变化趋势

## 主要功能
1. 微博数据情绪分析（两套评估方式）
2. 热度变化趋势预测
3. 消融实验分析特征重要性

## 情感评估系统
项目包含两套独立的情感评估方式：
1. `emotion_analyze.py`
   - 加入情绪影响力分数(EIS)
   - 考虑情绪的非线性效应和交互作用
2. `emotion_analyze_v2.py`
   - 使用效价(valence)和唤醒度(arousal)的简单计算

## 目录结构
```
CS173-EmotionPopularity/
├── hotness_prediction/        # 训练部分
│   ├── feature_engineering.py # 特征工程
│   ├── model_training.py      # 模型训练
│   ├── main.py                # 主程序
│   └── README.md              # 训练说明
├── prepare_dataset/
│   ├── emotion_analyze.py     # 基础情绪分析
│   ├── emotion_analyze_v2.py  # 改进情绪分析
│   ├── preprocess.py          # 数据清洗和格式化
│   ├── postprocess.py         # 结果后处理（处理没有正确分析的博文，仅V1流程需要）
│   ├── test_api.py            # API测试脚本
│   ├── data/                  # 原始数据
│   │   ├── gt.csv             # 真实热度数据
│   │   ├── blog_data.csv      
│   │   └── comment_data.csv  
│   ├── processed_data/        # 经过了preprocess的数据
│   │   ├── blog_data.csv      
│   │   └── comment_data.csv  
│   └── emotion_analysis_output/ # 情绪分析结果
│       └── merged_data_v2.csv # 仅valence和arousal特征
│       └── merged_data.csv    # 带有Russel情绪模型特征
└── README.md                  # 本文档
CS173proj                      # 可视化
```

## 数据处理流程
1. 数据预处理 (`preprocess.py`)
   - 清洗原始微博数据
   - 标准化数据格式
   - 处理缺失值和异常值

2. 情绪分析 (任选其一)
   - `emotion_analyze.py`: 基础情绪分析
     - 计算效价(valence)和唤醒度(arousal)
   - `emotion_analyze_v2.py`: 改进情绪分析
     - 计算情绪影响力分数(EIS)
     - 考虑情绪的非线性效应

3. 后处理 (`postprocess.py`)
   - 合并情绪分析结果
   - V1流程中会有少量数据没有正确标注情感，手工补齐
   - 生成最终特征数据集
   - 保存为merged_data_v2.csv