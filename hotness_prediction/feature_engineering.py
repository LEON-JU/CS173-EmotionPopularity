import pandas as pd
import numpy as np
from datetime import datetime

def load_data():
    # 加载微博数据
    weibo_df = pd.read_csv('../prepare_dataset/emotion_analysis_output/merged_data_v2.csv')
    # 加载热度数据
    heat_df = pd.read_csv('../prepare_dataset/data/gt.csv', sep='\t')
    return weibo_df, heat_df

def improved_emotional_impact_score(valence, arousal):
    """计算改进的情绪影响力分数"""
    # 情绪强度 = arousal的绝对值
    # 情绪极性 = valence的正负
    return np.abs(arousal) * np.sign(valence)

def extract_hourly_features(weibo_df):
    # 转换时间格式
    weibo_df['发布时间'] = pd.to_datetime(weibo_df['发布时间'], format='mixed', errors='coerce')
    # 过滤无效时间和1-6点(保留0点)
    weibo_df = weibo_df[weibo_df['发布时间'].notna()]
    weibo_df = weibo_df[
        (weibo_df['发布时间'].dt.hour >= 7)
    ].copy()
    weibo_df['hour'] = weibo_df['发布时间'].dt.strftime('%Y-%m-%d %H')
    
    # 计算EIS(情绪影响力分数)
    weibo_df['EIS'] = improved_emotional_impact_score(
        weibo_df['arousal'], weibo_df['valence'])
    
    # 计算小时级别特征
    hourly_features = weibo_df.groupby('hour').agg({
        'valence': ['mean', 'std'],
        'arousal': ['mean', 'std'],
        'EIS': ['mean', 'std']
    })
    hourly_features.columns = [
        'valence_mean', 'valence_std',
        'arousal_mean', 'arousal_std',
        'EIS_mean', 'EIS_std'
    ]
    return hourly_features

def calculate_heat_change(heat_df):
    # 计算热度变化方向(1=上升，0=下降)
    heat_df['heat_change'] = (heat_df['heat'].diff() > 0).astype(int)
    heat_df['next_heat_change'] = heat_df['heat_change'].shift(-1)
    return heat_df

def merge_features(hourly_features, heat_df):
    # 合并特征和标签
    merged_df = pd.merge(hourly_features, heat_df, 
                         left_on='hour', right_on='hour', how='inner')
    return merged_df

if __name__ == '__main__':
    weibo_df, heat_df = load_data()
    hourly_features = extract_hourly_features(weibo_df)
    heat_df = calculate_heat_change(heat_df)
    final_df = merge_features(hourly_features, heat_df)
    final_df.to_csv('processed_data.csv', index=False)