from feature_engineering import *
from model_training import *
import pandas as pd
import json

def predict_next_hour(model, latest_data):
    """使用训练好的模型预测下一小时热度变化"""
    features = add_time_features(latest_data)
    X = df[['valence_std', 'arousal_mean', 'EIS_mean', 'EIS_std', 'arousal_mean_ma3']]
    return model.predict(X)[0]

if __name__ == '__main__':
    # 1. 特征工程
    weibo_df, heat_df = load_data()
    hourly_features = extract_hourly_features(weibo_df)
    heat_df = calculate_heat_change(heat_df)
    final_df = merge_features(hourly_features, heat_df)
    final_df.to_csv('processed_data.csv', index=False)
    
    # 2. 模型训练
    df = pd.read_csv('processed_data.csv')
    df = add_time_features(df)
    X = df[['valence_std', 'arousal_mean', 'EIS_mean', 'EIS_std', 'arousal_mean_ma3']]
    y = df['next_heat_change']
    model = train_model(X, y)
    
    # 3. 保存最新数据用于预测
    latest_data = df.iloc[-1:].copy()
    latest_data.to_json('latest_data.json', orient='records')
    
    print("模型训练完成，结果已保存")