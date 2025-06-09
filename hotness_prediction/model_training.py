import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt

def load_processed_data():
    return pd.read_csv('processed_data.csv')

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

def add_time_features(df):
    # 添加时间序列特征
    df['hour_of_day'] = pd.to_datetime(df['hour']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['hour']).dt.dayofweek
    
    # 添加滞后特征
    for lag in [1, 2, 3]:
        df[f'heat_change_lag{lag}'] = df['heat_change'].shift(lag)
        df[f'valence_mean_lag{lag}'] = df['valence_mean'].shift(lag)
        df[f'arousal_mean_lag{lag}'] = df['arousal_mean'].shift(lag)
    
    # 添加移动平均特征
    df['heat_change_ma3'] = df['heat_change'].rolling(3).mean()
    df['valence_mean_ma3'] = df['valence_mean'].rolling(3).mean()
    df['arousal_mean_ma3'] = df['arousal_mean'].rolling(3).mean()
    
    return df.dropna()

def train_model(X, y):
    # 消融实验1: 完整特征集
    print("\n=== 完整特征集 ===")
    X_full = X.copy()
    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.4, random_state=1)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        objective='binary:logistic'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1分数: {f1_score(y_test, y_pred):.4f}")

    # 消融实验2: 去除EIS特征
    print("\n=== 去除EIS特征 ===")
    X_no_eis = X[[col for col in X.columns if not col.startswith('EIS_')]]
    X_train, X_test, y_train, y_test = train_test_split(
        X_no_eis, y, test_size=0.4, random_state=1)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        objective='binary:logistic'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1分数: {f1_score(y_test, y_pred):.4f}")

    # 消融实验3: 去除arousal_mean_ma3特征
    print("\n=== 去除arousal_mean_ma3特征 ===")
    X_no_ma = X[[col for col in X.columns if col != 'arousal_mean_ma3']]
    X_train, X_test, y_train, y_test = train_test_split(
        X_no_ma, y, test_size=0.4, random_state=1)
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        objective='binary:logistic'
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f"准确率: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1分数: {f1_score(y_test, y_pred):.4f}")
    
    # 使用完整特征集训练最终模型
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=1)
    
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        random_state=42,
        objective='binary:logistic'
    )
    model.fit(X_train, y_train)
    
    # 评估模型
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'\n完整模型 - Accuracy: {acc:.4f}, F1: {f1:.4f}')
    print(classification_report(y_test, y_pred))
    
    # 特征重要性
    plt.figure(figsize=(10, 6))
    plt.barh(X.columns, model.feature_importances_)
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    
    return model

if __name__ == '__main__':
    df = load_processed_data()
    df = add_time_features(df)
    
    # 准备特征和目标变量
    X = df[['valence_std', 'arousal_mean', 'EIS_mean', 'EIS_std', 'arousal_mean_ma3']]
    y = df['next_heat_change']
    
    model = train_model(X, y)
    model.save_model('hotness_predictor.json')