import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 读取数据
df = pd.read_csv('../hotness_prediction/processed_data.csv')
df['hour'] = pd.to_datetime(df['hour'])

# 过滤1-6点数据(与训练数据一致)
df = df[df['hour'].dt.hour >= 7].copy()

# 设置完整时间索引(处理缺失时间段)
full_range = pd.date_range(
    start=df['hour'].min(),
    end=df['hour'].max(),
    freq='H'
)
df = df.set_index('hour').reindex(full_range).reset_index()
df = df.rename(columns={'index': 'hour'})

# 设置图表样式
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['axes.grid'] = True

# 可视化valence_std
plt.figure()
plt.plot(df['hour'], df['valence_std'], color='r', linewidth=2, marker='o', markersize=3)
plt.title('Valence Standard Deviation Over Time')
plt.xlabel('Time')
plt.ylabel('Valence Std')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/valence_std_trend.png')
plt.close()

# 可视化valence_mean
plt.figure()
plt.plot(df['hour'], df['valence_mean'], color='orange', linewidth=2, marker='o', markersize=3)
plt.title('Valence Mean Over Time')
plt.xlabel('Time')
plt.ylabel('Valence Mean')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/valence_mean_trend.png')
plt.close()

# 可视化arousal_mean
plt.figure()
plt.plot(df['hour'], df['arousal_mean'], color='g', linewidth=2, marker='o', markersize=3)
plt.title('Arousal Mean Over Time')
plt.xlabel('Time')
plt.ylabel('Arousal Mean')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/arousal_mean_trend.png')
plt.close()

# 可视化EIS_mean
plt.figure()
plt.plot(df['hour'], df['EIS_mean'], color='b', linewidth=2, marker='o', markersize=3)
plt.title('Emotional Impact Score (EIS) Mean Over Time')
plt.xlabel('Time')
plt.ylabel('EIS Mean')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/EIS_mean_trend.png')
plt.close()

# 可视化EIS_std
plt.figure()
plt.plot(df['hour'], df['EIS_std'], color='purple', linewidth=2, marker='o', markersize=3)
plt.title('Emotional Impact Score (EIS) Standard Deviation Over Time')
plt.xlabel('Time')
plt.ylabel('EIS Std')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.grid(True)
plt.tight_layout()
plt.savefig('./graphs/EIS_std_trend.png')
plt.close()