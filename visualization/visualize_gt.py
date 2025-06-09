import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# 读取数据
df = pd.read_csv('../prepare_dataset/data/gt.csv', sep='\t')
df['hour'] = pd.to_datetime(df['hour'])

# 创建图表
plt.figure(figsize=(12, 6))
plt.plot(df['hour'], df['heat'], marker='o', linestyle='-', color='b')

# 设置图表格式
plt.title('GT Heat Trend (2025-03-05 to 2025-03-09)')
plt.xlabel('Time')
plt.ylabel('Heat Value')
plt.grid(True)

# 设置x轴日期格式
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator())
plt.gcf().autofmt_xdate()

# 保存图表
plt.tight_layout()
plt.savefig('./graphs/gt_heat_trend.png')
plt.show()