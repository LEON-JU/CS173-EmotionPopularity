import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba
from collections import defaultdict
import seaborn as sns
FONT_PATH = 'simhei.ttf'
EMOTIONS = [
    "Happy", "Delighted", "Excited", "Astonished", "Aroused",
    "Tense", "Alarmed", "Angry", "Afraid", "Annoyed",
    "Distressed", "Frustrated", "Miserable", "Sad", "Gloomy",
    "Depressed", "Bored", "Droopy", "Tired", "Sleepy",
    "Calm", "Serene", "Pleased", "Content", "At Ease", "Relaxed"
]
# 读取数据
df = pd.read_csv('data.csv', parse_dates=[1])
emotion_columns = df.columns[8:34]  # I列到AH列（需验证索引位置）
df.rename(columns=dict(zip(emotion_columns, EMOTIONS)), inplace=True)
emotion_columns = EMOTIONS  # 更新列名引用

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 辅助函数：中文分词
def chinese_word_segmentation(text):
    return ' '.join(jieba.cut(text))

# ========== 分时段的群体情绪分布 ==========
# 提取小时信息
df['hour'] = df.iloc[:, 1].dt.hour

# 按小时分组计算平均情绪概率
hourly_emotions = df.groupby('hour')[emotion_columns].mean()

# 可视化
plt.figure(figsize=(18, 8))  # 加宽画布
hourly_emotions.T.plot(kind='bar', stacked=True, colormap='tab20')
plt.title('分时段群体情绪分布')
plt.xlabel('情绪类型')
plt.ylabel('平均概率')
plt.xticks(rotation=45, ha='right', fontsize=8)  # 增加对齐和字体调整
plt.legend(title='小时', bbox_to_anchor=(1.05, 1))
plt.tight_layout()
plt.show()

# 折线图方案
plt.figure(figsize=(18, 8))
for emotion in hourly_emotions.columns:
    plt.plot(hourly_emotions.index, hourly_emotions[emotion], 
            marker='o', linewidth=2, markersize=8, label=emotion)
plt.title('分时段群体情绪变化趋势')
plt.xlabel('小时')
plt.ylabel('平均概率')
plt.legend(bbox_to_anchor=(1.05, 1), fontsize=8)  # 缩小图例字体
plt.grid(True)
plt.tight_layout()
plt.show()

# 热力图方案
plt.figure(figsize=(18, 12))  # 加大画布尺寸
sns.heatmap(hourly_emotions.T, cmap='YlGnBu', annot=True, fmt=".2f",
           cbar_kws={'shrink': 0.8}, annot_kws={'size':8})
plt.title('分时段情绪概率热力图')
plt.xlabel('小时')
plt.ylabel('情绪类型')
plt.yticks(rotation=0, fontsize=8)  # 调整y轴标签
plt.tight_layout()
plt.show()

def plot_top_posts(hour_data, hour):  # 修改函数参数
    plt.figure(figsize=(20, 15))
    plt.suptitle(f'{hour}:00时段 点赞TOP5博文分析', y=1.02, fontsize=16)
    
    # 获取TOP5数据（使用iloc定位列）
    top5 = hour_data.nlargest(5, df.columns[3])  # D列是第4列（索引3）
    
    for idx, (_, row) in enumerate(top5.iterrows(), 1):
        # 文本处理（自动换行）
        content = '\n'.join([row.iloc[5][i:i+30] for i in range(0, len(row.iloc[5]), 30)])  # F列内容
        likes = row.iloc[3]  # D列点赞数
        
        # 创建子图（2行5列布局）
        ax_left = plt.subplot(5, 2, 2*idx-1)
        ax_right = plt.subplot(5, 2, 2*idx)
        
        # 左侧：博文内容展示
        ax_left.text(0.1, 0.5, 
                    f"▲ 点赞数：{likes}\n{content}",
                    va='center', fontsize=10)
        ax_left.axis('off')
        
        # 右侧：情绪分布条形图
        emotions = row[EMOTIONS].sort_values(ascending=False)[:5]  # 取概率最高的前5个情绪
        colors = plt.cm.tab20(np.linspace(0, 1, len(emotions)))
        ax_right.barh(emotions.index, emotions.values, color=colors)
        ax_right.set_xlim(0, 1.0)
        ax_right.set_xlabel('情绪概率')
        ax_right.grid(axis='x', alpha=0.3)
        
    plt.tight_layout()
    plt.show()
    
for hour, group in df.groupby('hour'):
    if len(group) < 5: 
        print(f'时段 {hour}:00 数据不足5条')
        continue
    plot_top_posts(group, hour)  # 传入当前小时数
# ========== 分时段的TF-IDF词云 ==========
# 按小时分组文本
hourly_texts = df.groupby('hour').apply(lambda x: ' '.join(x.iloc[:, 4].astype(str)))

# 生成停用词集合（需准备中文停用词文件）
STOPWORDS_PATH = 'hit_stopwords.txt'
with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
    stopwords = [line.strip() for line in f if line.strip()]

# 只取F列（第6列，索引5）的评论内容
hourly_comments = df.groupby('hour').apply(lambda x: ' '.join(x.iloc[:, 5].astype(str)))

# 添加省份城市名过滤列表（可根据数据情况补充）
province_cities = ['北京', '上海', '广东', '江苏', '河南', '湖北', '陕西', '四川', '重庆', '安徽']
stopwords += province_cities  # 将地区名加入停用词

specific = ['考研', '英语','学生','老师','凯文','周思成','今年','英一','不是','不能']
stopwords += specific  

# 增强版分词函数（过滤地名和短词）
def advanced_segmentation(text):
    words = jieba.cut(str(text))
    # 过滤条件：非停用词、长度>1、非纯数字、非地区名
    return ' '.join([word for word in words if 
                    word not in stopwords and 
                    len(word) > 1 and 
                    not word.isdigit() and 
                    not word.endswith(('省', '市', '区'))])

# 修改后的词云生成流程
for hour, text in hourly_comments.items():
    seg_text = advanced_segmentation(text)
    
    # 强化过滤的TF-IDF参数
    tfidf = TfidfVectorizer(
        stop_words=stopwords,
        max_features=80,
        token_pattern=r'(?u)\b[^\d\W]{2,}\b'  # 排除含数字的词汇
    )
    
    try:
        matrix = tfidf.fit_transform([seg_text])
        # 添加权重过滤阈值
        word_weights = {word: weight for word, weight in 
                       zip(tfidf.get_feature_names_out(), matrix.toarray().flatten())
                       if weight > 0.1}  # 过滤低权重词汇
        
        if word_weights:
            wc = WordCloud(
                font_path=FONT_PATH,
                background_color='white',
                collocations=False,
                regexp=r'[\u4e00-\u9fa5]{2,}'  # 仅匹配中文词语
            ).generate_from_frequencies(word_weights)
            # 可视化部分保持不变
        else:
            print(f"时段 {hour}:00 无有效词汇")
    except Exception as e:
        print(f"时段 {hour}:00 处理异常: {str(e)}")
    
    
    # 生成词云
    wc = WordCloud(
        font_path='simhei.ttf',
        background_color='white',
        max_words=50,
        width=800,
        height=600
    ).generate_from_frequencies(word_weights)
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.imshow(wc, interpolation='bilinear')
    plt.title(f'时段 {hour}:00 的TF-IDF词云')
    plt.axis('off')
    plt.show()