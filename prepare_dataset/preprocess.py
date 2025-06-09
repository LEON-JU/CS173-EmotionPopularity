import pandas as pd
import re
import os


def clean_weibo_text(text):
    if not isinstance(text, str):
        return text
    
    # 规则3: 删除回复@用户:前缀，保留后面内容
    text = re.sub(r'^回复@[^:]+:(.*)', r'\1', text)
    
    # 规则1&2: 删除@用户部分
    text = re.sub(r'@[^\s]+', '', text)

    # 规则5: 删除#或##包裹的话题信息
    text = re.sub(r'#[^#]+#', '', text)

    # 规则4: 删除水军评论
    spam_keywords = ['感谢分享', '晚上好', '谢谢分享', '内容不错', 
                    '周末愉快', '真不错', '下午好']
    if any(keyword in text for keyword in spam_keywords):
        return ''
    
    return text.strip()

def preprocess_comments():
    # 确保输出目录存在
    os.makedirs('processed_data', exist_ok=True)
    
    # 读取数据
    df = pd.read_csv('data/comment_data.csv')
    
    # 清洗微博正文
    df['微博正文'] = df['微博正文'].apply(clean_weibo_text)
    
    # 处理ip列，去除"来自"前缀
    df['ip'] = df['ip'].str.replace('来自', '', regex=False)
    
    # 删除空评论
    df = df[df['微博正文'] != '']
    
    # 保存处理后的数据
    df.to_csv('processed_data/comment_data.csv', index=False, encoding='utf-8-sig')
    print("数据预处理完成，已保存到 processed_data/comment_data.csv")

def preprocess_blogs():
    # 确保输出目录存在
    os.makedirs('processed_data', exist_ok=True)
    
    # 读取数据
    df = pd.read_csv('data/blog_data.csv')

    # 删除视频号的行
    df = df[df['发布工具'] != "微博视频号"]
    
    # 选择并重排列
    columns_to_keep = [
        "bid", "发布时间", "id", "点赞数", "ip", "微博正文",
        "user_id", "用户昵称"
    ]
    df = df[columns_to_keep]

    df['微博正文'] = df['微博正文'].apply(clean_weibo_text)
    
    # 保存处理后的数据
    df.to_csv('processed_data/blog_data.csv', index=False, encoding='utf-8-sig')
    print("博客数据预处理完成，已保存到 processed_data/blog_data.csv")

if __name__ == '__main__':
    # preprocess_comments()
    preprocess_blogs()