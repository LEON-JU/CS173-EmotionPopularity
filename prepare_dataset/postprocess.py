import pandas as pd
import re
import os



def preprocess_comments():
    df = pd.read_csv('emotion_analysis_output/comment_data_refined.csv')
    
    # 删除Relaxed之后的列
    keep_cols = df.columns.tolist()[:df.columns.get_loc('Relaxed')+1]
    df = df[keep_cols]

    # 情感列从Happy到Relaxed
    emotion_cols = ['Happy', 'Delighted', 'Excited', 'Astonished', 'Aroused',
                   'Tense', 'Alarmed', 'Angry', 'Afraid', 'Annoyed',
                   'Distressed', 'Frustrated', 'Miserable', 'Sad', 'Gloomy',
                   'Depressed', 'Bored', 'Droopy', 'Tired', 'Sleepy',
                   'Calm', 'Serene', 'Pleased', 'Content', 'At Ease', 'Relaxed']

    # 计算情绪得分总和
    df['emotion_sum'] = df[emotion_cols].sum(axis=1)

    zero_count = len(df[df['emotion_sum'] != 1])
    print(f"情绪加和不为1的行数: {zero_count}")

    
    if zero_count > 0:
        # 重新分析情绪加和不为1的行
        invalid_rows = df[abs(df['emotion_sum'] - 1) > 0.01]  # 允许0.01的浮点误差
        temp_input = "temp_invalid_comments.csv"
        temp_output = "temp_reanalyzed_comments.csv"
        
        # 保存需要重新分析的行
        invalid_rows.to_csv(temp_input, index=False, encoding='utf-8-sig')
        
        # 重新分析情绪
        from emotion_analyze import analyze_weibo_emotions
        analyze_weibo_emotions(temp_input, temp_output)
        # input()
        
        # 读取重新分析的结果
        reanalyzed = pd.read_csv(temp_output)
        
        # 更新原始数据中的情绪得分
        invalid_indices = invalid_rows.index.tolist()
        for i, idx in enumerate(invalid_indices):
            for col in emotion_cols:
                df.at[idx, col] = reanalyzed.iloc[i][col]
        
        # 清理临时文件
        if os.path.exists(temp_input):
            os.remove(temp_input)
        if os.path.exists(temp_output):
            os.remove(temp_output)
    
    # 删除临时列
    df = df.drop(columns=['emotion_sum'])
    
    df.to_csv('emotion_analysis_output/comment_data_refined.csv', index=False, encoding='utf-8-sig')
    print("评论数据后处理完成")

def preprocess_blogs():
    df = pd.read_csv('emotion_analysis_output/blog_data_refined.csv')
    
    # 删除Relaxed之后的列
    keep_cols = df.columns.tolist()[:df.columns.get_loc('Relaxed')+1]
    df = df[keep_cols]

    # 情感列从Happy到Relaxed
    emotion_cols = ['Happy', 'Delighted', 'Excited', 'Astonished', 'Aroused',
                   'Tense', 'Alarmed', 'Angry', 'Afraid', 'Annoyed',
                   'Distressed', 'Frustrated', 'Miserable', 'Sad', 'Gloomy',
                   'Depressed', 'Bored', 'Droopy', 'Tired', 'Sleepy',
                   'Calm', 'Serene', 'Pleased', 'Content', 'At Ease', 'Relaxed']

    # 计算情绪得分总和
    df['emotion_sum'] = df[emotion_cols].sum(axis=1)

    zero_count = len(df[df['emotion_sum'] != 1])
    print(f"情绪加和不为1的行数: {zero_count}")

    
    if zero_count > 0:
        # 重新分析情绪加和不为1的行
        invalid_rows = df[abs(df['emotion_sum'] - 1) > 0.01]  # 允许0.01的浮点误差
        temp_input = "temp_invalid_comments.csv"
        temp_output = "temp_reanalyzed_comments.csv"
        
        # 保存需要重新分析的行
        invalid_rows.to_csv(temp_input, index=False, encoding='utf-8-sig')
        
        # 重新分析情绪
        from emotion_analyze import analyze_weibo_emotions
        analyze_weibo_emotions(temp_input, temp_output)
        # input()
        
        # 读取重新分析的结果
        reanalyzed = pd.read_csv(temp_output)
        
        # 更新原始数据中的情绪得分
        invalid_indices = invalid_rows.index.tolist()
        for i, idx in enumerate(invalid_indices):
            for col in emotion_cols:
                df.at[idx, col] = reanalyzed.iloc[i][col]
        
        # 清理临时文件
        if os.path.exists(temp_input):
            os.remove(temp_input)
        if os.path.exists(temp_output):
            os.remove(temp_output)
    
    # 删除临时列
    df = df.drop(columns=['emotion_sum'])
    
    df.to_csv('emotion_analysis_output/blog_data_refined.csv', index=False, encoding='utf-8-sig')
    print("博客数据后处理完成")

# def merge_data():
#     """合并博客和评论数据"""
#     # 读取博客数据
#     blog_df = pd.read_csv('emotion_analysis_output/blog_data_refined.csv')
#     # 读取评论数据
#     comment_df = pd.read_csv('emotion_analysis_output/comment_data_refined.csv')
    
#     # 合并数据
#     merged_df = pd.concat([blog_df, comment_df], ignore_index=True)
    
#     # 按发布时间排序
#     merged_df = merged_df.sort_values('发布时间', ascending=True)
    
#     # 保存合并结果
#     merged_df.to_csv('emotion_analysis_output/merged_data.csv', index=False, encoding='utf-8-sig')
#     print("数据合并完成，已保存到 emotion_analysis_output/merged_data.csv")

def merge_data():
    """合并博客和评论数据"""
    # 读取博客数据
    blog_df = pd.read_csv('processed_data/blog_data.csv')
    # 读取评论数据
    comment_df = pd.read_csv('processed_data/comment_data.csv')
    
    # 合并数据
    merged_df = pd.concat([blog_df, comment_df], ignore_index=True)
    
    # 按发布时间排序
    merged_df = merged_df.sort_values('发布时间', ascending=True)
    
    # 保存合并结果
    merged_df.to_csv('processed_data/merged_data.csv', index=False, encoding='utf-8-sig')
    print("数据合并完成，已保存到 processed_data/merged_data.csv")

if __name__ == '__main__':
    # preprocess_comments()
    # preprocess_blogs()
    merge_data()