import pandas as pd
from test_api import analyze_sentiment
import time
import os
import json

def save_progress(output_file, current_index, processed_indices):
    """保存处理进度到临时文件"""
    progress_file = f"{output_file}.progress"
    progress_data = {
        'current_index': current_index,
        'processed_indices': processed_indices
    }
    with open(progress_file, 'w') as f:
        json.dump(progress_data, f)

def load_progress(output_file):
    """从临时文件加载处理进度"""
    progress_file = f"{output_file}.progress"
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            progress_data = json.load(f)
        return progress_data['current_index'], progress_data['processed_indices']
    return 0, []

def analyze_weibo_emotions(input_file, output_file, batch_size=5):
    """
    分析微博数据的情感并保存结果
    :param input_file: 输入CSV文件路径
    :param output_file: 输出CSV文件路径
    :param batch_size: 每次API调用的批处理大小
    """
    try:
        # 读取数据
        df = pd.read_csv(input_file)
        
        # 如果输出文件已存在，直接加载
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            print(f"检测到已有分析结果文件，继续处理...")
        else:
            # 初始化情感分析结果列
            df["sentiment"] = 0.0
            df["intensity"] = 0.0
        
        # 加载进度
        start_index, processed_indices = load_progress(output_file)
        
        # 分批处理微博正文
        for i in range(start_index, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            messages = batch["微博正文"].tolist()
            current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            
            try:
                # 情感分析
                scores, _ = analyze_sentiment(messages)
                
                if not scores:
                    print(f"[{current_time}] 警告: 批次 {i}-{i+batch_size} 分析返回空结果，跳过...")
                    continue
                
                # 更新情感分析结果
                for score in scores:
                    idx = score["id"] - 1  # 转换为0-based索引
                    absolute_idx = i + idx
                    processed_indices.append(absolute_idx)
                    df.at[absolute_idx, "sentiment"] = score["sentiment"]
                    df.at[absolute_idx, "intensity"] = score["intensity"]
                
                # 保存进度 (每隔10批次保存一次)
                if (i // batch_size) % 10 == 0:
                    df.to_csv(output_file, index=False, encoding='utf_8_sig')
                    save_progress(output_file, i + batch_size, processed_indices)
                
                # 打印进度
                progress = (i + batch_size) / len(df) * 100
                print(f"[{current_time}] 进度: {min(i + batch_size, len(df))}/{len(df)} ({progress:.1f}%) | "
                      f"当前批次: {i}-{min(i + batch_size, len(df))}")
                
                # time.sleep(0.5)  # API限速
                
            except Exception as batch_error:
                print(f"[{current_time}] 处理批次 {i}-{i+batch_size} 时出错: {str(batch_error)}")
                save_progress(output_file, i, processed_indices)
                df.to_csv(output_file, index=False, encoding='utf_8_sig')
                print("当前进度和结果已保存，程序退出")
                return
        
        # 完成处理，保存最终结果并清理进度文件
        df.to_csv(output_file, index=False, encoding='utf_8_sig')
        progress_file = f"{output_file}.progress"
        if os.path.exists(progress_file):
            os.remove(progress_file)
        print(f"分析完成，结果已保存到 {output_file}")
        
    except Exception as e:
        print(f"程序出错: {str(e)}")
        raise

if __name__ == "__main__":
    # analyze_weibo_emotions("processed_data/blog_data.csv", "emotion_analysis_output/blog_data_v2.csv")
    analyze_weibo_emotions("processed_data/merged_data.csv", "emotion_analysis_output/merged_data_v2.csv")