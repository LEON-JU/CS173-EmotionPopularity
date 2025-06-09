import csv
from emotional_model import calculate_valence_arousal
from visualization.draw_EIS import improved_emotional_impact_score

def safe_float(s):
    """安全转换字符串为浮点数，空字符串返回0.0"""
    return float(s) if s.strip() else 0.0

# 情绪名称顺序（与emotional_model.py中EMOTION_ANGLES一致）
EMOTION_ORDER = [
    "Happy", "Delighted", "Excited", "Astonished", "Aroused", "Tense", "Alarmed", 
    "Angry", "Afraid", "Annoyed", "Distressed", "Frustrated", "Miserable", "Sad", 
    "Gloomy", "Depressed", "Bored", "Droopy", "Tired", "Sleepy", "Calm", "Serene", 
    "Pleased", "Content", "At Ease", "Relaxed"
]

def process_csv(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8', newline='') as outfile:
        
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        # 处理每一行
        for row in reader:
            # 前7个字段是元数据，后面是情绪值
            metadata = row[:8]
            emotion_values = list(map(safe_float, row[8:34]))  # 26个情绪值
            
            # 创建情绪字典
            emotion_dict = {
                emotion: intensity 
                for emotion, intensity in zip(EMOTION_ORDER, emotion_values)
            }
            
            # 计算效价和唤醒度
            valence, arousal = calculate_valence_arousal(emotion_dict)
            
            # 计算EIS
            eis = improved_emotional_impact_score(arousal, valence)
            
            # 写入新行（原数据+valence+arousal+EIS）
            new_row = row + [valence, arousal, eis]
            writer.writerow(new_row)

if __name__ == "__main__":
    input_file = "何凯文.csv"
    output_file = "何凯文_with_emotion.csv"
    process_csv(input_file, output_file)
    print(f"处理完成，结果已保存到 {output_file}，包含valence, arousal和EIS")