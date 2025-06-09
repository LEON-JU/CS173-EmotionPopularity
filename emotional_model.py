import math
import numpy as np

# 情绪角度映射（单位：度）
EMOTION_ANGLES = {
    "Happy": 7.8,
    "Delighted": 24.9,
    "Excited": 48.6,
    "Astonished": 69.8,
    "Aroused": 73.8,
    "Tense": 92.8,
    "Alarmed": 96.5,
    "Angry": 105.1,
    "Afraid": 115.4,
    "Annoyed": 122.5,
    "Distressed": 131.3,
    "Frustrated": 140.0,
    "Miserable": 188.7,
    "Sad": 207.5,
    "Gloomy": 216.2,
    "Depressed": 225.0,
    "Bored": 233.3,
    "Droopy": 256.6,
    "Tired": 267.7,
    "Sleepy": 271.9,
    "Calm": 316.2,
    "Serene": 328.6,
    "Pleased": 353.2,
    "Content": 326.2,
    "At Ease": 318.8,
    "Relaxed": 318.8
}

def calculate_valence_arousal(emotion_dict):
    """
    计算情绪字典的综合效价和唤醒度
    
    参数:
        emotion_dict (dict): 情绪名称到强度的映射，如 {"Happy": 0.3, "Excited": 0.7}
    
    返回:
        tuple: (效价, 唤醒度)
    """
    total_valence = 0.0
    total_arousal = 0.0
    
    for emotion, intensity in emotion_dict.items():
        if emotion not in EMOTION_ANGLES:
            continue
            
        # 将角度转换为弧度
        angle_rad = math.radians(EMOTION_ANGLES[emotion])
        
        # 计算该情绪在效价和唤醒度上的分量
        valence = math.cos(angle_rad) * intensity
        arousal = math.sin(angle_rad) * intensity
        
        total_valence += valence
        total_arousal += arousal
    
    return normalize_valence_arousal(total_valence, total_arousal)

def normalize_valence_arousal(valence, arousal):
    """
    将效价和唤醒度从[-1, 1]归一化到[0, 1]
    
    参数:
        valence (float): 原始效价
        arousal (float): 原始唤醒度
    
    返回:
        tuple: 归一化后的效价和唤醒度
    """
    normalized_valence = (valence + 1) / 2
    normalized_arousal = (arousal + 1) / 2
    return (normalized_valence, normalized_arousal)



if __name__ == "__main__":
    # 示例用法
    emotions = {"Happy": 0.3, "Excited": 0.7}
    valence, arousal = calculate_valence_arousal(emotions)
    print(f"原始效价和唤醒度: {valence:.3f}, {arousal:.3f}")