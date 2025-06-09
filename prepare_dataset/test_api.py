import json
import re
import time
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx", # 替换成用户的硅基流动api
    # api_key="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",
    base_url="https://api.siliconflow.cn/v1"
    # base_url="https://openrouter.ai/api/v1"
)

# 默认系统提示词
DEFAULT_SYSTEM_PROMPT = """作为专业的情感分析模型，请按以下要求对带编号的评论进行精细评分：

评分维度：
1. 情绪倾向（0.00-1.00）：
   - 0.00-0.30：明显负面（如愤怒、失望）
   - 0.31-0.69：中性或混合情绪（需根据细微差别调整，如轻微负面0.45，轻微正面0.55）
   - 0.70-1.00：明显正面（如赞扬、喜悦）
   * 示例：讽刺性表扬应接近0.25，理性分析接近0.50

2. 情绪强度（0.00-1.00）：
   - 0.00-0.30：平和陈述/客观描述（如产品参数讨论）
   - 0.31-0.70：带有情绪修饰（如"还不错"→0.4，"非常满意"→0.6）
   - 0.71-1.00：强烈情感表达（如感叹号、质问、情绪化词汇）

评分规则：
1. 必须保留两位小数（如0.25而非0.25）
2. 优先使用中间值（如0.83比0.80更可取）
3. 极端值(0.00/1.00)仅用于明确无误的情况
4. 注意：
   - 反问句强度≥0.65
   - 连续感叹号每个增加0.15强度
   - 理性分析倾向接近0.50，强度≤0.30

请返回严格遵循此结构的JSON（不带json前缀，禁止输出除了json之外的任何文字）：
{
    "scores": [
        {"id": 1, "sentiment": x.xx, "intensity": x.xx},
        ...
    ]
}

请现在开始分析以下输入内容：
"""

# DEFAULT_SYSTEM_PROMPT = """作为专业的情感分析模型，你需要根据Russell情绪模型对输入的每句话进行情绪分类和分析。请严格使用以下26种情绪分类：

# Happy
# Delighted
# Excited
# Astonished
# Aroused
# Tense
# Alarmed
# Angry
# Afraid
# Annoyed
# Distressed
# Frustrated
# Miserable
# Sad
# Gloomy
# Depressed
# Bored
# Droopy
# Tired
# Sleepy
# Calm
# Serene
# Pleased
# Content
# At Ease
# Relaxed

# 分析规则：
# 1. 每句话可能包含1-3种主要情绪
# 2. 为每种识别出的情绪分配一个占比值(0-1)，所有情绪占比总和为1
# 3. 优先考虑最强烈/最明显的情绪，非必要不添加多个情绪
# 4. 对于确实无明显情绪倾向的句子，选择最接近的低唤醒情绪（如Serene/Bored/Calm）
# 5. 不可添加以上列举范围之外的情绪

# 输出要求：
# 对每句话输出一个JSON对象，包含：
# - 原始文本引用
# - 识别出的情绪及其占比（严格使用上述26种情绪）

# 示例输出（不带json前缀，禁止输出除了json之外的任何文字）：
# {
#   "scores": [
#     {"id": 1, "emotions": {"Happy": 0.3, "Excited": 0.7}},
#     {"id": 2, "emotions": {"Angry": 1.0}},
#     {"id": 3, "emotions": {"Calm": 0.5, "Content": 0.5}},
#     {"id": 4, "emotions": {"Bored": 0.6, "Tired": 0.4}},
#     {"id": 5, "emotions": {"Astonished": 0.5, "Afraid": 0.2, "Sad": 0.3}}
#   ]
# }


# 请现在开始分析以下输入内容：
# """


# def extract_response_data(response_text):
#     """
#     从响应文本中提取批量评分结果和思考过程
#     返回 (emotion_data, thinking)，其中emotion_data包含每条评论的emotions字典
#     """
#     scores = []
#     thinking = None
#     try:
#         # 直接解析JSON响应
#         data = json.loads(response_text)
#         scores = data.get("scores", [])
            
#         # 提取思考过程
#         think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
#         if think_match:
#             thinking = think_match.group(1).strip()
#     except Exception as e:
#         print(f"Error parsing response: {e}\nResponse text: {response_text}")
        
#     return scores, thinking

def extract_response_data(response_text):
    """
    从响应文本中提取批量评分结果和思考过程
    返回 (emotion_data, thinking)，其中emotion_data包含每条评论的情感和强度数据
    """
    emotion_data = []
    thinking = None
    try:
        # 直接解析JSON响应
        data = json.loads(response_text)
        scores = data.get("scores", [])
        
        # 提取每条评论的情感和强度数据
        for score in scores:
            entry = {
                "id": score.get("id"),
                "sentiment": score.get("sentiment"),
                "intensity": score.get("intensity")
            }
            emotion_data.append(entry)
            
        # 提取思考过程
        think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
    except Exception as e:
        print(f"Error parsing response: {e}\nResponse text: {response_text}")
        
    return emotion_data, thinking

def analyze_sentiment(messages, system_prompt=DEFAULT_SYSTEM_PROMPT):
    """
    批量分析情感
    参数：
        messages: 评论列表
        system_prompt: 系统提示词，默认为DEFAULT_SYSTEM_PROMPT
    返回：
        (scores, thinking)
    """
    try:
        response = client.chat.completions.create(
            model="Qwen/Qwen2.5-14B-Instruct",
            # model="deepseek/deepseek-r1-distill-qwen-14b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": "\n".join([f"{i+1}. {msg}" for i, msg in enumerate(messages)])}
            ],
            temperature=0.1,
            max_tokens=4000
        )
        return extract_response_data(response.choices[0].message.content)
    except Exception as e:
        print(f"API调用失败：{str(e)}")
        return [], None

# def display_analysis(messages, scores, thinking, show_thinking=True):
#     """
#     显示批量分析结果
#     """
#     print(f"批量处理 {len(messages)} 条评论：")
#     if show_thinking and thinking:
#         print(f"思考过程：{thinking}")
#     for msg, data in zip(messages, scores):
#         print(f"\n{data['id']}. 评论：{msg}")
#         print("   情绪成分分析：")
#         for emotion, value in data['emotions'].items():
#             print(f"   - {emotion}: {value:.2f}")

def display_analysis(messages, scores, thinking, show_thinking=True):
    """
    显示批量分析结果
    """
    print(f"批量处理 {len(messages)} 条评论：")
    if show_thinking and thinking:
        print(f"思考过程：{thinking}")
    for msg, data in zip(messages, scores):
        print(f"\n{data['id']}. 评论：{msg}")
        print("   情绪成分分析：")
        print(f"   - 情感 (sentiment): {data['sentiment']:.2f}")
        print(f"   - 强度 (intensity): {data['intensity']:.2f}")

if __name__ == "__main__":
    # 汇总所有测试消息
    test_messages = [
        "今年考研英语一机构老师们乱成一锅粥了，赶紧趁热喝了吧哈哈哈哈哈哈哈哈，数学老师也参与这场战斗了，英一事变这场闹剧到底什么时候结束呢，谁会是最后的赢家呢，让我们拭目以待！！英一出题组这下你们满意了吧哈哈哈哈哈哈哈哈哈哈哈",
        "#考研英语一##田静##何凯文# 你们两个真是能力差人品更是让人恶心 考完英语的晚上分明刷到了某位姐发的视频 说什么只是微微难 当时英语一在考场上就已经让我觉得不对劲 结果刷到视频眼泪直接哗啦啦还以为就我今年英语这么烂 好嘛！成绩出来了一个发假成绩一个找一堆理由不敢发 是牛的！水货老师们 ​ ",
        "田静 周思成 何凯文 吃瓜 群星闪耀时刻",
        "心累了，受够这些新闻了",
        "简单理解这事，就是教育圈的'猫一杯'。猫一杯一个娱乐博主只是编造了一个小学生作业的段子，就被封杀了，而@何凯文 老师面对的是千千万万的即将接受高等教育的研究生们，孰轻孰重，所以风暴还没起来，真正的审判在酝酿中#何凯文# #何凯文上午退出考研教育下午开播收礼#"
    ]
    
    start_time = time.time()
    scores, thinking = analyze_sentiment(test_messages)
    end_time = time.time()
    display_analysis(test_messages, scores, thinking, show_thinking=True)
    print(f"平均每条评论处理时间：{(end_time - start_time)/len(test_messages):.2f}秒")
    

