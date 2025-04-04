import json
import re
import time
from openai import OpenAI

# 初始化客户端
client = OpenAI(
    api_key="sk-taamkskxomltqzkeebgckdhihzrfkzxakjsvrgsckjubatpi",
    base_url="https://api.siliconflow.cn/v1"
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

请返回严格遵循此结构的JSON：
{
    "scores": [
        {"id": 1, "sentiment": x.xx, "intensity": x.xx},
        ...
    ]
}"""


def extract_response_data(response_text):
    """
    从响应文本中提取批量评分结果和思考过程
    返回 (sentiment_data, thinking)，其中sentiment_data包含每条评论的sentiment和intensity
    """
    scores = []
    thinking = None
    
    try:
        # 提取分数
        json_match = re.search(r'```json\s*({.*?})\s*```', response_text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
            data = json.loads(json_str)
            scores = data.get("scores", [])
            
        # 提取思考过程
        think_match = re.search(r'<think>(.*?)</think>', response_text, re.DOTALL)
        if think_match:
            thinking = think_match.group(1).strip()
    except:
        pass
        
    return scores, thinking

def emotional_score(valence, arousal):
    """
    将sentiment和intensity融合并映射到0-10分数区间
    使用情绪煽动计算公式：arousal * (4 * (valence - 0.5)**2)
    """
    return arousal * (4 * (valence - 0.5)**2)

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
            model="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
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

def display_analysis(messages, scores, thinking, show_thinking=True):
    """
    显示批量分析结果
    """
    print(f"批量处理 {len(messages)} 条评论：")
    if show_thinking and thinking:
        print(f"思考过程：{thinking}")
    for msg, data in zip(messages, scores):
        score = emotional_score(data['sentiment'], data['intensity'])
        print(f"{data['id']}. 评论：{msg}")
        print(f"   情感倾向：{data['sentiment']:.2f}")
        print(f"   情感强度：{data['intensity']:.2f}")
        print(f"   综合得分：{score}")

if __name__ == "__main__":
    # 汇总所有测试消息
    test_messages = [
        "简直是垃圾，浪费钱！",
        "这个产品太棒了！",
        "质量一般，但是价格便宜",
        "周思成和何凯文这件事，有人说最后受益者肯定不是周思成，而是渔翁得利其他老师们捡漏。还有人蹦跶出来开始撕他了，但是他本人八字从前三柱来看本身甲己合，但是今年乙巳合，对于他而言越撕名气越大。管他其他老师怎么抨击他，又开始爆料，都没用。就是越撕他越有名，该喜欢他的越来越喜欢他。翻车？不存在的。",
        "今年考研英语一机构老师们乱成一锅粥了，赶紧趁热喝了吧哈哈哈哈哈哈哈哈，数学老师也参与这场战斗了，英一事变这场闹剧到底什么时候结束呢，谁会是最后的赢家呢，让我们拭目以待！！英一出题组这下你们满意了吧哈哈哈哈哈哈哈哈哈哈哈",
        "#考研英语一##田静##何凯文# 你们两个真是能力差人品更是让人恶心 考完英语的晚上分明刷到了某位姐发的视频 说什么只是微微难 当时英语一在考场上就已经让我觉得不对劲 结果刷到视频眼泪直接哗啦啦还以为就我今年英语这么烂 好嘛！成绩出来了一个发假成绩一个找一堆理由不敢发 是牛的！水货老师们 ​ ",
        "田静 周思成 何凯文 吃瓜 群星闪耀时刻",
        "简单理解这事，就是教育圈的'猫一杯'。猫一杯一个娱乐博主只是编造了一个小学生作业的段子，就被封杀了，而@何凯文 老师面对的是千千万万的即将接受高等教育的研究生们，孰轻孰重，所以风暴还没起来，真正的审判在酝酿中#何凯文# #何凯文上午退出考研教育下午开播收礼#"
    ]
    
    start_time = time.time()
    scores, thinking = analyze_sentiment(test_messages)
    end_time = time.time()
    display_analysis(test_messages, scores, thinking, show_thinking=True)
    print(f"平均每条评论处理时间：{(end_time - start_time)/len(test_messages):.2f}秒")
    

