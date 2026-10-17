
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 读取 API Key
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    raise ValueError("请先配置 API Key")

client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 两次测试使用同一段通知
notice = "本周五前提交高数作业，作业包括第三章习题1、3、5，请提交到学习平台。"

# 简单提示词
simple_prompt = "请总结以下通知：\n" + notice

# 详细提示词
detailed_prompt = """
你是一个校园信息助手，请整理下面的通知。

按照以下格式输出：

一句话总结：
关键信息：
待办事项：

要求：
1. 回答简单清楚。
2. 不要编造原文中没有的信息。
3. 如果没有明确给出时间或地点，写“未提及”。
4. 如果没有待办事项，写“无”。

通知：
""" + notice


def call_model(prompt):
    response = client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content


# 第一次测试
print("========== 简单提示词 ==========")
print("输入：", simple_prompt)

simple_result = call_model(simple_prompt)

print("\n模型回答：")
print(simple_result)

# 第二次测试
print("\n========== 详细提示词 ==========")
print("输入：", detailed_prompt)

detailed_result = call_model(detailed_prompt)

print("\n模型回答：")
print(detailed_result)
