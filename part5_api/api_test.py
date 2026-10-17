
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

# 创建客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 调用模型
response = client.chat.completions.create(
    model="qwen-plus",
    messages=[
        {
            "role": "user",
            "content": "你好，请简单介绍一下什么是机器学习。"
        }
    ]
)

# 输出结果
print("模型回答：")
print(response.choices[0].message.content)
