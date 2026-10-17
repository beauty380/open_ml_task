
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 读取 API Key
BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("DASHSCOPE_API_KEY")

if not api_key:
    raise ValueError("请先在 .env 文件中配置 API Key")

# 创建客户端
client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 提示词
system_prompt = """
你是一个校园信息助手，负责整理校园通知和回答学习问题。

请按照下面的格式回答：

一句话总结：
关键信息：
待办事项：

要求：
1. 回答简单清楚，不要添加多余内容。
2. 不能编造原文中没有的信息。
3. 如果没有明确的时间或地点，写“未提及”。
4. 如果没有需要完成的事情，写“无”。
5. 遇到学习问题时，简单回答问题，并在待办事项中给出必要的学习建议。
"""

# 命令行交互
print("校园信息助手")
print("输入 exit 退出程序")

while True:
    user_input = input("\n请输入校园信息：").strip()

    if user_input.lower() == "exit":
        print("程序已退出")
        break

    if not user_input:
        print("请输入内容")
        continue

    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input}
            ],
            temperature=0
        )

        print("\n模型回答：")
        print(response.choices[0].message.content)

    except Exception as e:
        print("调用失败：", e)
