# 使用模型绑定工具
from langchain_deepseek import ChatDeepSeek

llm_deepseek = ChatDeepSeek(
    model="deepseek-chat",
    api_key="sk-f5109c1b919f47b2bca39f5055aa4ce0",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
