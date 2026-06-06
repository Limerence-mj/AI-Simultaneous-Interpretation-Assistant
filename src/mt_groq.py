"""
Groq Llama 3 流式翻译 — 实时流式输出中文译文
"""
from typing import Callable, Optional


SYSTEM_PROMPT = """你是一个专业的英译中同声传译员。规则：
1. 只输出中文译文，不要任何解释
2. 保持口语化、自然流畅
3. 根据上下文纠正 ASR 可能的小错误
4. 如果输入不完整，输出当前能翻译的部分
5. 保持简洁，不要添加原文没有的内容"""


class GroqMT:
    """Groq Llama 3 流式翻译引擎"""

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        from openai import OpenAI
        self._client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )
        self._model = model
        self._available = True

    @property
    def is_available(self) -> bool:
        return self._available

    def translate_stream(self, en_text: str, on_chunk: Callable[[str], None]):
        """
        流式翻译：每收到一段中文就回调 on_chunk

        Args:
            en_text: 英文原文
            on_chunk: 回调 (chinese_text_chunk: str)
        """
        try:
            stream = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": en_text},
                ],
                stream=True,
                temperature=0.1,
                max_tokens=256,
            )
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    on_chunk(chunk.choices[0].delta.content)
        except Exception as e:
            print(f"[GroqMT] 翻译失败: {e}")

    def translate(self, en_text: str) -> Optional[str]:
        """非流式翻译，返回完整译文"""
        result = []
        self.translate_stream(en_text, result.append)
        return "".join(result) if result else None
