"""
Groq Whisper ASR — 将音频段上传 Groq 快速转录
延迟：~300-500ms/段（比本地 tiny 快 2 倍，比 small 快 6 倍）
"""
import io
import wave
import numpy as np
from typing import Optional


class GroqASR:
    """Groq Whisper API 语音识别"""

    def __init__(self, api_key: str, model: str = "whisper-large-v3"):
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

    def transcribe(self, audio_bytes: bytes) -> Optional[str]:
        """
        转录 PCM 16-bit 单声道 16kHz 音频 → 英文文本

        Args:
            audio_bytes: PCM 16-bit WAV 格式字节

        Returns:
            英文文本，失败返回 None
        """
        try:
            # 包装为 WAV 文件对象
            wav_io = io.BytesIO()
            with wave.open(wav_io, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(audio_bytes)
            wav_io.seek(0)
            wav_io.name = "audio.wav"

            transcript = self._client.audio.transcriptions.create(
                model=self._model,
                file=wav_io,
                response_format="text",
                language="en",
            )
            return transcript.strip()
        except Exception as e:
            print(f"[GroqASR] 转录失败: {e}")
            return None
