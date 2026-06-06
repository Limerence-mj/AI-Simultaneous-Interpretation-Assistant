"""
Groq API 管线 — 流式同传
VAD 切句 → Groq Whisper(快速转录) → Groq Llama(流式翻译) → 字幕实时更新
"""
import threading
import queue
import time
from typing import Callable, Optional

import numpy as np

from src.vad import VADProcessor
from src.asr_groq import GroqASR
from src.mt_groq import GroqMT


class APIPipeline:
    """
    Groq API 同传管线

    架构：
    - VAD 本地切句（2-4秒短段）
    - Groq Whisper 转录（~500ms，极快）
    - Groq Llama 流式翻译（实时输出中文字幕）
    """

    def __init__(self, api_key: str, on_subtitle: Callable[[str], None]):
        self._on_subtitle = on_subtitle
        self._asr = GroqASR(api_key=api_key)
        self._mt = GroqMT(api_key=api_key)
        self._vad = VADProcessor(
            min_silence_ms=200, min_speech_ms=150, max_speech_ms=2500
        )
        self._running = False
        self._queue: queue.Queue = queue.Queue(maxsize=10)
        self._worker: Optional[threading.Thread] = None
        self._latest_zh = ""

    def start(self):
        self._running = True
        self._worker = threading.Thread(target=self._loop, name="API-Pipeline", daemon=True)
        self._worker.start()
        print("[API Pipeline] Groq 同传管线就绪")

    def feed(self, audio_chunk: np.ndarray):
        if not self._running:
            return
        segment = self._vad.process(audio_chunk)
        if segment:
            try:
                self._queue.put_nowait(segment)
            except queue.Full:
                pass

    def _loop(self):
        while self._running or not self._queue.empty():
            try:
                segment = self._queue.get(timeout=0.3)
            except queue.Empty:
                continue

            # 跳过积压
            latest = segment
            while True:
                try:
                    latest = self._queue.get_nowait()
                except queue.Empty:
                    break

            # ASR
            en_text = self._asr.transcribe(latest.audio_data)
            if not en_text:
                continue

            # 流式翻译
            self._latest_zh = ""
            self._mt.translate_stream(en_text, self._on_translate_chunk)

    def _on_translate_chunk(self, chunk: str):
        self._latest_zh += chunk
        if self._on_subtitle:
            self._on_subtitle(self._latest_zh)

    def finish(self):
        self._running = False
        if self._worker:
            self._worker.join(timeout=10)
        flushed = self._vad.flush()
        if flushed:
            en = self._asr.transcribe(flushed.audio_data)
            if en:
                zh = self._mt.translate(en)
                if zh and self._on_subtitle:
                    self._on_subtitle(zh)

    def stop(self):
        self.finish()
