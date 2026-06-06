"""
流式翻译管线 v4 — 真正的实时同传
Sherpa-ONNX 流式 ASR + opus-mt 翻译 = 低延迟
Partial 结果仅显原文，Final 结果翻译+显示+TTS
"""
from typing import Callable, Optional
import numpy as np

from src.asr_engine_streaming import StreamingASREngine
from src.mt_engine import MTEngine
from src.state_manager import StateManager


class StreamingPipeline:
    """
    流式同传管线 v4

    架构:
    - 音频 → StreamingASREngine (Sherpa-ONNX, <500ms)
    - partial: 显示英文原文（可选）
    - final: 翻译 + 更新字幕 + TTS 回调
    """

    def __init__(self, on_subtitle: Optional[Callable[[str], None]] = None,
                 on_tts: Optional[Callable[[str], None]] = None):
        self._on_subtitle = on_subtitle
        self._on_tts = on_tts
        self._state = StateManager()
        self._mt = MTEngine()
        self._asr: Optional[StreamingASREngine] = None
        self._running = False

    def start(self):
        self._mt.load_model()
        self._asr = StreamingASREngine(
            on_partial=self._on_asr_partial,
            on_final=self._on_asr_final,
        )
        if not self._asr.is_available:
            raise RuntimeError("流式 ASR 引擎不可用")
        self._asr.start()
        self._running = True
        print("[StreamingPipeline] 流式同传管线就绪")

    def feed_audio(self, audio_chunk: np.ndarray):
        if not self._running or self._asr is None:
            return
        self._asr.feed_audio(audio_chunk)

    def _on_asr_partial(self, text: str):
        """部分识别结果 — 不做翻译，仅可选显示原文（暂不处理）"""
        pass  # 不刷字幕，等 final 结果

    def _on_asr_final(self, text: str):
        """句子结束 → 翻译 + 字幕 + TTS"""
        if not text or len(text.strip()) < 2:
            return
        try:
            zh = self._mt.translate(text)
            if zh:
                self._state.set_subtitle(zh)
                if self._on_subtitle:
                    self._on_subtitle(zh)
                if self._on_tts:
                    self._on_tts(zh)
        except Exception as e:
            print(f"[StreamingPipeline] 翻译失败: {e}")

    def stop(self):
        self._running = False
        if self._asr:
            self._asr.stop()

    @property
    def is_running(self) -> bool:
        return self._running
