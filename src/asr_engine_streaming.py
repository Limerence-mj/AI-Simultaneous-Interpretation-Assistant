"""
流式语音识别引擎 — 基于 Sherpa-ONNX OnlineRecognizer
边听边识别，实时产出部分文本，延迟 < 500ms
"""
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
MODEL_DIR = PROJECT_ROOT / "models" / "sherpa_zipformer_en"


class StreamingASREngine:
    """流式 ASR 引擎"""

    def __init__(
        self,
        on_partial: Optional[Callable[[str], None]] = None,
        on_final: Optional[Callable[[str], None]] = None,
    ):
        self._on_partial = on_partial
        self._on_final = on_final
        self._recognizer = None
        self._stream = None
        self._available = False
        self._running = False
        self._init_recognizer()

    def _init_recognizer(self):
        """初始化 Sherpa-ONNX Zipformer 流式识别器"""
        try:
            from sherpa_onnx.online_recognizer import OnlineRecognizer

            encoder = str(MODEL_DIR / "encoder-epoch-99-avg-1.int8.onnx")
            decoder = str(MODEL_DIR / "decoder-epoch-99-avg-1.int8.onnx")
            joiner = str(MODEL_DIR / "joiner-epoch-99-avg-1.int8.onnx")
            tokens = str(MODEL_DIR / "tokens.txt")

            if not Path(encoder).exists():
                encoder = str(MODEL_DIR / "encoder-epoch-99-avg-1.onnx")
                decoder = str(MODEL_DIR / "decoder-epoch-99-avg-1.onnx")
                joiner = str(MODEL_DIR / "joiner-epoch-99-avg-1.onnx")

            self._recognizer = OnlineRecognizer.from_transducer(
                tokens=tokens,
                encoder=encoder,
                decoder=decoder,
                joiner=joiner,
                num_threads=4,
                sample_rate=16000,
                feature_dim=80,
                enable_endpoint_detection=True,
                rule1_min_trailing_silence=0.5,
                rule2_min_trailing_silence=1.2,
                rule3_min_utterance_length=0.8,
                decoding_method='greedy_search',
            )
            self._stream = self._recognizer.create_stream()
            self._available = True
            print(f"[StreamingASR] Zipformer 流式识别器就绪")
        except Exception as e:
            print(f"[StreamingASR] 初始化失败: {e}")
            import traceback
            traceback.print_exc()
            self._available = False

    @property
    def is_available(self) -> bool:
        return self._available

    def start(self):
        """开始识别"""
        if not self._available:
            return
        self._running = True
        if self._stream is None:
            self._stream = self._recognizer.create_stream()

    def feed_audio(self, audio_chunk: np.ndarray):
        """送入 16kHz 单声道 float32 音频。结果通过回调返回。"""
        if not self._running or not self._available:
            return
        try:
            samples = audio_chunk.astype(np.float32).flatten()
            self._stream.accept_waveform(16000, samples)

            while self._recognizer.is_ready(self._stream):
                self._recognizer.decode_stream(self._stream)

            partial = self._recognizer.get_result(self._stream)
            if partial and self._on_partial:
                self._on_partial(partial)

            if self._recognizer.is_endpoint(self._stream):
                final_text = self._recognizer.get_result(self._stream)
                self._recognizer.reset(self._stream)
                if final_text and self._on_final:
                    self._on_final(final_text)
        except Exception as e:
            print(f"[StreamingASR] feed 错误: {e}")

    def stop(self):
        """停止"""
        self._running = False
        if self._stream and self._recognizer:
            try:
                final = self._recognizer.get_result(self._stream)
                if final and self._on_final:
                    self._on_final(final)
            except Exception:
                pass

    def reset(self):
        if self._recognizer and self._stream:
            try:
                self._recognizer.reset(self._stream)
            except Exception:
                pass
