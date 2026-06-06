"""
VAD → ASR 管道 v3
faster-whisper + 短 VAD 窗口 + 单 ASR 线程
"""
import time
import threading
import queue
from typing import List, Optional, Callable

import numpy as np

from src.vad import VADProcessor, SpeechSegment
from src.asr_engine import ASREngine, ASRResult


class Pipeline:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        on_result: Optional[Callable[[ASRResult], None]] = None,
    ):
        self.on_result = on_result
        self.vad = VADProcessor(
            threshold=0.5,
            min_silence_ms=150,     # 150ms 停顿即切句，响应更快
            min_speech_ms=150,
            max_speech_ms=5000,    # 仅极端安全网，由自然停顿切句
        )
        self.asr = ASREngine(
            model_size=model_size,
            device=device,
            compute_type="int8" if device == "cpu" else "float16",
        )
        self._results: List[ASRResult] = []
        self._running = False
        self._asr_queue = queue.Queue(maxsize=10)  # 足够容纳 VAD 产出
        self._asr_worker: Optional[threading.Thread] = None

    def start(self):
        if not self.asr._model:
            t0 = time.time()
            self.asr.load_model()
            print(f"[Pipeline] ASR 就绪 ({time.time()-t0:.1f}s, {self.asr.model_size})")
        self._running = True
        self._asr_worker = threading.Thread(target=self._asr_loop, name="ASR-Worker", daemon=True)
        self._asr_worker.start()

    def feed(self, audio_chunk: np.ndarray) -> None:
        if not self._running:
            return
        segment = self.vad.process(audio_chunk)
        if segment is None:
            return
        try:
            self._asr_queue.put_nowait(segment)
        except queue.Full:
            try:
                self._asr_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._asr_queue.put_nowait(segment)
            except queue.Full:
                pass

    def _asr_loop(self):
        while self._running or not self._asr_queue.empty():
            try:
                segment = self._asr_queue.get(timeout=0.3)
            except queue.Empty:
                continue
            try:
                result = self.asr.transcribe(segment.audio_data)
                result.segment_id = segment.id
                result.start_ms = segment.start_ms
                result.end_ms = segment.end_ms
                self._results.append(result)
                if self.on_result:
                    self.on_result(result)
            except Exception as e:
                print(f"[Pipeline] ASR 失败: {e}")

    def finish(self) -> List[ASRResult]:
        self._running = False
        if self._asr_worker and self._asr_worker.is_alive():
            self._asr_worker.join(timeout=30.0)  # 等待所有段处理完
        flushed = self.vad.flush()
        if flushed is not None:
            try:
                result = self.asr.transcribe(flushed.audio_data)
                result.segment_id = flushed.id
                result.start_ms = flushed.start_ms
                result.end_ms = flushed.end_ms
                self._results.append(result)
                if self.on_result:
                    self.on_result(result)
            except Exception as e:
                print(f"[Pipeline] Flush 失败: {e}")
        return self._results

    def reset(self):
        self.vad.reset()
        self._results.clear()
        while not self._asr_queue.empty():
            try:
                self._asr_queue.get_nowait()
            except queue.Empty:
                break
