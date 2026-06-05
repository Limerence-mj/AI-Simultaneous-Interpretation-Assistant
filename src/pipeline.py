"""
VAD → ASR 管道骨架
连接语音活动检测与语音识别，提供文件处理和流式处理的统一接口
"""
import time
from typing import List, Optional

import numpy as np

from src.vad import VADProcessor, SpeechSegment
from src.asr_engine import ASREngine, ASRResult


class Pipeline:
    """
    VAD → ASR 处理管道

    用法 — 处理音频文件:
        pipeline = Pipeline()
        results = pipeline.process_file("audio.wav")
        for r in results:
            print(f"[{r.start_ms:.0f}-{r.end_ms:.0f}ms] {r.en_text}")

    用法 — 流式处理:
        pipeline = Pipeline()
        pipeline.start()
        for chunk in audio_stream:
            pipeline.feed(chunk)
        results = pipeline.finish()
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
    ):
        # 初始化 VAD
        self.vad = VADProcessor(
            min_silence_ms=500,
            min_speech_ms=300,
            max_speech_ms=15000,
        )

        # 初始化 ASR（延迟加载）
        self.asr = ASREngine(
            model_size=model_size,
            device=device,
            compute_type="int8" if device == "cpu" else "float16",
        )

        # 流式处理状态
        self._results: List[ASRResult] = []
        self._block_size = 1024  # 与 AudioCapture.BLOCK_SIZE 一致

    # ─── 文件处理 ───

    def process_file(self, audio_array: np.ndarray, sample_rate: int = 16000) -> List[ASRResult]:
        """
        处理完整音频文件（numpy float32 数组）

        Args:
            audio_array: 1D float32 数组, 16kHz
            sample_rate: 采样率

        Returns:
            ASRResult 列表，按时间排序
        """
        results: List[ASRResult] = []

        # 确保模型已加载
        if not self.asr._model:
            load_time = self.asr.load_model()
            print(f"[Pipeline] ASR 模型加载: {load_time:.1f}s ({self.asr.model_size})")

        # 模拟流式输入
        t_total_start = time.time()
        total_samples = len(audio_array)

        for offset in range(0, total_samples, self._block_size):
            chunk = audio_array[offset:offset + self._block_size]
            if len(chunk) < self._block_size:
                pad = np.zeros(self._block_size - len(chunk), dtype=np.float32)
                chunk = np.concatenate([chunk, pad])

            # VAD 切句
            segment = self.vad.process(chunk)
            if segment is not None:
                # ASR 识别
                asr_result = self.asr.transcribe(segment.audio_data)
                asr_result.segment_id = segment.id
                asr_result.start_ms = segment.start_ms
                asr_result.end_ms = segment.end_ms
                results.append(asr_result)

        # Flush 末尾未完成段
        flushed = self.vad.flush()
        if flushed is not None:
            asr_result = self.asr.transcribe(flushed.audio_data)
            asr_result.segment_id = flushed.id
            asr_result.start_ms = flushed.start_ms
            asr_result.end_ms = flushed.end_ms
            results.append(asr_result)

        elapsed = time.time() - t_total_start
        audio_duration = total_samples / sample_rate
        print(f"[Pipeline] 处理 {audio_duration:.1f}s 音频, "
              f"耗时 {elapsed:.1f}s, 产出 {len(results)} 句")

        return results

    # ─── 流式处理 ───

    def start(self):
        """启动流式管道（加载 ASR 模型）"""
        if not self.asr._model:
            t0 = time.time()
            self.asr.load_model()
            print(f"[Pipeline] 模型就绪 ({time.time()-t0:.1f}s)")

    def feed(self, audio_chunk: np.ndarray) -> Optional[ASRResult]:
        """
        送入一块音频数据，如有完整句子则返回 ASRResult

        Args:
            audio_chunk: 1D float32, 16kHz

        Returns:
            ASRResult 或 None
        """
        segment = self.vad.process(audio_chunk)
        if segment is None:
            return None

        result = self.asr.transcribe(segment.audio_data)
        result.segment_id = segment.id
        result.start_ms = segment.start_ms
        result.end_ms = segment.end_ms
        self._results.append(result)
        return result

    def finish(self) -> List[ASRResult]:
        """结束流式处理，flush 剩余段并返回全部结果"""
        flushed = self.vad.flush()
        if flushed is not None:
            result = self.asr.transcribe(flushed.audio_data)
            result.segment_id = flushed.id
            result.start_ms = flushed.start_ms
            result.end_ms = flushed.end_ms
            self._results.append(result)
        return self._results

    def reset(self):
        """重置管道状态"""
        self.vad.reset()
        self._results.clear()
