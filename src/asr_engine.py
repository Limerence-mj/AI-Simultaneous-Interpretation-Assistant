"""
语音识别引擎模块 — ASREngine
基于 faster-whisper，将语音段 (SpeechSegment) 转写为英文文本
支持 CPU/GPU 自动切换、模型降级、异步队列处理
"""
import os
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# ─── 强制模型缓存落在项目目录 ───
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(MODELS_DIR / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(MODELS_DIR))


@dataclass
class ASRResult:
    """语音识别结果"""
    segment_id: int
    en_text: str
    confidence: float          # 0.0 ~ 1.0
    start_ms: float
    end_ms: float


class ASREngine:
    """
    ASR 引擎：加载 Whisper 模型，提供同步/异步转写接口

    用法:
        engine = ASREngine(model_size="small", device="cpu")
        engine.start()                          # 启动工作线程
        engine.submit(speech_segment)           # 提交语音段
        result = engine.get_result()            # 获取识别结果 (阻塞)
        engine.stop()                           # 停止工作线程
    """

    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        beam_size: int = 5,
    ):
        """
        Args:
            model_size: "tiny" | "small" | "medium"
            device: "cpu" | "cuda"
            compute_type: "int8" (CPU) | "float16" (GPU)
            cpu_threads: CPU 推理线程数
            beam_size: beam search 宽度
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.beam_size = beam_size

        self._model = None
        self._input_queue: queue.Queue = queue.Queue()
        self._output_queue: queue.Queue = queue.Queue()
        self._worker_thread: Optional[threading.Thread] = None
        self._running = False
        self._load_time: float = 0.0

    # ─── 模型加载 ───

    def load_model(self) -> float:
        """加载 Whisper 模型，返回加载耗时 (秒)"""
        from faster_whisper import WhisperModel

        t0 = time.time()
        try:
            self._model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                cpu_threads=4,
                download_root=str(MODELS_DIR),
            )
        except Exception:
            # 降级：small → tiny
            if self.model_size == "small":
                print(f"[ASR] small 模型加载失败，降级尝试 tiny...")
                self.model_size = "tiny"
                self._model = WhisperModel(
                    "tiny",
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=4,
                    download_root=str(MODELS_DIR),
                )
            else:
                raise

        self._load_time = time.time() - t0
        return self._load_time

    # ─── 同步转写 ───

    def transcribe(self, audio_data: bytes, sample_rate: int = 16000) -> ASRResult:
        """
        同步转写一段音频（PCM 16-bit 字节）

        Args:
            audio_data: PCM 16-bit little-endian 字节
            sample_rate: 采样率

        Returns:
            ASRResult 含识别文本和置信度
        """
        if self._model is None:
            raise RuntimeError("模型未加载，请先调用 load_model()")

        # 空输入 / 极短输入保护
        if not audio_data or len(audio_data) < 32:
            return ASRResult(
                segment_id=0, en_text="", confidence=0.0,
                start_ms=0.0, end_ms=0.0,
            )

        # PCM 16-bit → float32 numpy array
        audio_i16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_f32 = audio_i16.astype(np.float32) / 32767.0

        # 检测纯静音（RMS 能量极低），跳过 ASR 避免幻觉
        rms = float(np.sqrt(np.mean(audio_f32 ** 2)))
        if rms < 0.0005:  # -66 dBFS 以下视为静音
            return ASRResult(
                segment_id=0, en_text="", confidence=0.0,
                start_ms=0.0, end_ms=0.0,
            )

        # 转写
        segments_iter, info = self._model.transcribe(
            audio_f32,
            language="en",
            beam_size=self.beam_size,
            best_of=1,
            vad_filter=False,
            condition_on_previous_text=False,
            temperature=0.0,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
        )

        # 合并所有片段文本
        texts = []
        total_log_prob = 0.0
        seg_count = 0

        for seg in segments_iter:
            text = seg.text.strip()
            if text:  # 过滤空文本
                texts.append(text)
                total_log_prob += seg.avg_logprob
                seg_count += 1

        en_text = " ".join(texts)
        confidence = np.exp(total_log_prob / max(seg_count, 1))  # 对数概率→线性
        # clamp 置信度到 [0, 1]
        confidence = max(0.0, min(1.0, float(confidence)))

        return ASRResult(
            segment_id=0,               # 由调用者填充
            en_text=en_text,
            confidence=confidence,
            start_ms=0.0,               # 由调用者填充
            end_ms=0.0,
        )

    # ─── 异步队列接口 ───

    def start(self):
        """启动工作线程，开始处理提交的语音段"""
        if self._model is None:
            self.load_model()
        self._running = True
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="ASR-Worker",
            daemon=True,
        )
        self._worker_thread.start()

    def stop(self):
        """停止工作线程"""
        self._running = False
        # 放入哨兵值唤醒阻塞的 worker
        self._input_queue.put(None)
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=10.0)
        self._worker_thread = None

    def submit(self, speech_segment) -> None:
        """
        提交语音段到处理队列（非阻塞）

        Args:
            speech_segment: SpeechSegment 对象 (from src.vad)
        """
        self._input_queue.put(speech_segment)

    def get_result(self, timeout: Optional[float] = None) -> Optional[ASRResult]:
        """
        获取下一个识别结果（阻塞）

        Args:
            timeout: 超时秒数，None = 无限等待

        Returns:
            ASRResult 或 None（超时/停止）
        """
        try:
            return self._output_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def pending_count(self) -> int:
        """队列中待处理的语音段数量"""
        return self._input_queue.qsize()

    # ─── 内部 ───

    def _worker_loop(self):
        """工作线程主循环"""
        while self._running:
            segment = self._input_queue.get()
            if segment is None:          # 哨兵，停止信号
                break

            try:
                t0 = time.time()
                result = self.transcribe(segment.audio_data)
                inference_ms = (time.time() - t0) * 1000

                # 填入元信息
                result.segment_id = segment.id
                result.start_ms = segment.start_ms
                result.end_ms = segment.end_ms

                self._output_queue.put(result)

            except Exception as e:
                # 单次失败不中断整体，将错误文本放入结果
                print(f"[ASR] 识别失败 (seg #{segment.id}): {e}")
                error_result = ASRResult(
                    segment_id=segment.id,
                    en_text=f"[ASR_ERROR: {e}]",
                    confidence=0.0,
                    start_ms=segment.start_ms,
                    end_ms=segment.end_ms,
                )
                self._output_queue.put(error_result)

        # 清空队列中剩余项（避免内存泄漏）
        while not self._input_queue.empty():
            try:
                self._input_queue.get_nowait()
            except queue.Empty:
                break
