"""
语音活动检测 (VAD) 模块 — VADProcessor
基于 silero-vad 进行流式语音切句，将连续音频流切分为完整短句片段
"""
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from silero_vad import VADIterator, load_silero_vad


@dataclass
class SpeechSegment:
    """语音段"""
    id: int
    start_ms: float          # 开始时间，毫秒
    end_ms: float            # 结束时间，毫秒
    audio_data: bytes        # 原始音频数据 (PCM 16-bit little-endian)
    duration_ms: float       # 时长，毫秒


# ─── 阈值常量 (设计文档 6.2.2) ───
SILENCE_THRESHOLD_DBFS = -35      # 静音判定 (dBFS)，预留，silero-vad 使用概率阈值
VAD_THRESHOLD = 0.5               # 语音概率阈值
MIN_SILENCE_DURATION_MS = 500     # 最小静音时长触发切句
MIN_SPEECH_DURATION_MS = 300      # 最短语音段（过滤杂音）
MAX_SPEECH_DURATION_MS = 15000    # 最长单句（强制切断）
SPEECH_PAD_MS = 30                # 语音段前后填充
SAMPLE_RATE = 16000
WINDOW_SIZE = 512                 # silero-vad 要求 16kHz 时每块 512 样本


class VADProcessor:
    """
    流式语音活动检测处理器

    使用 silero-vad VADIterator 进行实时语音检测，
    内置状态机管理语音/静音转换，输出完整语音段。

    用法:
        vad = VADProcessor()
        for chunk in audio_stream:
            segment = vad.process(chunk)
            if segment:
                print(f"语音段: {segment.start_ms:.0f}-{segment.end_ms:.0f}ms")
    """

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        threshold: float = VAD_THRESHOLD,
        min_silence_ms: int = MIN_SILENCE_DURATION_MS,
        min_speech_ms: int = MIN_SPEECH_DURATION_MS,
        max_speech_ms: int = MAX_SPEECH_DURATION_MS,
    ):
        self.sample_rate = sample_rate
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self.max_speech_samples = int(sample_rate * max_speech_ms / 1000)

        # 加载 silero-vad 模型
        self._model = load_silero_vad()
        self._vad_iter = VADIterator(
            self._model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_ms,
            speech_pad_ms=SPEECH_PAD_MS,
        )

        # 音频累积缓冲区 — 按时间顺序存储 (sample_offset, audio_array)
        self._audio_chunks: List[tuple] = []  # [(start_sample, np.array), ...]
        self._total_samples = 0               # 已接收总样本数
        self._trimmed_offset = 0              # 已裁剪的样本偏移

        # 当前语音段追踪
        self._current_speech_start: Optional[int] = None  # 绝对样本位置
        self._segment_id = 0
        self._triggered = False

    def process(self, audio_chunk: np.ndarray) -> Optional[SpeechSegment]:
        """
        处理音频块，返回完整语音段（如果有）

        Args:
            audio_chunk: 1D numpy float32 数组, 16kHz

        Returns:
            SpeechSegment 当检测到一个完整语音段结束时; 否则 None
        """
        chunk_len = len(audio_chunk)

        # 存储音频数据（含绝对位置）
        self._audio_chunks.append((self._total_samples, audio_chunk.copy()))
        self._total_samples += chunk_len

        # 将音频累积为 torch tensor，以 512 样本窗口送入 VADIterator
        result = self._process_vad(audio_chunk)

        # 裁剪旧音频（保留最近 30 秒）
        self._trim_old_audio()

        return result

    def _process_vad(self, audio_chunk: np.ndarray) -> Optional[SpeechSegment]:
        """将音频块切分为 512 样本窗口，送入 VADIterator 并追踪状态"""
        # 转为 torch tensor
        if not isinstance(audio_chunk, torch.Tensor):
            tensor = torch.from_numpy(audio_chunk.copy()).float()
        else:
            tensor = audio_chunk

        # 确保是 1D
        if tensor.dim() > 1:
            tensor = tensor.squeeze()

        total = len(tensor)
        result_segment: Optional[SpeechSegment] = None

        # 以 512 样本步长滑动
        for offset in range(0, total, WINDOW_SIZE):
            window = tensor[offset:offset + WINDOW_SIZE]
            if len(window) < WINDOW_SIZE:
                # 补齐到 512 样本
                pad_len = WINDOW_SIZE - len(window)
                window = torch.nn.functional.pad(window, (0, pad_len))

            vad_result = self._vad_iter(window)

            if vad_result is None:
                # 检查是否超过最大语音段长度（VADIterator 不内置此功能）
                if self._triggered and self._current_speech_start is not None:
                    current_pos = self._vad_iter.current_sample
                    speech_len = current_pos - self._current_speech_start
                    if speech_len >= self.max_speech_samples:
                        # 强制切断
                        segment = self._force_end_segment(current_pos)
                        if segment is not None:
                            result_segment = segment
                continue

            if 'start' in vad_result:
                self._current_speech_start = vad_result['start']
                self._triggered = True

            elif 'end' in vad_result and self._triggered:
                speech_end = vad_result['end']
                segment = self._create_segment(self._current_speech_start, speech_end)
                self._triggered = False
                self._current_speech_start = None
                if segment is not None:
                    result_segment = segment

        return result_segment

    def _force_end_segment(self, end_sample: int) -> Optional[SpeechSegment]:
        """强制结束当前语音段（超过最大长度时调用）"""
        if self._current_speech_start is None:
            return None
        segment = self._create_segment(self._current_speech_start, end_sample)
        self._vad_iter.reset_states()
        self._triggered = False
        self._current_speech_start = None
        return segment

    def _create_segment(self, start_sample: int, end_sample: int) -> Optional[SpeechSegment]:
        """从音频缓冲区提取语音段"""
        duration_samples = end_sample - start_sample
        if duration_samples < self.min_speech_samples:
            return None  # 太短，过滤

        # 从累积的音频块中提取
        audio_data = self._extract_audio(start_sample, end_sample)
        if len(audio_data) == 0:
            return None

        self._segment_id += 1
        duration_ms = duration_samples / self.sample_rate * 1000

        # 转为 PCM 16-bit 字节（方便后续存 WAV）
        audio_int16 = (audio_data * 32767).astype(np.int16)
        audio_bytes = audio_int16.tobytes()

        return SpeechSegment(
            id=self._segment_id,
            start_ms=start_sample / self.sample_rate * 1000,
            end_ms=end_sample / self.sample_rate * 1000,
            audio_data=audio_bytes,
            duration_ms=duration_ms,
        )

    def _extract_audio(self, start_sample: int, end_sample: int) -> np.ndarray:
        """从累积的音频块中提取指定范围的数据"""
        # 考虑已裁剪的偏移
        effective_start = start_sample - self._trimmed_offset
        effective_end = end_sample - self._trimmed_offset

        if effective_end <= 0:
            return np.array([], dtype=np.float32)

        # 拼接所有块
        all_audio = np.concatenate([chunk for _, chunk in self._audio_chunks])

        start_idx = max(0, effective_start)
        end_idx = min(len(all_audio), effective_end)

        if start_idx >= end_idx:
            return np.array([], dtype=np.float32)

        return all_audio[start_idx:end_idx]

    def _trim_old_audio(self):
        """裁剪 30 秒前的旧音频，控制内存占用"""
        max_buffer_samples = self.sample_rate * 30  # 30 秒
        if self._total_samples - self._trimmed_offset <= max_buffer_samples:
            return

        # 计算裁剪点
        trim_point = self._total_samples - max_buffer_samples
        # 更新偏移
        self._trimmed_offset = trim_point

        # 移除旧块
        kept_chunks = []
        for start, chunk in self._audio_chunks:
            chunk_end = start + len(chunk)
            if chunk_end <= trim_point:
                continue  # 整个块都是旧的
            elif start < trim_point:
                # 部分旧的 — 只保留新部分
                cut = trim_point - start
                kept_chunks.append((start, chunk[cut:]))
            else:
                kept_chunks.append((start, chunk))
        self._audio_chunks = kept_chunks

    def flush(self) -> Optional[SpeechSegment]:
        """
        强制结束当前语音段（流结束时调用）
        用于处理音频流末尾未完成的语音段
        """
        if not self._triggered or self._current_speech_start is None:
            return None

        # 使用当前累积的总样本数作为结束位置
        end_sample = self._total_samples + self._trimmed_offset
        segment = self._create_segment(self._current_speech_start, end_sample)
        self._triggered = False
        self._current_speech_start = None
        return segment

    def reset(self) -> None:
        """重置 VAD 状态和所有缓冲区"""
        self._vad_iter.reset_states()
        self._audio_chunks.clear()
        self._total_samples = 0
        self._trimmed_offset = 0
        self._current_speech_start = None
        self._triggered = False
        # 注意: segment_id 不重置，保持全局递增
