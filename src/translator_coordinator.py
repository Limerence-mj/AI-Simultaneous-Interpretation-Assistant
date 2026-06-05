"""
翻译协调器 — TranslationCoordinator
串联 ASR → MT 管道，管理上下文窗口，更新状态管理器
v0.1: 简易版，不含修正逻辑
"""
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

from src.state_manager import StateManager, TranslationRecord, AppStatus
from src.mt_engine import MTEngine, CONTEXT_SIZE
from src.asr_engine import ASRResult


@dataclass
class MTResult:
    """翻译结果"""
    segment_id: int
    zh_text: str
    en_text: str
    is_corrected: bool = False
    prev_translation: Optional[str] = None


class TranslationCoordinator:
    """
    翻译协调器 — 管理从 ASR 结果到最终译文的全流程

    用法:
        coordinator = TranslationCoordinator()
        coordinator.initialize()  # 加载 MT 模型

        # 每次 ASR 完成后调用:
        result = coordinator.process(asr_result)

        # result.zh_text 即为最终中文译文
    """

    def __init__(self):
        self.mt = MTEngine()
        self.state = StateManager()

        # 上下文窗口：存储最近 N 句英文原文
        self._context_window: deque = deque(maxlen=CONTEXT_SIZE + 1)

        # 上一句译文（预留，用于修正比较）
        self._last_translation: Optional[str] = None

        # 统计
        self._total_process_time: float = 0.0
        self._sentence_count: int = 0

    def initialize(self) -> float:
        """加载 MT 模型，返回耗时"""
        self.state.set_status(AppStatus.LOADING)
        try:
            t = self.mt.load_model()
            self.state.set_status(AppStatus.STOPPED)
            return t
        except Exception as e:
            self.state.set_status(AppStatus.ERROR)
            raise RuntimeError(f"翻译模型加载失败: {e}") from e

    def process(self, asr_result: ASRResult) -> MTResult:
        """
        处理单个 ASR 结果，返回翻译结果

        Args:
            asr_result: ASR 识别结果

        Returns:
            MTResult 包含中文译文
        """
        t0 = time.time()

        en_text = asr_result.en_text

        # 跳过空文本
        if not en_text or en_text.startswith("[ASR_ERROR"):
            return MTResult(
                segment_id=asr_result.segment_id,
                zh_text="",
                en_text=en_text,
            )

        # v0.1: 单句独立翻译，不拼接上下文
        # 上下文窗口用于步骤五的修正重译 (Context-aware Re-translation)
        zh_text = self.mt.translate(en_text)

        # 更新上下文窗口
        self._context_window.append(en_text)

        # 创建翻译记录
        record = TranslationRecord(
            id=asr_result.segment_id,
            start_time=asr_result.start_ms / 1000.0,
            end_time=asr_result.end_ms / 1000.0,
            en_text=en_text,
            first_translation=zh_text,
            final_translation=zh_text,
            is_corrected=False,
            correction_count=0,
        )
        self.state.add_record(record)

        # 更新当前字幕
        self.state.set_subtitle(zh_text)

        # 记录统计
        latency_ms = (time.time() - t0) * 1000
        self.state.record_latency(latency_ms)
        self._total_process_time += latency_ms
        self._sentence_count += 1

        self._last_translation = zh_text

        return MTResult(
            segment_id=asr_result.segment_id,
            zh_text=zh_text,
            en_text=en_text,
        )

    def reset(self):
        """重置协调器状态"""
        self._context_window.clear()
        self._last_translation = None
        self._total_process_time = 0.0
        self._sentence_count = 0

    @property
    def avg_translation_time_ms(self) -> float:
        if self._sentence_count == 0:
            return 0.0
        return self._total_process_time / self._sentence_count
