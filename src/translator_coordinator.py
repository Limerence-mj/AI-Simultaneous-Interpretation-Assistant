"""
翻译协调器 — TranslationCoordinator v1.0
串联 ASR → MT 管道，管理上下文窗口，更新状态管理器
v1.0: 完整修正逻辑 — 断句合并(Reshoot) + 上下文回顾修正(Context-aware Re-translation)
"""
import time
from collections import deque
from dataclasses import dataclass
from typing import List, Optional

from src.state_manager import StateManager, TranslationRecord, AppStatus
from src.mt_engine import MTEngine, CONTEXT_SIZE
from src.asr_engine import ASRResult


# ─── 修正参数 ───
RESHOOT_GAP_MS = 300          # 断句间隔 < 300ms 触发合并
EDIT_DISTANCE_THRESHOLD = 2   # 编辑距离 > 2 触发上下文修正


@dataclass
class MTResult:
    """翻译结果"""
    segment_id: int
    zh_text: str
    en_text: str
    is_corrected: bool = False
    correction_type: str = ""          # "reshoot" | "context" | ""
    prev_translation: Optional[str] = None


class TranslationCoordinator:
    """
    翻译协调器 v1.0 — 含完整修正逻辑

    修正类型:
    1. Reshoot (断句合并): VAD 误将一句话切成两段时，合并后重译
    2. Context-aware (上下文修正): 积累上下文后重新翻译，消解代词/歧义

    用法:
        coordinator = TranslationCoordinator()
        coordinator.initialize()
        result = coordinator.process(asr_result)
    """

    def __init__(self):
        self.mt = MTEngine()
        self.state = StateManager()

        # 上下文窗口：最近 N+1 句 (英文, 中文首次译文)
        self._context_window: deque = deque(maxlen=CONTEXT_SIZE + 1)

        # TTS 引擎 (可选)
        self._tts = None

        # Reshoot 追踪
        self._last_asr_result: Optional[ASRResult] = None
        self._last_first_translation: Optional[str] = None

        # 统计
        self._total_process_time: float = 0.0
        self._sentence_count: int = 0
        self._reshoot_count: int = 0
        self._context_correction_count: int = 0

    def set_tts(self, tts_engine):
        """设置 TTS 引擎（可选）"""
        self._tts = tts_engine

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

    # ─── 主处理流程 ───

    def process(self, asr_result: ASRResult) -> MTResult:
        """
        处理单个 ASR 结果，返回最终翻译结果

        流程: Reshoot检测 → 首次翻译 → 字幕 → TTS播报
        注：上下文修正暂时禁用，优先保证翻译准确性和连续性
        """
        t0 = time.time()

        en_text = asr_result.en_text

        # 跳过空/错误文本
        if not en_text or en_text.startswith("[ASR_ERROR"):
            self._last_asr_result = asr_result
            return MTResult(
                segment_id=asr_result.segment_id,
                zh_text="", en_text=en_text,
            )

        correction_type = ""
        prev_translation: Optional[str] = None

        # ─── 阶段1: Reshoot 断句合并检测 ───
        merged_en, did_reshoot = self._check_reshoot(asr_result)
        if did_reshoot:
            en_text = merged_en
            correction_type = "reshoot"
            self._reshoot_count += 1

        # ─── 阶段2: 直接翻译 ───
        zh_text = self.mt.translate(en_text)

        # ─── 阶段3: 更新上下文窗口 ───
        if not did_reshoot:
            self._context_window.append((en_text, zh_text))
        else:
            if len(self._context_window) > 0:
                prev_en, prev_zh = self._context_window[-1]
                self._context_window[-1] = (en_text, prev_zh)
            else:
                self._context_window.append((en_text, zh_text))

        # ─── 阶段4: 创建/更新翻译记录 ───
        is_corrected = correction_type != ""

        if did_reshoot:
            self._update_last_record(
                en_text=en_text,
                final_translation=zh_text,
                correction_count=1,
            )
        else:
            record = TranslationRecord(
                id=asr_result.segment_id,
                start_time=asr_result.start_ms / 1000.0,
                end_time=asr_result.end_ms / 1000.0,
                en_text=en_text,
                first_translation=zh_text,
                final_translation=zh_text,
                is_corrected=is_corrected,
                correction_count=1 if is_corrected else 0,
            )
            self.state.add_record(record)

        # 更新当前字幕
        self.state.set_subtitle(zh_text)

        # TTS 语音播报
        if self._tts and self._tts.is_available and zh_text:
            self._tts.speak(zh_text)

        # 统计
        latency_ms = (time.time() - t0) * 1000
        self.state.record_latency(latency_ms)
        self._total_process_time += latency_ms
        self._sentence_count += 1

        # 保存状态用于下次 Reshoot 检测
        self._last_asr_result = asr_result
        self._last_first_translation = zh_text

        return MTResult(
            segment_id=asr_result.segment_id,
            zh_text=zh_text,
            en_text=en_text,
            is_corrected=is_corrected,
            correction_type=correction_type,
            prev_translation=prev_translation,
        )

    # ─── Reshoot 断句合并 ───

    def _check_reshoot(self, current: ASRResult) -> tuple:
        """
        检测是否需要断句合并
        返回 (merged_en_text, did_reshoot)
        """
        if self._last_asr_result is None:
            return current.en_text, False

        last = self._last_asr_result

        # 前一句不以句号结束
        last_text = last.en_text.strip()
        if not last_text:
            return current.en_text, False
        ends_with_punct = last_text[-1] in '.!?'

        # 时间间隔 < 阈值
        gap_ms = current.start_ms - last.end_ms
        is_close = gap_ms < RESHOOT_GAP_MS

        if is_close and not ends_with_punct:
            # 合并两句
            merged = last_text + " " + current.en_text
            return merged, True

        return current.en_text, False

    # ─── 上下文翻译提取 ───

    def _extract_current_translation(self, raw_output: str, en_context: List[str]) -> str:
        """
        从 opus-mt 上下文拼接输出中提取仅当前句的译文

        opus-mt 的设计限制：会翻译整个拼接输入，无法原生只输出当前句。
        此处采用启发式剥离策略：逐句去除上文独立翻译。
        若剥离后为空，返回空 → 回退到首次翻译。
        """
        if not raw_output or not en_context:
            return raw_output

        result = raw_output

        for en in en_context:
            ctx_zh = self.mt.translate(en)
            if not ctx_zh or len(ctx_zh) < 2:
                continue

            if ctx_zh in result:
                result = result.replace(ctx_zh, "", 1).strip()
                continue

            # 模糊匹配：上文翻译可能因上下文而措辞微调
            # 找 "。" 分割点
            common_prefix_len = self._common_prefix_len(ctx_zh, result)
            if common_prefix_len >= min(len(ctx_zh), 6):
                # 上文翻译在开头，找句号边界切除
                cut = result.find("。", common_prefix_len - 2)
                if cut > 0 and cut < len(result) - 2:
                    result = result[cut + 1:].strip()

        return result if len(result) > 1 else ""

    @staticmethod
    def _common_prefix_len(a: str, b: str) -> int:
        """计算两个字符串的公共前缀长度"""
        n = 0
        for ca, cb in zip(a, b):
            if ca != cb:
                break
            n += 1
        return n

    # ─── 历史记录更新 ───

    def _update_last_record(self, **kwargs):
        """更新 StateManager 中最后一条翻译记录"""
        history = self.state.history
        if not history:
            return
        last = history[-1]
        for key, value in kwargs.items():
            if hasattr(last, key):
                setattr(last, key, value)
        last.is_corrected = True
        if 'correction_count' in kwargs:
            last.correction_count = kwargs['correction_count']

    # ─── 重置与统计 ───

    def reset(self):
        """重置协调器状态"""
        self._context_window.clear()
        self._last_asr_result = None
        self._last_first_translation = None
        self._total_process_time = 0.0
        self._sentence_count = 0
        self._reshoot_count = 0
        self._context_correction_count = 0
        # 重置 StateManager 历史（避免跨测试污染）
        self.state._history.clear()
        self.state._translated_count = 0
        self.state._correction_count = 0
        self.state._last_latencies.clear()

    @property
    def avg_translation_time_ms(self) -> float:
        if self._sentence_count == 0:
            return 0.0
        return self._total_process_time / self._sentence_count

    @property
    def correction_stats(self) -> dict:
        return {
            "reshoot_count": self._reshoot_count,
            "context_correction_count": self._context_correction_count,
            "total_sentences": self._sentence_count,
        }


# ─── 编辑距离 (Levenshtein) ───

def _edit_distance(a: str, b: str) -> int:
    """计算两个字符串的编辑距离 (Levenshtein)"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)

    # 使用较短字符串作为列，减少内存
    if len(a) < len(b):
        a, b = b, a

    prev = list(range(len(b) + 1))
    curr = [0] * (len(b) + 1)

    for i, ca in enumerate(a, 1):
        curr[0] = i
        for j, cb in enumerate(b, 1):
            if ca == cb:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev, curr = curr, prev

    return prev[-1]
