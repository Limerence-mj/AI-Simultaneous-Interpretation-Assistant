"""
状态管理器 — StateManager (单例)
跨模块共享状态：当前译文、历史记录、运行状态、配置
线程安全：内部使用 threading.Lock 保护所有写操作
"""
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Callable


class AppStatus(Enum):
    STOPPED = "stopped"
    LOADING = "loading"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class TranslationRecord:
    """翻译历史记录"""
    id: int
    start_time: float          # 秒
    end_time: float            # 秒
    en_text: str
    first_translation: str     # 首次译文
    final_translation: str     # 最终译文（修正后）
    is_corrected: bool = False
    correction_count: int = 0


@dataclass
class AppConfig:
    """应用配置"""
    font_size: int = 24
    opacity: float = 0.6
    audio_device_id: Optional[int] = None
    audio_source: str = "system"   # "system" | "microphone"
    source_language: str = "en"
    target_language: str = "zh"
    auto_correct: bool = True
    tts_enabled: bool = False
    tts_speed: float = 1.0
    whisper_model: str = "small"


class StateManager:
    """
    单例状态管理器，跨所有模块共享

    用法:
        state = StateManager()
        state.set_subtitle("你好世界")
        state.add_record(record)
        print(state.current_subtitle)
    """

    _instance: Optional["StateManager"] = None
    _lock = threading.Lock()

    def __new__(cls) -> "StateManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self._data_lock = threading.Lock()

        # 当前状态
        self._status: AppStatus = AppStatus.STOPPED
        self._current_subtitle: str = ""
        self._current_en_text: str = ""
        self._subtitle_version: int = 0  # 递增版本号，UI 据此判断是否需要刷新

        # 历史记录
        self._history: List[TranslationRecord] = []

        # 统计
        self._translated_count: int = 0
        self._correction_count: int = 0
        self._last_latencies: List[float] = []  # 最近 5 句延迟 (ms)

        # 配置
        self._config: AppConfig = AppConfig()

        # 回调 (非线程安全，仅在主线程设置)
        self._on_subtitle_changed: Optional[Callable[[str], None]] = None
        self._on_status_changed: Optional[Callable[[AppStatus], None]] = None

    # ─── 字幕 ───

    @property
    def current_subtitle(self) -> str:
        with self._data_lock:
            return self._current_subtitle

    @property
    def current_en_text(self) -> str:
        with self._data_lock:
            return self._current_en_text

    @property
    def subtitle_version(self) -> int:
        return self._subtitle_version

    def set_subtitle(self, zh_text: str, en_text: str = ""):
        """更新当前字幕（中英双语）"""
        with self._data_lock:
            self._current_subtitle = zh_text
            self._current_en_text = en_text
            self._subtitle_version += 1

    # ─── 状态 ───

    @property
    def status(self) -> AppStatus:
        with self._data_lock:
            return self._status

    def set_status(self, status: AppStatus):
        with self._data_lock:
            self._status = status
        if self._on_status_changed:
            self._on_status_changed(status)

    # ─── 历史 ───

    @property
    def history(self) -> List[TranslationRecord]:
        with self._data_lock:
            return list(self._history)

    def add_record(self, record: TranslationRecord):
        with self._data_lock:
            self._history.append(record)
            self._translated_count += 1
            if record.is_corrected:
                self._correction_count += 1

    # ─── 统计 ───

    @property
    def translated_count(self) -> int:
        return self._translated_count

    @property
    def correction_count(self) -> int:
        return self._correction_count

    def record_latency(self, latency_ms: float):
        """记录延迟 (保留最近 5 句)"""
        with self._data_lock:
            self._last_latencies.append(latency_ms)
            if len(self._last_latencies) > 5:
                self._last_latencies.pop(0)

    @property
    def avg_latency_ms(self) -> float:
        with self._data_lock:
            if not self._last_latencies:
                return 0.0
            return sum(self._last_latencies) / len(self._last_latencies)

    # ─── 配置 ───

    @property
    def config(self) -> AppConfig:
        with self._data_lock:
            return self._config

    def update_config(self, **kwargs):
        with self._data_lock:
            for key, value in kwargs.items():
                if hasattr(self._config, key):
                    setattr(self._config, key, value)

    # ─── 回调 ───

    def set_subtitle_callback(self, callback: Optional[Callable[[str], None]]):
        """设置字幕变化回调 (主线程)"""
        self._on_subtitle_changed = callback

    def set_status_callback(self, callback: Optional[Callable[[AppStatus], None]]):
        """设置状态变化回调 (主线程)"""
        self._on_status_changed = callback
