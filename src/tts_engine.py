"""
语音合成引擎 — TTSEngine
使用 pyttsx3 (Windows SAPI5) 实现完全离线中文语音播报
支持队列管理、打断策略、语速调节、异常降级
"""
import queue
import threading
import time
from typing import Optional


class TTSEngine:
    """
    离线 TTS 语音播报引擎

    用法:
        tts = TTSEngine()
        tts.start()
        tts.speak("你好世界")        # 朗读（打断当前朗读）
        tts.set_speed(1.5)           # 1.5x 语速
        tts.stop()                   # 停止
    """

    # 语速档位 (pyttsx3 rate 默认 ~200 wpm)
    SPEED_PRESETS = {
        0.8: 140,
        1.0: 190,
        1.2: 230,
        1.5: 280,
    }

    def __init__(self, speed: float = 1.0):
        self._speed = speed
        self._engine = None
        self._available = False

        # 队列和线程
        self._queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._current_text: str = ""

        # 尝试初始化
        self._init_engine()

    def _init_engine(self):
        """初始化 pyttsx3 引擎"""
        try:
            import pyttsx3
            self._engine = pyttsx3.init()
            self._engine.setProperty('rate', self._get_rate())
            self._available = True

            # 尝试设置中文语音
            voices = self._engine.getProperty('voices')
            for voice in voices:
                if 'chinese' in voice.name.lower() or 'zh' in voice.id.lower():
                    self._engine.setProperty('voice', voice.id)
                    break
            print(f"[TTS] 引擎就绪 (pyttsx3, speed={self._speed}x)")
        except Exception as e:
            print(f"[TTS] 引擎不可用: {e} — 将使用纯字幕模式")
            self._available = False
            self._engine = None

    def _get_rate(self) -> int:
        """根据 speed 参数返回 pyttsx3 rate 值"""
        if self._speed in self.SPEED_PRESETS:
            return self.SPEED_PRESETS[self._speed]
        # 线性插值
        return int(self._speed * 190)

    # ─── 公开接口 ───

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def speed(self) -> float:
        return self._speed

    def start(self):
        """启动 TTS 工作线程"""
        if not self._available:
            return
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="TTS-Worker",
            daemon=True,
        )
        self._worker.start()

    def stop(self):
        """停止 TTS 工作线程并清空队列"""
        self._running = False
        # 清空队列
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        # 停止当前朗读
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass

    def speak(self, text: str):
        """
        将文本加入朗读队列（非阻塞）
        根据打断策略，新文本会清空队列中等待的旧文本

        Args:
            text: 中文文本（空文本会被忽略）
        """
        if not self._available or not text or not text.strip():
            return

        # 打断策略：清空队列中等待的旧内容，只保留最新一句
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        self._queue.put(text.strip())

    def set_speed(self, speed: float):
        """设置语速 (0.8 / 1.0 / 1.2 / 1.5)"""
        self._speed = speed
        if self._engine:
            try:
                self._engine.setProperty('rate', self._get_rate())
            except Exception:
                pass

    # ─── 内部 ───

    def _worker_loop(self):
        """工作线程：不断读取队列并朗读"""
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:  # 哨兵
                break

            self._current_text = text
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                # 单次朗读失败不影响后续
                print(f"[TTS] 朗读失败: {e}")
            finally:
                self._current_text = ""
