"""
高质量离线 TTS 语音合成引擎 — 基于 Piper TTS
所有语音模型存放于 D:/AI-Simultaneous-Interpretation-Assistant/models/piper_voices/
支持多音色切换、语速调节
"""
import queue
import threading
import time
import os
import logging
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
logger = logging.getLogger(__name__)
DEFAULT_VOICE_DIR = PROJECT_ROOT / "models" / "piper_voices"

# 中文音色列表 (名称 → piper 语音 key)
VOICE_OPTIONS = {
    "晓雅 (女声)": {
        "key": "zh_CN-xiao_ya-medium",
        "file": "zh_CN-xiao_ya-medium",
    },
    "花言 (女声)": {
        "key": "zh_CN-huayan-medium",
        "file": "zh_CN-huayan-medium",
    },
    "超文 (男声)": {
        "key": "zh_CN-chaowen-medium",
        "file": "zh_CN-chaowen-medium",
    },
}


class PiperTTSEngine:
    """
    基于 Piper 的离线 TTS 引擎（高质量神经语音合成）

    用法:
        tts = PiperTTSEngine(voice="晓雅 (女声)", speed=1.0)
        tts.start()
        tts.speak("你好世界")
        tts.set_voice("超文 (男声)")
        tts.stop()
    """

    def __init__(self, voice: str = "晓雅 (女声)", speed: float = 1.0):
        self._speed = speed
        self._voice_name = voice
        self._voice: Optional[object] = None  # PiperVoice 实例
        self._available = False
        self._voice_dir = DEFAULT_VOICE_DIR

        # 队列和线程（与 pyttsx3 版保持一致接口）
        self._queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._current_text: str = ""

        self._init_engine()

    def _init_engine(self):
        """初始化 Piper TTS 引擎"""
        try:
            from piper import PiperVoice
            import json

            voice_info = VOICE_OPTIONS.get(self._voice_name, VOICE_OPTIONS["晓雅 (女声)"])
            model_path = self._voice_dir / f"{voice_info['file']}.onnx"
            config_path = self._voice_dir / f"{voice_info['file']}.onnx.json"

            if not model_path.exists():
                print(f"[PiperTTS] 语音模型未找到: {model_path}")
                print(f"[PiperTTS] 请先运行下载脚本下载语音模型到 {self._voice_dir}")
                self._available = False
                return

            if not config_path.exists():
                print(f"[PiperTTS] 语音配置未找到: {config_path}")
                self._available = False
                return

            # 加载语音模型
            self._voice = PiperVoice.load(str(model_path), config_path=str(config_path))
            self._available = True
            print(f"[PiperTTS] 引擎就绪 (Piper, voice={self._voice_name}, speed={self._speed}x)")
        except Exception as e:
            print(f"[PiperTTS] 引擎不可用: {e} — 将使用纯字幕模式")
            import traceback
            traceback.print_exc()
            self._available = False
            self._voice = None

    # ─── 公开接口 ───

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def speed(self) -> float:
        return self._speed

    @property
    def voice_name(self) -> str:
        return self._voice_name

    @property
    def available_voices(self) -> list:
        """返回可用的音色列表（模型文件存在且加载成功）"""
        available = []
        for name, info in VOICE_OPTIONS.items():
            model_path = self._voice_dir / f"{info['file']}.onnx"
            if model_path.exists():
                available.append(name)
        return available if available else list(VOICE_OPTIONS.keys())

    def start(self):
        """启动 TTS 工作线程"""
        if not self._available:
            return
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="PiperTTS-Worker",
            daemon=True,
        )
        self._worker.start()

    def stop(self):
        """停止 TTS 工作线程并清空队列"""
        self._running = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def speak(self, text: str):
        """将文本加入朗读队列（非阻塞），新文本打断旧文本"""
        if not self._available or not text or not text.strip():
            return
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(text.strip())

    def set_speed(self, speed: float):
        """设置语速 (0.8 / 1.0 / 1.2 / 1.5)"""
        self._speed = speed

    def set_voice(self, voice_name: str):
        """切换音色（需要重新加载语音模型）"""
        if voice_name not in VOICE_OPTIONS:
            print(f"[PiperTTS] 未知音色: {voice_name}")
            return
        was_running = self._running
        if was_running:
            self.stop()
        self._voice_name = voice_name
        self._init_engine()
        if was_running and self._available:
            self.start()

    # ─── 内部 ───

    def _worker_loop(self):
        """工作线程：读取队列并合成语音"""
        import wave
        import struct

        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:
                break

            self._current_text = text
            logger.info(f"[PiperTTS] 开始合成: {text[:30]}...")
            wav_path = None
            try:
                # Piper 合成
                audio_chunks = []
                sample_rate = 22050
                for chunk in self._voice.synthesize(text):
                    audio_chunks.append(chunk.audio_float_array)
                    sample_rate = chunk.sample_rate

                if not audio_chunks:
                    logger.warning("[PiperTTS] 合成结果为空")
                    continue

                audio_data = np.concatenate(audio_chunks)
                logger.info(f"[PiperTTS] 合成完成: {len(audio_data)/sample_rate:.1f}秒, {sample_rate}Hz")

                # 语速调整
                if self._speed != 1.0:
                    target_len = max(1, int(len(audio_data) / self._speed))
                    indices = np.linspace(0, len(audio_data) - 1, target_len)
                    lo = np.floor(indices).astype(int)
                    hi = np.minimum(lo + 1, len(audio_data) - 1)
                    frac = indices - lo
                    audio_data = audio_data[lo] * (1 - frac) + audio_data[hi] * frac

                # 保存 WAV
                wav_dir = PROJECT_ROOT / "exports"
                wav_dir.mkdir(parents=True, exist_ok=True)
                wav_path = wav_dir / "_tts_tmp.wav"
                audio_int16 = (np.clip(audio_data, -1.0, 1.0) * 32767).astype(np.int16)

                with wave.open(str(wav_path), 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_int16.tobytes())

                # 播放：使用 winsound（Windows 原生，不冲突 WASAPI）
                import winsound
                winsound.PlaySound(str(wav_path), winsound.SND_FILENAME)
                logger.info("[PiperTTS] 播放完成")

            except Exception as e:
                logger.error(f"[PiperTTS] 合成失败: {e}")
                import traceback
                traceback.print_exc()
            finally:
                self._current_text = ""
                if wav_path and wav_path.exists():
                    try:
                        wav_path.unlink()
                    except Exception:
                        pass
