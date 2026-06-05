"""
音频捕获模块 — AudioCapture
支持系统音频 Loopback (WASAPI) 和麦克风输入
所有产出物仅存放于项目目录 D:/AI-Simultaneous-Interpretation-Assistant/
"""
import threading
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd


@dataclass
class AudioDevice:
    """音频设备信息"""
    id: int
    name: str
    device_type: str          # "input" | "loopback"
    channels: int
    sample_rate: int
    hostapi_name: str = ""


# 全局采样率常量
SAMPLE_RATE = 16000
BLOCK_SIZE = 1024
CHANNELS = 1
DTYPE = np.float32


class AudioCapture:
    """
    音频捕获器
    - Windows: WASAPI loopback 捕获系统音频; 或麦克风输入
    - macOS: 需 BlackHole 虚拟设备 (文档说明)
    """

    def __init__(self):
        self._stream: Optional[sd.InputStream] = None
        self._running = False
        self._lock = threading.Lock()

    @staticmethod
    def list_devices() -> List[AudioDevice]:
        """枚举可用音频设备，区分 input 和 loopback 类型"""
        devices: List[AudioDevice] = []
        sd_devices = sd.query_devices()
        hostapis = sd.query_hostapis()

        for i, dev in enumerate(sd_devices):
            hostapi_name = hostapis[dev['hostapi']]['name'] if dev['hostapi'] < len(hostapis) else ""

            # 输出设备 (WASAPI) → 可用于 Loopback 捕获系统声音
            if dev['max_output_channels'] > 0 and 'WASAPI' in hostapi_name:
                devices.append(AudioDevice(
                    id=i,
                    name=dev['name'],
                    device_type="loopback",
                    channels=dev['max_output_channels'],
                    sample_rate=int(dev['default_samplerate']),
                    hostapi_name=hostapi_name,
                ))

            # 输入设备 (麦克风等)
            if dev['max_input_channels'] > 0:
                # 避免重复添加 (有些设备同时是 input+output)
                if not any(d.id == i for d in devices):
                    devices.append(AudioDevice(
                        id=i,
                        name=dev['name'],
                        device_type="input",
                        channels=dev['max_input_channels'],
                        sample_rate=int(dev['default_samplerate']),
                        hostapi_name=hostapi_name,
                    ))

        return devices

    def start(self, device_id: Optional[int], callback: Callable[[np.ndarray], None]) -> None:
        """
        开始音频捕获

        Args:
            device_id: 设备 ID (None = 系统默认输入)
            callback: 回调函数，接收 (audio_chunk: np.ndarray) — 1D float32, 16kHz 单声道
        """
        with self._lock:
            if self._running:
                return

            def _sd_callback(indata, frames, time_info, status):
                """sounddevice 回调：转为 float32 单声道后传递给上层"""
                if status:
                    print(f"[AudioCapture] 警告: {status}")
                # 转单声道 + float32
                audio = indata.astype(DTYPE)
                if audio.ndim > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)  # 立体声→单声道
                audio = audio.flatten()
                callback(audio)

            try:
                self._stream = sd.InputStream(
                    device=device_id,
                    channels=CHANNELS,
                    samplerate=SAMPLE_RATE,
                    blocksize=BLOCK_SIZE,
                    dtype=DTYPE,
                    callback=_sd_callback,
                )
                self._stream.start()
                self._running = True
            except sd.PortAudioError as e:
                raise RuntimeError(f"音频设备打开失败: {e}") from e

    def stop(self) -> None:
        """停止音频捕获"""
        with self._lock:
            if self._stream is not None:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._running = False

    @property
    def is_running(self) -> bool:
        return self._running
