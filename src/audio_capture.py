"""
音频捕获模块 — AudioCapture
- Windows WASAPI Loopback: 纯 ctypes 实现，捕获系统音频（绕过 sounddevice 兼容问题）
- 麦克风输入: 通过 sounddevice 使用 MME/DirectSound 设备
- 所有产出物仅存放于项目目录 D:/AI-Simultaneous-Interpretation-Assistant/
"""
import ctypes
from ctypes import byref, c_void_p, POINTER, c_uint32, c_uint64, sizeof, WINFUNCTYPE
from ctypes.wintypes import DWORD, WORD, BYTE
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import numpy as np
import sounddevice as sd


# ═══════════════════════════════════════════════════════════
# 全局常量
# ═══════════════════════════════════════════════════════════

SAMPLE_RATE = 16000          # 管道统一采样率
BLOCK_SIZE = 1024            # 每次回调的帧数
CHANNELS = 1                 # 管道统一声道数
LOOPBACK_SAMPLE_RATE = 48000 # WASAPI 混音格式采样率


# ═══════════════════════════════════════════════════════════
# Windows WASAPI 类型定义
# ═══════════════════════════════════════════════════════════

class _GUID(ctypes.Structure):
    _fields_ = [
        ('Data1', c_uint32),
        ('Data2', ctypes.c_ushort),
        ('Data3', ctypes.c_ushort),
        ('Data4', BYTE * 8),
    ]

    @classmethod
    def from_str(cls, s):
        s = s.strip('{}')
        parts = s.split('-')
        g = cls()
        g.Data1 = int(parts[0], 16)
        g.Data2 = int(parts[1], 16)
        g.Data3 = int(parts[2], 16)
        hex_bytes = parts[3] + parts[4]
        for i in range(8):
            g.Data4[i] = int(hex_bytes[i*2:i*2+2], 16)
        return g


class _WAVEFORMATEX(ctypes.Structure):
    _fields_ = [
        ('wFormatTag', WORD),
        ('nChannels', WORD),
        ('nSamplesPerSec', DWORD),
        ('nAvgBytesPerSec', DWORD),
        ('nBlockAlign', WORD),
        ('wBitsPerSample', WORD),
        ('cbSize', WORD),
    ]


class _WAVEFORMATEXTENSIBLE(ctypes.Structure):
    _fields_ = [
        ('Format', _WAVEFORMATEX),
        ('wValidBitsPerSample', WORD),
        ('dwChannelMask', DWORD),
        ('SubFormat', _GUID),
    ]


# HRESULT 常量
_RPC_E_CHANGED_MODE = -2147417850  # 0x80010106 的有符号表示

# WASAPI GUID 常量
_CLSCTX_ALL = 0x17
_AUDCLNT_SHAREMODE_SHARED = 0
_AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
_IID_IAudioClient = _GUID.from_str('{1CB9AD4C-DBFA-4C32-B178-C2F568A703B2}')
_IID_IAudioCaptureClient = _GUID.from_str('{C8ADBD64-E71E-48A0-A4DE-185C395CD317}')
_IID_IMMDeviceEnumerator = _GUID.from_str('{A95664D2-9614-4F35-A746-DE8DB63617E6}')
_CLSID_MMDeviceEnumerator = _GUID.from_str('{BCDE0395-E52F-467C-8E3D-C4579291692E}')


# ═══════════════════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════════════════

def _vtbl_call(pInterface, index, proto):
    """获取 COM 虚函数表中指定索引的函数"""
    vtbl = ctypes.cast(pInterface, POINTER(c_void_p))
    entries = ctypes.cast(vtbl[0], POINTER(c_void_p))
    return proto(ctypes.cast(entries[index], c_void_p).value)


# ═══════════════════════════════════════════════════════════
# WASAPI Loopback 音频捕获器
# ═══════════════════════════════════════════════════════════

class WASAPILoopbackCapture:
    """
    使用 Windows WASAPI Core Audio API（纯 ctypes）捕获系统音频输出。
    在后台线程运行，将 48000Hz 立体声重采样为 16000Hz 单声道后回调。

    用法:
        cap = WASAPILoopbackCapture()
        cap.start(callback)
        cap.stop()
    """

    def __init__(self):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._callback: Optional[Callable[[np.ndarray], None]] = None

    @staticmethod
    def is_available() -> bool:
        """测试 WASAPI loopback 是否可用"""
        ole32 = ctypes.windll.ole32
        hr = ole32.CoInitializeEx(None, 0)
        # S_OK=0, S_FALSE=1 (已初始化同模式), RPC_E_CHANGED_MODE=0x80010106 (已初始化但不同模式)
        # 这些情况 COM 都可用，继续执行
        com_initialized_here = (hr == 0)
        if hr < 0 and hr != _RPC_E_CHANGED_MODE:
            return False
        try:
            pEnum = c_void_p()
            hr = ole32.CoCreateInstance(
                byref(_CLSID_MMDeviceEnumerator), None, _CLSCTX_ALL,
                byref(_IID_IMMDeviceEnumerator), byref(pEnum),
            )
            if hr < 0 or not pEnum.value:
                return False

            # GetDefaultAudioEndpoint
            proto = WINFUNCTYPE(ctypes.c_long, c_void_p, ctypes.c_int, ctypes.c_int, POINTER(c_void_p))
            get_def = _vtbl_call(pEnum, 4, proto)
            pDev = c_void_p()
            hr = get_def(pEnum.value, 0, 0, byref(pDev))  # eRender=0, eConsole=0
            if hr < 0 or not pDev.value:
                return False

            # Activate IAudioClient
            proto = WINFUNCTYPE(ctypes.c_long, c_void_p, POINTER(_GUID), DWORD, c_void_p, POINTER(c_void_p))
            activate = _vtbl_call(pDev, 3, proto)
            pClient = c_void_p()
            hr = activate(pDev.value, byref(_IID_IAudioClient), _CLSCTX_ALL, None, byref(pClient))
            if hr < 0 or not pClient.value:
                return False

            # Cleanup
            for ptr in [pClient, pDev, pEnum]:
                vtbl = ctypes.cast(ptr, POINTER(c_void_p))
                entries = ctypes.cast(vtbl[0], POINTER(c_void_p))
                release = WINFUNCTYPE(DWORD, c_void_p)(ctypes.cast(entries[2], c_void_p).value)
                release(ptr.value)

            return True
        except Exception:
            return False
        finally:
            if com_initialized_here:
                ole32.CoUninitialize()

    def start(self, callback: Callable[[np.ndarray], None]) -> bool:
        """
        启动 WASAPI loopback 捕获。成功返回 True，失败返回 False。

        Args:
            callback: 接收 1D float32 16kHz 单声道 numpy 数组
        """
        if self._running:
            return True
        self._callback = callback
        self._running = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            name="WASAPI-Loopback",
            daemon=True,
        )
        self._thread.start()
        return True

    def stop(self):
        """停止捕获"""
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def is_running(self) -> bool:
        return self._running

    # ─── 内部: 音频捕获循环 ───

    def _capture_loop(self):
        """在后台线程中运行的 WASAPI loopback 捕获循环"""
        ole32 = ctypes.windll.ole32
        hr = ole32.CoInitializeEx(None, 0)  # COINIT_APARTMENTTHREADED
        com_initialized = (hr == 0)
        if hr < 0 and hr != _RPC_E_CHANGED_MODE:  # RPC_E_CHANGED_MODE 也继续
            self._running = False
            return

        pEnum = None
        pDev = None
        pClient = None
        pCapture = None

        try:
            # 1. 创建 MMDeviceEnumerator
            pEnum = c_void_p()
            hr = ole32.CoCreateInstance(
                byref(_CLSID_MMDeviceEnumerator), None, _CLSCTX_ALL,
                byref(_IID_IMMDeviceEnumerator), byref(pEnum),
            )
            if hr < 0 or not pEnum.value:
                return

            # 2. GetDefaultAudioEndpoint(eRender, eConsole)
            proto = WINFUNCTYPE(ctypes.c_long, c_void_p, ctypes.c_int, ctypes.c_int, POINTER(c_void_p))
            get_def = _vtbl_call(pEnum, 4, proto)
            pDev = c_void_p()
            hr = get_def(pEnum.value, 0, 0, byref(pDev))
            if hr < 0 or not pDev.value:
                return

            # 3. Activate IAudioClient
            proto = WINFUNCTYPE(ctypes.c_long, c_void_p, POINTER(_GUID), DWORD, c_void_p, POINTER(c_void_p))
            activate = _vtbl_call(pDev, 3, proto)
            pClient = c_void_p()
            hr = activate(pDev.value, byref(_IID_IAudioClient), _CLSCTX_ALL, None, byref(pClient))
            if hr < 0 or not pClient.value:
                return

            # 4. GetMixFormat (vtable 索引 8)
            proto = WINFUNCTYPE(ctypes.c_long, c_void_p, POINTER(c_void_p))
            get_mix = _vtbl_call(pClient, 8, proto)
            pWF = c_void_p()
            hr = get_mix(pClient.value, byref(pWF))
            if hr < 0 or not pWF.value:
                return
            wf = ctypes.cast(pWF.value, POINTER(_WAVEFORMATEX)).contents
            src_channels = wf.nChannels
            src_rate = wf.nSamplesPerSec

            # 5. Initialize (vtable 索引 3)
            proto = WINFUNCTYPE(ctypes.c_long, c_void_p, ctypes.c_int, DWORD,
                              ctypes.c_longlong, ctypes.c_longlong,
                              c_void_p, POINTER(_GUID))
            initialize = _vtbl_call(pClient, 3, proto)
            hns_period = ctypes.c_longlong(int(0.1 * 10000000))  # 100ms 低延迟
            hr = initialize(pClient.value, _AUDCLNT_SHAREMODE_SHARED,
                          _AUDCLNT_STREAMFLAGS_LOOPBACK, 0, hns_period,
                          pWF.value, None)
            if hr < 0:
                return

            # 6. GetService IAudioCaptureClient (vtable 索引 14)
            proto = WINFUNCTYPE(ctypes.c_long, c_void_p, POINTER(_GUID), POINTER(c_void_p))
            get_svc = _vtbl_call(pClient, 14, proto)
            pCapture = c_void_p()
            hr = get_svc(pClient.value, byref(_IID_IAudioCaptureClient), byref(pCapture))
            if hr < 0 or not pCapture.value:
                return

            # 7. Start (vtable 索引 10)
            proto = WINFUNCTYPE(ctypes.c_long, c_void_p)
            start_func = _vtbl_call(pClient, 10, proto)
            hr = start_func(pClient.value)
            if hr < 0:
                return

            # 8. 循环读取
            # GetBuffer (vtable 索引 3)
            proto = WINFUNCTYPE(ctypes.c_long, c_void_p, POINTER(c_void_p),
                              POINTER(c_uint32), POINTER(DWORD), c_uint64, c_uint64)
            get_buf = _vtbl_call(pCapture, 3, proto)

            # ReleaseBuffer (vtable 索引 4)
            proto = WINFUNCTYPE(ctypes.c_long, c_void_p, c_uint32)
            rel_buf = _vtbl_call(pCapture, 4, proto)

            resample_ratio = SAMPLE_RATE / src_rate
            # 累积缓冲区：按比例预估每次读到的帧数对应的输出帧数
            accum = np.array([], dtype=np.float32)

            while self._running:
                pData = c_void_p()
                numFrames = c_uint32()
                dwFlags = DWORD()
                hr = get_buf(pCapture.value, byref(pData), byref(numFrames),
                           byref(dwFlags), 0, 0)

                if hr >= 0 and numFrames.value > 0:
                    # 读取原始音频: 32-bit float, stereo (或更多声道)
                    n = numFrames.value * src_channels
                    raw = np.ctypeslib.as_array(
                        ctypes.cast(pData.value, POINTER(ctypes.c_float * n)).contents,
                        shape=(n,)
                    ).astype(np.float32).reshape(-1, src_channels)

                    # 多声道 → 单声道 (取平均)
                    mono = np.mean(raw, axis=1)

                    # 重采样到 16000Hz (线性插值)
                    if src_rate != SAMPLE_RATE:
                        in_len = len(mono)
                        out_len = max(1, int(in_len * resample_ratio))
                        indices = np.linspace(0, in_len - 1, out_len)
                        lo = np.floor(indices).astype(int)
                        hi = np.minimum(lo + 1, in_len - 1)
                        frac = indices - lo
                        mono = mono[lo] * (1 - frac) + mono[hi] * frac

                    accum = np.concatenate([accum, mono])

                    rel_buf(pCapture.value, numFrames.value)
                else:
                    # 无数据时短暂休眠
                    time.sleep(0.005)

                # 当累积数据达到 BLOCK_SIZE 时回调
                while len(accum) >= BLOCK_SIZE and self._running:
                    chunk = accum[:BLOCK_SIZE]
                    accum = accum[BLOCK_SIZE:]
                    try:
                        self._callback(chunk.astype(np.float32))
                    except Exception:
                        pass

            # 9. Stop (vtable 索引 11)
            proto = WINFUNCTYPE(ctypes.c_long, c_void_p)
            stop_func = _vtbl_call(pClient, 11, proto)
            stop_func(pClient.value)

        except Exception as e:
            print(f"[WASAPI] 捕获异常: {e}")
        finally:
            # 释放 COM 资源
            for ptr in [pCapture, pClient, pDev, pEnum]:
                if ptr and ptr.value:
                    try:
                        vtbl = ctypes.cast(ptr, POINTER(c_void_p))
                        entries = ctypes.cast(vtbl[0], POINTER(c_void_p))
                        release = WINFUNCTYPE(DWORD, c_void_p)(ctypes.cast(entries[2], c_void_p).value)
                        release(ptr.value)
                    except Exception:
                        pass
            if com_initialized:
                ole32.CoUninitialize()


# ═══════════════════════════════════════════════════════════
# 音频设备信息
# ═══════════════════════════════════════════════════════════

@dataclass
class AudioDevice:
    """音频设备信息"""
    id: int
    name: str
    device_type: str          # "loopback" | "input"
    channels: int
    sample_rate: int
    hostapi_name: str = ""


# ═══════════════════════════════════════════════════════════
# AudioCapture — 统一接口
# ═══════════════════════════════════════════════════════════

class AudioCapture:
    """
    音频捕获器

    两种模式:
    - 系统音频 (loopback): 使用 WASAPILoopbackCapture (纯 ctypes WASAPI)
    - 麦克风 (input): 使用 sounddevice

    用法:
        cap = AudioCapture()
        devices = AudioCapture.list_devices()
        cap.start(device_id, callback)
        cap.stop()
    """

    def __init__(self):
        self._sd_stream: Optional[sd.InputStream] = None
        self._wasapi_capture: Optional[WASAPILoopbackCapture] = None
        self._running = False
        self._lock = threading.Lock()
        self._mode: str = ""          # "loopback" | "input"
        self._callback: Optional[Callable[[np.ndarray], None]] = None

    @staticmethod
    def list_devices() -> List[AudioDevice]:
        """枚举可用音频设备"""
        devices: List[AudioDevice] = []

        # ── 系统音频: 检查 WASAPI loopback 是否可用 ──
        if WASAPILoopbackCapture.is_available():
            # 获取默认输出设备名称
            try:
                default_out = sd.query_devices(kind='output')
                devices.append(AudioDevice(
                    id=-1,  # 特殊 ID 表示默认系统音频
                    name=f"系统音频 ({default_out['name']})",
                    device_type="loopback",
                    channels=2,
                    sample_rate=LOOPBACK_SAMPLE_RATE,
                    hostapi_name="Windows WASAPI",
                ))
            except Exception:
                devices.append(AudioDevice(
                    id=-1, name="系统音频 (WASAPI Loopback)",
                    device_type="loopback",
                    channels=2,
                    sample_rate=LOOPBACK_SAMPLE_RATE,
                ))

        # ── 麦克风设备: 通过 sounddevice 枚举 ──
        sd_devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for i, dev in enumerate(sd_devices):
            hostapi_name = hostapis[dev['hostapi']]['name'] if dev['hostapi'] < len(hostapis) else ""
            # 只取 MME/DirectSound/WASAPI 的输入设备
            if dev['max_input_channels'] > 0 and hostapi_name in (
                'MME', 'Windows DirectSound', 'Windows WASAPI'
            ):
                # 验证可用性
                try:
                    ch = min(dev['max_input_channels'], 2)
                    stream = sd.InputStream(
                        device=i, channels=ch,
                        samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE,
                        dtype='float32',
                    )
                    stream.close()
                    devices.append(AudioDevice(
                        id=i, name=dev['name'],
                        device_type="input",
                        channels=dev['max_input_channels'],
                        sample_rate=int(dev['default_samplerate']),
                        hostapi_name=hostapi_name,
                    ))
                except Exception:
                    # 尝试用设备默认采样率
                    try:
                        stream = sd.InputStream(
                            device=i, channels=min(dev['max_input_channels'], 2),
                            samplerate=int(dev['default_samplerate']),
                            blocksize=BLOCK_SIZE, dtype='float32',
                        )
                        stream.close()
                        devices.append(AudioDevice(
                            id=i, name=dev['name'],
                            device_type="input",
                            channels=dev['max_input_channels'],
                            sample_rate=int(dev['default_samplerate']),
                            hostapi_name=hostapi_name,
                        ))
                    except Exception:
                        pass

        return devices

    def start(self, device_id: Optional[int], callback: Callable[[np.ndarray], None]) -> None:
        """
        开始音频捕获

        Args:
            device_id: 设备 ID (-1 = 系统音频 loopback, 其他 = sounddevice 设备)
            callback: 回调 (1D float32 16kHz 单声道 numpy 数组)
        """
        with self._lock:
            if self._running:
                return

            self._callback = callback

            # ── 系统音频 Loopback ──
            if device_id == -1:
                self._mode = "loopback"
                self._wasapi_capture = WASAPILoopbackCapture()
                ok = self._wasapi_capture.start(callback)
                if not ok:
                    self._wasapi_capture = None
                    raise RuntimeError("WASAPI Loopback 启动失败")
                self._running = True
                return

            # ── 麦克风输入 ──
            self._mode = "input"
            def _sd_callback(indata, frames, time_info, status):
                if status:
                    print(f"[AudioCapture] 警告: {status}")
                audio = indata.astype(np.float32)
                if audio.ndim > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                audio = audio.flatten()
                callback(audio)

            try:
                dev_info = sd.query_devices(device_id)
                channels = min(dev_info.get('max_input_channels', 2), 2)
            except Exception:
                channels = CHANNELS

            # 尝试使用设备原生采样率，然后回调中重采样
            try:
                dev_sample_rate = int(sd.query_devices(device_id).get('default_samplerate', SAMPLE_RATE))
            except Exception:
                dev_sample_rate = SAMPLE_RATE

            # 如果采样率与管道不匹配，在回调中做重采样
            need_resample = (dev_sample_rate != SAMPLE_RATE)

            def _callback_with_resample(indata, frames, time_info, status):
                if status:
                    print(f"[AudioCapture] 警告: {status}")
                audio = indata.astype(np.float32)
                if audio.ndim > 1 and audio.shape[1] > 1:
                    audio = np.mean(audio, axis=1)
                audio = audio.flatten()
                if need_resample:
                    in_len = len(audio)
                    out_len = max(1, int(in_len * SAMPLE_RATE / dev_sample_rate))
                    indices = np.linspace(0, in_len - 1, out_len)
                    lo = np.floor(indices).astype(int)
                    hi = np.minimum(lo + 1, in_len - 1)
                    frac = indices - lo
                    audio = audio[lo] * (1 - frac) + audio[hi] * frac
                callback(audio)

            try:
                self._sd_stream = sd.InputStream(
                    device=device_id,
                    channels=channels,
                    samplerate=dev_sample_rate,
                    blocksize=BLOCK_SIZE,
                    dtype='float32',
                    callback=_callback_with_resample,
                )
                self._sd_stream.start()
                self._running = True
            except sd.PortAudioError as e:
                raise RuntimeError(f"音频设备打开失败: {e}") from e

    def stop(self) -> None:
        """停止音频捕获"""
        with self._lock:
            if self._sd_stream is not None:
                try:
                    self._sd_stream.stop()
                    self._sd_stream.close()
                except Exception:
                    pass
                self._sd_stream = None

            if self._wasapi_capture is not None:
                self._wasapi_capture.stop()
                self._wasapi_capture = None

            self._running = False

    @property
    def is_running(self) -> bool:
        return self._running
