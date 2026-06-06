"""
Edge-TTS 语音合成引擎（微软免费在线 TTS）
音质极好，零 ML 依赖，音频缓存到 D 盘
"""
import queue
import threading
import time
import asyncio
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).parent.parent

VOICE_OPTIONS = {
    "晓晓 (女声)": "zh-CN-XiaoxiaoNeural",
    "云希 (男声)": "zh-CN-YunxiNeural",
    "晓伊 (女声)": "zh-CN-XiaoyiNeural",
    "云扬 (男声-新闻)": "zh-CN-YunyangNeural",
}


class EdgeTTSEngine:
    """Edge-TTS 语音引擎"""

    def __init__(self, voice: str = "晓晓 (女声)", speed: float = 1.0):
        self._voice_name = voice
        self._voice_id = VOICE_OPTIONS.get(voice, "zh-CN-XiaoxiaoNeural")
        self._speed = speed
        self._available = True

        self._queue: queue.Queue = queue.Queue()
        self._worker: Optional[threading.Thread] = None
        self._running = False

        # 缓存目录
        self._cache_dir = PROJECT_ROOT / "exports" / "tts_cache"
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"[EdgeTTS] 引擎就绪 (voice={voice})")

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def voice_name(self) -> str:
        return self._voice_name

    @property
    def available_voices(self) -> list:
        return list(VOICE_OPTIONS.keys())

    def start(self):
        self._running = True
        self._worker = threading.Thread(
            target=self._worker_loop, name="EdgeTTS-Worker", daemon=True
        )
        self._worker.start()

    def stop(self):
        self._running = False
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def speak(self, text: str):
        if not text or not text.strip():
            return
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(text.strip())

    def set_speed(self, speed: float):
        self._speed = speed

    def set_voice(self, voice_name: str):
        if voice_name in VOICE_OPTIONS:
            self._voice_name = voice_name
            self._voice_id = VOICE_OPTIONS[voice_name]

    def _worker_loop(self):
        import io
        import winsound
        from pydub import AudioSegment

        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if text is None:
                break

            cache_path = self._cache_dir / f"tts_{hash(text) & 0xFFFFFFFF}.mp3"
            try:
                # 合成（异步转同步，缓存复用）
                if not cache_path.exists():
                    rate = "+" if self._speed >= 1.0 else "-"
                    rate_str = f"{rate}{int(abs(self._speed - 1.0) * 100)}%"
                    import edge_tts
                    async def _synth():
                        comm = edge_tts.Communicate(
                            text, self._voice_id, rate=rate_str
                        )
                        await comm.save(str(cache_path))
                    asyncio.run(_synth())

                # MP3 → 内存 WAV → winsound（不写 C 盘临时文件）
                if cache_path.exists():
                    audio = AudioSegment.from_mp3(str(cache_path))
                    wav_io = io.BytesIO()
                    audio.export(wav_io, format='wav')
                    winsound.PlaySound(wav_io.getvalue(), winsound.SND_MEMORY)
            except Exception as e:
                print(f"[EdgeTTS] 失败: {e}")
