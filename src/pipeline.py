"""VAD → ASR 管道 v4 — 异步翻译队列"""
import time, threading, queue
from typing import List, Optional, Callable
import numpy as np
from src.vad import VADProcessor, SpeechSegment
from src.asr_engine import ASREngine, ASRResult

class Pipeline:
    def __init__(self, model_size="small", device="cpu",
                 on_result: Optional[Callable[[ASRResult], None]] = None):
        self.on_result = on_result
        self.vad = VADProcessor(threshold=0.5, min_silence_ms=250,
                                min_speech_ms=200, max_speech_ms=3000)
        self.asr = ASREngine(model_size=model_size, device=device,
                             compute_type="int8" if device=="cpu" else "float16",
                             beam_size=1, cpu_threads=4)
        self._results: List[ASRResult] = []
        self._running = False
        self._asr_queue = queue.Queue(maxsize=10)
        self._asr_worker: Optional[threading.Thread] = None
        self._mt_queue = queue.Queue(maxsize=50)
        self._mt_worker: Optional[threading.Thread] = None

    def start(self):
        if not self.asr._model:
            t0 = time.time(); self.asr.load_model()
            print(f"[Pipeline] ASR 就绪 ({time.time()-t0:.1f}s, {self.asr.model_size})")
        self._running = True
        self._asr_worker = threading.Thread(target=self._asr_loop, name="ASR", daemon=True)
        self._asr_worker.start()
        self._mt_worker = threading.Thread(target=self._mt_loop, name="MT", daemon=True)
        self._mt_worker.start()

    def feed(self, audio_chunk):
        if not self._running: return
        seg = self.vad.process(audio_chunk)
        if seg is None: return
        try: self._asr_queue.put_nowait(seg)
        except queue.Full:
            try: self._asr_queue.get_nowait()
            except queue.Empty: pass
            try: self._asr_queue.put_nowait(seg)
            except queue.Full: pass

    def _asr_loop(self):
        while self._running or not self._asr_queue.empty():
            try: seg = self._asr_queue.get(timeout=0.3)
            except queue.Empty: continue
            try:
                r = self.asr.transcribe(seg.audio_data)
                r.segment_id = seg.id; r.start_ms = seg.start_ms; r.end_ms = seg.end_ms
                self._results.append(r); self._mt_queue.put(r)
            except Exception as e: print(f"[Pipeline] ASR 失败: {e}")

    def _mt_loop(self):
        while self._running or not self._mt_queue.empty():
            try: r = self._mt_queue.get(timeout=0.3)
            except queue.Empty: continue
            try:
                if self.on_result: self.on_result(r)
            except Exception:
                import traceback; traceback.print_exc()

    def finish(self):
        self._running = False
        if self._asr_worker and self._asr_worker.is_alive(): self._asr_worker.join(timeout=30)
        # 等待 MT 队列清空
        if self._mt_worker and self._mt_worker.is_alive():
            self._mt_worker.join(timeout=30)
        seg = self.vad.flush()
        if seg:
            try:
                r = self.asr.transcribe(seg.audio_data)
                r.segment_id = seg.id; r.start_ms = seg.start_ms; r.end_ms = seg.end_ms
                self._results.append(r)
                if self.on_result: self.on_result(r)
            except Exception as e: print(f"[Pipeline] Flush 失败: {e}")
        return self._results

    def reset(self):
        self.vad.reset(); self._results.clear()
        for q in [self._asr_queue, self._mt_queue]:
            while not q.empty():
                try: q.get_nowait()
                except queue.Empty: break
