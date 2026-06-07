"""VAD — 纯 numpy 能量检测"""
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

@dataclass
class SpeechSegment:
    id: int; start_ms: float; end_ms: float; audio_data: bytes; duration_ms: float

SAMPLE_RATE = 16000
MIN_SPEECH_MS = 200; MAX_SPEECH_MS = 3000; MIN_SILENCE_MS = 250
SPEECH_RATIO = 3.5

class VADProcessor:
    def __init__(self, sample_rate=SAMPLE_RATE, threshold=0.5,
                 min_silence_ms=MIN_SILENCE_MS, min_speech_ms=MIN_SPEECH_MS,
                 max_speech_ms=MAX_SPEECH_MS):
        self.sample_rate = sample_rate
        self.min_speech_samples = int(sample_rate * min_speech_ms / 1000)
        self.max_speech_samples = int(sample_rate * max_speech_ms / 1000)
        self.min_silence_samples = int(sample_rate * min_silence_ms / 1000)
        self._chunks: List[tuple] = []
        self._total_samples = 0; self._trimmed_offset = 0
        self._noise_floor = 0.002
        self._segment_id = 0; self._in_speech = False
        self._speech_start = 0; self._silence_samples = 0; self._speech_samples = 0

    def process(self, audio_chunk: np.ndarray) -> Optional[SpeechSegment]:
        chunk = audio_chunk.astype(np.float32).flatten(); cl = len(chunk)
        self._chunks.append((self._total_samples, chunk.copy())); self._total_samples += cl
        rms = float(np.sqrt(np.mean(chunk ** 2) + 1e-10)); result = None
        if not self._in_speech:
            self._noise_floor = 0.95 * self._noise_floor + 0.05 * rms
        self._noise_floor = max(self._noise_floor, 0.0003)
        st = self._noise_floor * SPEECH_RATIO; sit = self._noise_floor * 1.3
        if self._in_speech:
            if rms < sit:
                self._silence_samples += cl; self._speech_samples += cl
                if self._silence_samples >= self.min_silence_samples:
                    end = self._total_samples - self._silence_samples
                    seg = self._mk(self._speech_start, end)
                    self._in_speech = False; self._silence_samples = 0; self._speech_samples = 0
                    if seg: self._trim(end); result = seg
            else:
                self._silence_samples = 0; self._speech_samples += cl
            if self._in_speech and self._speech_samples >= self.max_speech_samples:
                end = self._total_samples
                seg = self._mk(self._speech_start, end)
                self._speech_start = end; self._speech_samples = 0; self._silence_samples = 0
                if seg: self._trim(end); result = seg
        elif rms >= st:
            self._in_speech = True; self._speech_start = self._total_samples - cl
            self._silence_samples = 0; self._speech_samples = cl
        self._trim_old(); return result

    def _mk(self, start, end):
        dur = end - start
        if dur < self.min_speech_samples: return None
        a = self._extract(start, end)
        if len(a) == 0: return None
        self._segment_id += 1
        return SpeechSegment(id=self._segment_id, start_ms=start/16000*1000,
            end_ms=end/16000*1000, audio_data=(np.clip(a,-1,1)*32767).astype(np.int16).tobytes(),
            duration_ms=dur/16000*1000)

    def _extract(self, start, end):
        a, b = start-self._trimmed_offset, end-self._trimmed_offset
        if b <= 0: return np.array([], dtype=np.float32)
        aa = np.concatenate([c for _, c in self._chunks])
        a, b = max(0,a), min(len(aa),b)
        return aa[a:b] if a < b else np.array([], dtype=np.float32)

    def _trim(self, pos):
        c = pos - self._trimmed_offset
        if c <= 0: return
        self._chunks = [(o, ch) for o, ch in self._chunks if o+len(ch)-self._trimmed_offset > c]
        self._trimmed_offset = pos

    def _trim_old(self):
        mb = self.sample_rate * 15
        if self._total_samples - self._trimmed_offset <= mb: return
        tp = self._total_samples - mb
        self._chunks = [(s, c[max(0,tp-s):]) for s, c in self._chunks if s+len(c) > tp]
        self._trimmed_offset = tp

    def flush(self):
        if not self._in_speech: return None
        end = self._total_samples + self._trimmed_offset
        seg = self._mk(self._speech_start, end); self._in_speech = False; return seg

    def reset(self):
        self._chunks.clear(); self._total_samples = 0; self._trimmed_offset = 0
        self._in_speech = False; self._silence_samples = 0; self._speech_samples = 0
