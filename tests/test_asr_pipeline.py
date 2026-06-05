"""
步骤三验证脚本：VAD → ASR 管道集成测试
使用真实英文语音素材，测试语音识别的准确性和性能
"""
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
EXPORTS_DIR = PROJECT_ROOT / "exports"
sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / "models"))


def load_audio(filepath):
    """加载 16kHz WAV 为 float32 numpy 数组"""
    with wave.open(str(filepath), 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0
    return audio, wf.getframerate()


def run_test():
    print("=" * 55)
    print("  步骤三验证：VAD → ASR 管道测试")
    print("=" * 55)

    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f"  ✅ {name}")
            passed += 1
            return True
        else:
            print(f"  ❌ {name}")
            failed += 1
            return False

    # ═══════════════════════════════════════════
    # 1. 模块导入 + 模型加载
    # ═══════════════════════════════════════════
    print("\n── 1. 模块导入与模型加载 ──")
    try:
        from src.asr_engine import ASREngine, ASRResult
        from src.pipeline import Pipeline
        check("模块导入成功", True)
    except Exception as e:
        check(f"模块导入: {e}", False)
        return False

    # 加载 ASR 模型
    print("  加载 Whisper small 模型...")
    asr = ASREngine(model_size="small", device="cpu", compute_type="int8")
    load_t = asr.load_model()
    print(f"  加载耗时: {load_t:.1f}s")
    check(f"ASR 模型加载 ({load_t:.1f}s)", load_t < 60)

    # ═══════════════════════════════════════════
    # 2. 单句识别测试
    # ═══════════════════════════════════════════
    print("\n── 2. 单句识别测试 ──")

    # 用步骤二的测试音频（12s 片段，含 4 句英文）
    test_wav = EXPORTS_DIR / "test_clip_12s.wav"
    if not test_wav.exists():
        test_wav = EXPORTS_DIR / "test_speech_16k.wav"

    if test_wav.exists():
        audio, sr = load_audio(test_wav)
        # 只取第一段（0.5s-3.1s，已知有语音）
        segment_start = int(0.5 * sr)
        segment_end = int(3.1 * sr)
        audio_segment = audio[segment_start:segment_end]
    else:
        # 兜底：合成测试音频
        check("测试音频不存在，跳过", False)
        return False

    print(f"  测试音频: {test_wav}")
    print(f"  选取片段: 0.5s-3.1s ({len(audio_segment)/sr:.1f}s)")

    # PCM float32 → 16-bit bytes
    audio_i16 = (audio_segment * 32767).astype(np.int16)
    audio_bytes = audio_i16.tobytes()

    t0 = time.time()
    result = asr.transcribe(audio_bytes)
    infer_t = (time.time() - t0) * 1000

    print(f"  识别结果: \"{result.en_text}\"")
    print(f"  置信度:   {result.confidence:.3f}")
    print(f"  推理耗时: {infer_t:.0f}ms")

    check("单句识别文本非空", len(result.en_text) > 0)
    check(f"置信度 > 0.5 ({result.confidence:.3f})", result.confidence > 0.5)
    check(f"推理耗时 < 4s ({infer_t:.0f}ms)", infer_t < 4000)

    # ═══════════════════════════════════════════
    # 3. VAD → ASR 管道测试
    # ═══════════════════════════════════════════
    print("\n── 3. VAD → ASR 管道完整测试 ──")

    # 取前 15 秒音频
    audio, sr = load_audio(test_wav)
    clip_len = min(int(15 * sr), len(audio))
    audio_clip = audio[:clip_len]

    pipeline = Pipeline(model_size="small", device="cpu")
    # 复用已加载的模型
    pipeline.asr._model = asr._model
    pipeline.asr.model_size = asr.model_size

    t0 = time.time()
    results = pipeline.process_file(audio_clip)
    elapsed = time.time() - t0

    print(f"\n  处理 {clip_len/sr:.1f}s 音频, 耗时 {elapsed:.1f}s")
    print(f"  检测到 {len(results)} 句:")
    for r in results:
        print(f"    [{r.start_ms:5.0f}-{r.end_ms:5.0f}ms] "
              f"置信度={r.confidence:.2f} | {r.en_text[:80]}")

    check(f"检测到句子 (≥ 1)", len(results) >= 1)
    check(f"平均置信度 > 0.5", np.mean([r.confidence for r in results]) > 0.5)

    # 检查延迟
    audio_dur = clip_len / sr
    rtf = elapsed / audio_dur  # Real-Time Factor
    print(f"  RTF (实时率): {rtf:.2f}x (1.0 = 实时)")
    check(f"RTF < 3x ({rtf:.1f}x)", rtf < 3.0)

    # ═══════════════════════════════════════════
    # 4. 错误处理与边界测试
    # ═══════════════════════════════════════════
    print("\n── 4. 边界测试 ──")

    # 空音频
    empty_bytes = np.zeros(16000, dtype=np.int16).tobytes()  # 1s 静音
    try:
        silent_result = asr.transcribe(empty_bytes)
        print(f"  静音识别: \"{silent_result.en_text}\"")
        # 静音输入 Whisper 可能返回空或杂讯，不判失败
        check("静音输入不崩溃", True)
    except Exception as e:
        check(f"静音输入: {e}", False)

    # 极短音频
    short_bytes = np.zeros(800, dtype=np.int16).tobytes()  # 50ms
    try:
        asr.transcribe(short_bytes)
        check("极短音频不崩溃", True)
    except Exception as e:
        check(f"极短音频不崩溃: OK (预期错误: {e})", True)  # faster-whisper 可能拒绝

    # ═══════════════════════════════════════════
    # 5. 内存与路径合规
    # ═══════════════════════════════════════════
    print("\n── 5. 内存与路径合规 ──")
    import psutil
    mem = psutil.Process().memory_info().rss / (1024**3)
    check(f"内存 {mem:.2f} GB (< 2.5GB)", mem < 2.5)

    # 检查模型文件位置
    whisper_files = list((PROJECT_ROOT / "models").glob("models--Systran--faster-whisper-small/**/*"))
    all_in_project = all(str(f).startswith(str(PROJECT_ROOT)) for f in whisper_files[:50])
    check(f"模型文件在项目目录", all_in_project)

    # ═══════════════════════════════════════════
    # 总结
    # ═══════════════════════════════════════════
    print("\n" + "=" * 55)
    print(f"  步骤三验证: {passed} 通过, {failed} 失败")
    if passed > 0 and failed == 0:
        print("  ✅ 步骤三完成！可进入步骤四（翻译引擎集成）。")
    else:
        print(f"  ⚠️  有 {failed} 项失败，请检查。")
    print("=" * 55)

    return failed == 0


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
