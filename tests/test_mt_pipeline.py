"""
步骤四验证脚本：MT 翻译 + 协调器 + 端到端管道测试
测试英文→中文翻译质量和完整 VAD→ASR→MT 管道
"""
import os
import sys
import time
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / "models"))


def load_audio(filepath):
    with wave.open(str(filepath), 'rb') as wf:
        frames = wf.readframes(wf.getnframes())
        return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0


def run_test():
    print("=" * 55)
    print("  步骤四验证：翻译引擎 + 协调器 + 端到端管道")
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
    # 1. MT 引擎单句翻译
    # ═══════════════════════════════════════════
    print("\n── 1. MT 引擎单句翻译 ──")
    from src.mt_engine import MTEngine

    mt = MTEngine()
    load_t = mt.load_model()
    print(f"  模型加载: {load_t:.1f}s")
    check(f"MT 模型加载 ({load_t:.1f}s)", load_t < 30)

    # 基础翻译测试
    test_cases = [
        ("Hello world.", "你好"),
        ("The Birch canoe slid on the smooth planks.", None),  # 任意合理输出
        ("It is easy to tell the depth of a well.", None),
        ("Glue the sheet to the dark blue background.", None),
    ]

    all_valid = True
    for en, expected_hint in test_cases:
        t0 = time.time()
        zh = mt.translate(en)
        dt = (time.time() - t0) * 1000
        valid = len(zh) > 0 and dt < 1000
        hint = f"→ '{zh}' ({dt:.0f}ms)"
        if not valid:
            all_valid = False
            hint += " ❌"
        print(f"    EN: \"{en[:50]}\" {hint}")

    check(f"单句翻译有效 ({len(test_cases)}句)", all_valid)

    # ═══════════════════════════════════════════
    # 2. 上下文翻译测试
    # ═══════════════════════════════════════════
    print("\n── 2. 上下文增强翻译 ──")

    # 模拟代词消歧场景
    context = ["Apple announced a new chip."]
    without_ctx = mt.translate("It is very fast.")
    with_ctx = mt.translate("It is very fast.", context=context)

    print(f"  上下文: {context}")
    print(f"  无上下文: \"{without_ctx}\"")
    print(f"  有上下文: \"{with_ctx}\"")

    check("上下文翻译非空", len(with_ctx) > 0)
    # 上下文翻译可能不同（理想情况下应更准确），这里只检查非空

    # ═══════════════════════════════════════════
    # 3. 翻译协调器
    # ═══════════════════════════════════════════
    print("\n── 3. 翻译协调器 ──")
    from src.translator_coordinator import TranslationCoordinator
    from src.asr_engine import ASRResult

    coordinator = TranslationCoordinator()
    coordinator.mt._model = mt._model     # 复用已加载模型
    coordinator.mt._tokenizer = mt._tokenizer

    # 模拟 ASR 结果
    mock_results = [
        ASRResult(segment_id=1, en_text="Hello world.", confidence=0.9, start_ms=0, end_ms=1000),
        ASRResult(segment_id=2, en_text="This is a test.", confidence=0.85, start_ms=1500, end_ms=3000),
        ASRResult(segment_id=3, en_text="", confidence=0.0, start_ms=3500, end_ms=4000),  # 空文本
    ]

    coordinator.state.set_status(AppStatus.RUNNING)

    for ar in mock_results:
        result = coordinator.process(ar)
        print(f"  ASR #{ar.segment_id}: \"{ar.en_text[:50]}\" → MT: \"{result.zh_text[:50]}\"")

    check("协调器生产 2 句有效翻译", coordinator.state.translated_count == 2)
    check(f"字幕非空: '{coordinator.state.current_subtitle}'", len(coordinator.state.current_subtitle) > 0)
    check(f"平均翻译延迟 {coordinator.avg_translation_time_ms:.0f}ms", coordinator.avg_translation_time_ms < 500)

    # ═══════════════════════════════════════════
    # 4. 端到端管道 (VAD → ASR → MT)
    # ═══════════════════════════════════════════
    print("\n── 4. 端到端管道测试 ──")
    from src.vad import VADProcessor
    from src.asr_engine import ASREngine

    test_wav = PROJECT_ROOT / "exports" / "test_clip_12s.wav"
    if not test_wav.exists():
        test_wav = PROJECT_ROOT / "exports" / "test_speech_16k.wav"

    if test_wav.exists():
        audio = load_audio(test_wav)[:int(12 * 16000)]
    else:
        check("测试音频不存在", False)
        return False

    # 加载 ASR 模型
    asr = ASREngine(model_size="small", device="cpu")
    asr.load_model()

    # 重置协调器
    coordinator.reset()

    # 运行管道
    BLOCK_SIZE = 1024
    vad = VADProcessor()
    e2e_results = []

    t0 = time.time()
    for offset in range(0, len(audio), BLOCK_SIZE):
        chunk = audio[offset:offset + BLOCK_SIZE]
        if len(chunk) < BLOCK_SIZE:
            pad = np.zeros(BLOCK_SIZE - len(chunk), dtype=np.float32)
            chunk = np.concatenate([chunk, pad])

        seg = vad.process(chunk)
        if seg is not None:
            ar = asr.transcribe(seg.audio_data)
            ar.segment_id = seg.id
            ar.start_ms = seg.start_ms
            ar.end_ms = seg.end_ms

            mr = coordinator.process(ar)
            e2e_results.append((ar, mr))

    flushed = vad.flush()
    if flushed is not None:
        ar = asr.transcribe(flushed.audio_data)
        ar.segment_id = flushed.id
        mr = coordinator.process(ar)
        e2e_results.append((ar, mr))

    elapsed = time.time() - t0

    print(f"  处理 12s 音频 → {len(e2e_results)} 句翻译 ({elapsed:.1f}s)")
    for ar, mr in e2e_results:
        print(f"    [{ar.start_ms/1000:.1f}s] EN: {ar.en_text[:60]}")
        print(f"             ZH: {mr.zh_text[:60]}")

    check(f"端到端产出 {len(e2e_results)} 句", len(e2e_results) >= 2)

    all_zh_valid = all(len(mr.zh_text) > 0 for _, mr in e2e_results)
    check("所有翻译非空", all_zh_valid)

    rtf = elapsed / 12.0
    print(f"  RTF: {rtf:.2f}x")
    check(f"RTF < 3x ({rtf:.1f}x)", rtf < 3.0)

    # ═══════════════════════════════════════════
    # 5. StateManager 验证
    # ═══════════════════════════════════════════
    print("\n── 5. StateManager 状态验证 ──")
    check(f"翻译计数: {coordinator.state.translated_count}", coordinator.state.translated_count >= 2)
    check(f"历史记录: {len(coordinator.state.history)} 条", len(coordinator.state.history) >= 2)
    check(f"字幕非空", len(coordinator.state.current_subtitle) > 0)

    # ═══════════════════════════════════════════
    # 总结
    # ═══════════════════════════════════════════
    print("\n" + "=" * 55)
    print(f"  步骤四验证: {passed} 通过, {failed} 失败")
    if passed > 0 and failed == 0:
        print("  ✅ 步骤四完成！可进入步骤五（上下文修正机制）。")
    else:
        print(f"  ⚠️  有 {failed} 项失败，请检查。")
    print("=" * 55)

    return failed == 0


if __name__ == "__main__":
    from src.state_manager import AppStatus
    success = run_test()
    sys.exit(0 if success else 1)
