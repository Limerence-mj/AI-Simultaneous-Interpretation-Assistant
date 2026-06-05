"""
步骤五验证脚本：上下文修正机制测试
测试 Reshoot 断句合并 + Context-aware Re-translation
"""
import os
import sys
import wave
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / ".hf_cache"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(PROJECT_ROOT / "models"))


def run_test():
    print("=" * 55)
    print("  步骤五验证：上下文修正机制")
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
    # 1. 编辑距离测试
    # ═══════════════════════════════════════════
    print("\n── 1. 编辑距离 (Levenshtein) ──")
    from src.translator_coordinator import _edit_distance

    dist_tests = [
        ("hello", "hello", 0),
        ("", "abc", 3),
        ("abc", "", 3),
        ("kitten", "sitting", 3),
        ("它非常快", "新芯片非常快", 3),
        ("它正在改变世界", "人工智能正在改变世界", 4),  # 插入 "人工智能" = 4 字符
        ("你好", "你好世界", 2),
    ]
    all_ok = True
    for a, b, expected in dist_tests:
        result = _edit_distance(a, b)
        ok = result == expected
        if not ok:
            all_ok = False
            print(f"    ❌ \"{a}\" vs \"{b}\": {result} (expected {expected})")
        else:
            print(f"    ✅ \"{a}\" vs \"{b}\": {result}")
    check(f"编辑距离 ({len(dist_tests)} 项)", all_ok)

    # ═══════════════════════════════════════════
    # 2. Reshoot 检测
    # ═══════════════════════════════════════════
    print("\n── 2. Reshoot 断句合并 ──")
    from src.translator_coordinator import TranslationCoordinator
    from src.asr_engine import ASRResult

    coordinator = TranslationCoordinator()

    # 场景: VAD 误切一句为两段
    # 句A: "I think the most important" (无句号, 结束于 2000ms)
    # 句B: "thing is timing" (开始于 2100ms, 间隔仅 100ms)
    ar1 = ASRResult(segment_id=1, en_text="I think the most important",
                    confidence=0.9, start_ms=0, end_ms=2000)
    ar2 = ASRResult(segment_id=2, en_text="thing is timing",
                    confidence=0.9, start_ms=2100, end_ms=3500)

    # 先处理第一句（不加载MT，只测检测逻辑）
    coordinator._last_asr_result = ar1  # 模拟上一句
    merged, did_reshoot = coordinator._check_reshoot(ar2)

    print(f"  句A: \"{ar1.en_text}\"")
    print(f"  句B: \"{ar2.en_text}\"")
    print(f"  间隔: {ar2.start_ms - ar1.end_ms}ms")
    print(f"  A句号结尾: {ar1.en_text[-1] in '.!?'}")
    print(f"  Reshoot触发: {did_reshoot}")
    if did_reshoot:
        print(f"  合并: \"{merged}\"")

    check("间隔 < 300ms 且无句号 → 触发 Reshoot", did_reshoot)
    check(f"合并文本正确: \"{merged}\"", merged == "I think the most important thing is timing")

    # 场景B: 不触发 Reshoot (A 有句号)
    ar3 = ASRResult(segment_id=3, en_text="Hello world.",
                    confidence=0.9, start_ms=1000, end_ms=3000)
    ar4 = ASRResult(segment_id=4, en_text="How are you.",
                    confidence=0.9, start_ms=3200, end_ms=5000)

    coordinator._last_asr_result = ar3
    merged2, did_reshoot2 = coordinator._check_reshoot(ar4)
    print(f"\n  场景B: \"{ar3.en_text}\" → \"{ar4.en_text}\" (gap={ar4.start_ms-ar3.end_ms}ms)")
    print(f"  Reshoot触发: {did_reshoot2}")
    check("句号结尾 → 不触发 Reshoot", not did_reshoot2)

    # 场景C: 不触发 Reshoot (间隔太大)
    ar5 = ASRResult(segment_id=5, en_text="A long sentence here.",
                    confidence=0.9, start_ms=0, end_ms=2000)
    ar6 = ASRResult(segment_id=6, en_text="Another sentence.",
                    confidence=0.9, start_ms=3000, end_ms=5000)

    coordinator._last_asr_result = ar5
    merged3, did_reshoot3 = coordinator._check_reshoot(ar6)
    print(f"\n  场景C: gap={ar6.start_ms-ar5.end_ms}ms")
    print(f"  Reshoot触发: {did_reshoot3}")
    check("间隔 > 300ms → 不触发 Reshoot", not did_reshoot3)

    # ═══════════════════════════════════════════
    # 3. 上下文修正 (需 MT 模型)
    # ═══════════════════════════════════════════
    print("\n── 3. 上下文修正 (Context-aware Re-translation) ──")
    from src.mt_engine import MTEngine

    # 加载 MT 模型
    mt = MTEngine()
    mt.load_model()

    # 设置 coordinator 的 MT 模型引用
    coordinator.mt._model = mt._model
    coordinator.mt._tokenizer = mt._tokenizer

    # 场景: 代词消歧
    # 句1: "Apple announced a new chip." → 上下文窗口
    # 句2: "It is very fast." → 首次: "它非常快", 上下文修正: 应更具体
    coordinator.reset()

    print("\n  代词消歧测试:")
    ar_ctx1 = ASRResult(segment_id=10, en_text="Apple announced a new chip.",
                        confidence=0.9, start_ms=0, end_ms=3000)
    ar_ctx2 = ASRResult(segment_id=11, en_text="It is very fast.",
                        confidence=0.9, start_ms=4000, end_ms=6000)

    r1 = coordinator.process(ar_ctx1)
    print(f"  句1: \"{ar_ctx1.en_text}\"")
    print(f"     → \"{r1.zh_text}\"")
    check("句1 无修正 (首句)", not r1.is_corrected)

    r2 = coordinator.process(ar_ctx2)
    print(f"  句2: \"{ar_ctx2.en_text}\"")
    print(f"     → \"{r2.zh_text}\"")
    if r2.is_corrected:
        print(f"     修正类型: {r2.correction_type}")
        print(f"     首次译文: \"{r2.prev_translation}\"")
        print(f"     最终译文: \"{r2.zh_text}\"")
    else:
        print(f"     (未触发修正，编辑距离不足以触发)")
        # 手动验证编辑距离
        first = mt.translate("It is very fast.")
        ctx = mt.translate("It is very fast.", context=["Apple announced a new chip."])
        dist = _edit_distance(first, ctx)
        print(f"     首次: \"{first}\"")
        print(f"     上下文: \"{ctx}\"")
        print(f"     编辑距离: {dist}")

    # 检查修正是否发生
    check("句2 非空翻译", len(r2.zh_text) > 0)

    # 如果发生了修正，验证正确性
    if r2.is_corrected:
        check(f"修正类型为 context", r2.correction_type == "context")
        check(f"有首次译文", r2.prev_translation is not None)

    # 检查历史记录
    history = coordinator.state.history
    check(f"历史记录: {len(history)} 条", len(history) == 2)

    corrected_records = [r for r in history if r.is_corrected]
    print(f"\n  修正记录数: {len(corrected_records)}")
    for rec in history:
        mark = "[修正]" if rec.is_corrected else ""
        print(f"    #{rec.id} {mark} EN: \"{rec.en_text[:50]}\"")
        print(f"       首次: \"{rec.first_translation}\"")
        print(f"       最终: \"{rec.final_translation}\"")

    # ═══════════════════════════════════════════
    # 4. 修正统计
    # ═══════════════════════════════════════════
    print("\n── 4. 修正统计 ──")
    stats = coordinator.correction_stats
    print(f"  总句数: {stats['total_sentences']}")
    print(f"  Reshoot: {stats['reshoot_count']}")
    print(f"  上下文修正: {stats['context_correction_count']}")
    check("统计正确", stats['total_sentences'] > 0)

    # ═══════════════════════════════════════════
    # 5. GUI 动画 (模块检查)
    # ═══════════════════════════════════════════
    print("\n── 5. GUI 修正动画 ──")
    try:
        import py_compile
        py_compile.compile(str(PROJECT_ROOT / "src" / "gui.py"), doraise=True)
        check("GUI 模块语法正确 (含修正动画)", True)
    except Exception as e:
        check(f"GUI 语法: {e}", False)

    # ═══════════════════════════════════════════
    # 总结
    # ═══════════════════════════════════════════
    print("\n" + "=" * 55)
    print(f"  步骤五验证: {passed} 通过, {failed} 失败")
    if passed > 0 and failed == 0:
        print("  ✅ 步骤五完成！可进入步骤六（完整 GUI 与辅助功能）。")
    else:
        print(f"  ⚠️  有 {failed} 项失败，请检查。")
    print("=" * 55)

    return failed == 0


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
