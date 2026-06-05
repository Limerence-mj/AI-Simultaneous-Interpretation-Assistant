"""
步骤六验证脚本：完整 GUI + 辅助功能测试
测试配置管理、日志系统、导出功能、GUI 模块
"""
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "models" / ".hf_cache"))


def run_test():
    print("=" * 55)
    print("  步骤六验证：完整 GUI + 辅助功能")
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
    # 1. ConfigManager
    # ═══════════════════════════════════════════
    print("\n── 1. ConfigManager 配置管理 ──")
    from src.config_manager import ConfigManager

    # 用临时文件测试
    test_cfg = PROJECT_ROOT / "exports" / "_test_config.json"
    mgr = ConfigManager(test_cfg)
    data = mgr.load()
    check("默认配置加载", len(data) > 5)
    check("默认字体大小", data.get("font_size") == 24)

    mgr.set("font_size", 32)
    mgr.set("opacity", 0.8)
    mgr.save()
    check("配置保存成功", test_cfg.exists())

    # 重新加载验证
    mgr2 = ConfigManager(test_cfg)
    data2 = mgr2.load()
    check("配置持久化 (font_size=32)", data2.get("font_size") == 32)
    check("配置持久化 (opacity=0.8)", abs(data2.get("opacity", 0) - 0.8) < 0.01)

    # 清理
    test_cfg.unlink(missing_ok=True)

    # ═══════════════════════════════════════════
    # 2. Logger
    # ═══════════════════════════════════════════
    print("\n── 2. Logger 日志系统 ──")
    from src.logger import setup_logger, get_logger

    log = setup_logger("TestLogger", "DEBUG", console=False)
    log.debug("debug message")
    log.info("info message")
    log.warning("warning message")
    log.error("error message")

    # 检查日志文件
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = PROJECT_ROOT / "logs" / f"app_{today}.log"
    check(f"日志文件存在: {log_file.name}", log_file.exists())

    if log_file.exists():
        content = log_file.read_text(encoding="utf-8")
        check("日志含 INFO", "INFO" in content)
        check("日志含 WARNING", "WARNING" in content)
        check("日志含 ERROR", "ERROR" in content)
        check("日志文件在项目目录", str(log_file).startswith(str(PROJECT_ROOT)))

    # ═══════════════════════════════════════════
    # 3. Export TXT / SRT
    # ═══════════════════════════════════════════
    print("\n── 3. 导出功能 (TXT / SRT) ──")
    from src.state_manager import TranslationRecord
    from src.export_utils import export_txt, export_srt

    records = [
        TranslationRecord(
            id=1, start_time=1.5, end_time=4.2,
            en_text="Hello world.",
            first_translation="你好世界。",
            final_translation="你好世界。",
            is_corrected=False, correction_count=0,
        ),
        TranslationRecord(
            id=2, start_time=5.0, end_time=8.3,
            en_text="It is very fast.",
            first_translation="它非常快。",
            final_translation="新芯片非常快。",
            is_corrected=True, correction_count=1,
        ),
    ]

    txt_path = export_txt(records)
    check(f"TXT 导出: {txt_path.name}", txt_path.exists())
    txt_content = txt_path.read_text(encoding="utf-8")
    check("TXT 包含译文", "你好世界" in txt_content)
    check("TXT 包含修正后译文", "新芯片非常快" in txt_content)
    check("TXT 在项目目录", str(txt_path).startswith(str(PROJECT_ROOT)))

    srt_path = export_srt(records)
    check(f"SRT 导出: {srt_path.name}", srt_path.exists())
    srt_content = srt_path.read_text(encoding="utf-8")
    check("SRT 含序号", "1\n" in srt_content)
    check("SRT 含时间轴", "-->" in srt_content)
    check("SRT 含修正标记", "修正前" in srt_content)

    # ═══════════════════════════════════════════
    # 4. GUI 模块
    # ═══════════════════════════════════════════
    print("\n── 4. GUI 模块检查 ──")
    import py_compile
    gui_file = PROJECT_ROOT / "src" / "gui.py"
    main_file = PROJECT_ROOT / "src" / "main.py"
    try:
        py_compile.compile(str(gui_file), doraise=True)
        check("gui.py 语法正确", True)
    except py_compile.PyCompileError as e:
        check(f"gui.py 语法: {e}", False)
    try:
        py_compile.compile(str(main_file), doraise=True)
        check("main.py 语法正确", True)
    except py_compile.PyCompileError as e:
        check(f"main.py 语法: {e}", False)

    # ═══════════════════════════════════════════
    # 5. 完整模块导入
    # ═══════════════════════════════════════════
    print("\n── 5. 完整模块导入 ──")
    modules = [
        "src.config_manager",
        "src.logger",
        "src.export_utils",
        "src.state_manager",
        "src.mt_engine",
        "src.vad",
        "src.asr_engine",
        "src.audio_capture",
        "src.pipeline",
        "src.translator_coordinator",
        "src.main",
    ]
    import importlib
    for mod_name in modules:
        try:
            importlib.import_module(mod_name)
            check(f"导入 {mod_name}", True)
        except Exception as e:
            check(f"导入 {mod_name}: {e}", False)

    # ═══════════════════════════════════════════
    # 总结
    # ═══════════════════════════════════════════
    print("\n" + "=" * 55)
    print(f"  步骤六验证: {passed} 通过, {failed} 失败")
    if passed > 0 and failed == 0:
        print("  ✅ 步骤六完成！可进入步骤七（TTS 语音播报）。")
    else:
        print(f"  ⚠️  有 {failed} 项失败，请检查。")
    print("=" * 55)

    return failed == 0


if __name__ == "__main__":
    success = run_test()
    sys.exit(0 if success else 1)
