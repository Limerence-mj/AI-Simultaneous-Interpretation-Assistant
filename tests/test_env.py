"""
步骤一：环境搭建与模型就绪 — 验证脚本
检查依赖、加载模型、执行推理测试
所有模型和缓存文件仅存放于 D:/AI-Simultaneous-Interpretation-Assistant/models/
"""
import os
import sys
import time
from pathlib import Path

# ═══════════════════════════════════════════════════
# 关键：强制所有模型缓存指向 D 盘项目目录
# ═══════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# HuggingFace 全家桶缓存 → models/
os.environ["HF_HOME"] = str(MODELS_DIR / ".hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR)
os.environ["HF_HUB_CACHE"] = str(MODELS_DIR)
# 禁止写入 ~/.cache (C盘)
os.environ["XDG_CACHE_HOME"] = str(MODELS_DIR / ".xdg_cache")

print(f"📁 模型目录: {MODELS_DIR}")
print(f"📁 HF_HOME:   {os.environ['HF_HOME']}")
print()

# 结果统计
passed = 0
failed = 0


def check(name, condition, fatal=False):
    global passed, failed
    if condition:
        print(f"  ✅ {name}")
        passed += 1
    else:
        print(f"  ❌ {name}")
        failed += 1
        if fatal:
            print("\n  [致命] 无法继续。")
            sys.exit(1)


# ═══════════════════════════════════════════════════
# 1. Python 版本
# ═══════════════════════════════════════════════════
print("── 1. Python 版本 ──")
v = sys.version_info
check(f"Python {v.major}.{v.minor}.{v.micro} (>=3.10)", v >= (3, 10), fatal=True)

# ═══════════════════════════════════════════════════
# 2. 核心依赖
# ═══════════════════════════════════════════════════
print("\n── 2. 核心依赖 ──")
deps = [
    ("faster_whisper", "faster-whisper (ASR)"),
    ("transformers", "transformers (MT)"),
    ("torch", "torch (推理框架)"),
    ("sounddevice", "sounddevice (音频)"),
    ("numpy", "numpy (数据处理)"),
]
for mod, name in deps:
    try:
        __import__(mod)
        check(f"{name}", True)
    except ImportError:
        check(f"{name} — 运行: pip install {mod}", False)

# ═══════════════════════════════════════════════════
# 3. GPU 检测
# ═══════════════════════════════════════════════════
print("\n── 3. GPU / 设备 ──")
try:
    import torch
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        print(f"  🎮 {gpu} ({mem:.1f} GB)")
        device_for_asr = "cuda"
    else:
        print(f"  💻 无 GPU，使用 CPU 模式")
        device_for_asr = "cpu"
except:
    print(f"  💻 无法检测 GPU，使用 CPU")
    device_for_asr = "cpu"

# ═══════════════════════════════════════════════════
# 4. ASR 模型加载与推理测试 (faster-whisper)
# ═══════════════════════════════════════════════════
print("\n── 4. ASR 模型 (Whisper small) ──")

asr_ok = False
try:
    from faster_whisper import WhisperModel
    import numpy as np

    # 使用 "small" 作为 model key，faster-whisper 负责下载/缓存
    # download_root 确保 CTranslate2 模型缓存至项目 models/ 目录
    model_key = "small"
    print(f"  加载模型: {model_key} ...")
    t0 = time.time()
    model = WhisperModel(
        model_key,
        device=device_for_asr,
        compute_type="float16" if device_for_asr == "cuda" else "int8",
        cpu_threads=4,
        download_root=str(MODELS_DIR),
    )
    load_time = time.time() - t0
    print(f"  加载耗时: {load_time:.1f}s")

    # 用一段简单英文测试音频（生成 3 秒 440Hz 正弦波模拟语音）
    sample_rate = 16000
    t = np.linspace(0, 3, sample_rate * 3, dtype=np.float32)
    # 用 "Hello world" 的简单频率模式
    test_audio = np.sin(2 * np.pi * 440 * t) * 0.5
    test_audio = test_audio.astype(np.float32)

    print(f"  执行识别测试...")
    t1 = time.time()
    segments, info = model.transcribe(test_audio, language="en", beam_size=5)
    results = [s.text.strip() for s in segments]
    infer_time = time.time() - t1

    check(f"加载成功 ({load_time:.1f}s), 识别测试完成 ({infer_time:.2f}s)", True)
    print(f"    模型: {info.language} (概率 {info.language_probability:.2f})")
    asr_ok = True

except Exception as e:
    check(f"ASR 模型加载/推理失败: {e}", False)

# ═══════════════════════════════════════════════════
# 5. MT 模型加载与推理测试 (opus-mt-en-zh)
# ═══════════════════════════════════════════════════
print("\n── 5. MT 模型 (opus-mt-en-zh) ──")

mt_ok = False
try:
    from transformers import MarianMTModel, MarianTokenizer

    local_mt = MODELS_DIR / "opus-mt-en-zh"
    model_name = str(local_mt) if (local_mt.exists() and any(local_mt.iterdir())) else "Helsinki-NLP/opus-mt-en-zh"

    print(f"  加载模型: {model_name} ...")
    t0 = time.time()
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)
    load_time = time.time() - t0

    print(f"  执行翻译测试...")
    t1 = time.time()
    test_input = "Hello, this is a test sentence for machine translation."
    inputs = tokenizer(test_input, return_tensors="pt")
    outputs = model.generate(**inputs, max_length=200, num_beams=4)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    infer_time = time.time() - t1

    check(f"加载成功 ({load_time:.1f}s), 翻译测试: '{test_input}' → '{result}'", len(result) > 0)
    print(f"    翻译耗时: {infer_time * 1000:.0f}ms")
    mt_ok = True

except Exception as e:
    check(f"MT 模型加载/推理失败: {e}", False)

# ═══════════════════════════════════════════════════
# 6. 音频设备
# ═══════════════════════════════════════════════════
print("\n── 6. 音频设备 ──")
try:
    import sounddevice as sd
    devices = sd.query_devices()
    inputs = [d for d in devices if d["max_input_channels"] > 0]
    check(f"检测到 {len(inputs)} 个输入设备", len(inputs) > 0)
    for d in inputs[:5]:
        print(f"    - {d['name']}")
except Exception as e:
    check(f"音频设备检测失败: {e}", False)

# ═══════════════════════════════════════════════════
# 7. 内存占用
# ═══════════════════════════════════════════════════
print("\n── 7. 内存占用 ──")
try:
    import psutil
    mem = psutil.Process().memory_info().rss / (1024**3)
    check(f"当前内存: {mem:.2f} GB (目标 <2.5GB)", mem < 2.5)
except:
    print("  ⚠️  psutil 未安装，跳过内存检测")

# ═══════════════════════════════════════════════════
# 总结
# ═══════════════════════════════════════════════════
print("\n" + "=" * 50)
print(f"  步骤一验证: {passed} 通过, {failed} 失败")
if passed > 0 and failed == 0:
    print("  ✅ 步骤一完成！环境就绪，可进入步骤二。")
elif asr_ok or mt_ok:
    print("  ⚠️  部分通过，部分模型需下载。")
else:
    print("  ❌ 有失败项，请检查依赖安装。")
print("=" * 50)
