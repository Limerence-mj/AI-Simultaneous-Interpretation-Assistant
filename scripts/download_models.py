"""
下载所有模型文件到 D 盘项目目录
运行: python scripts/download_models.py
所有模型下载到 ../models/ 目录下
"""
import os, sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# 强制所有缓存指向项目目录
os.environ["HF_HOME"] = str(MODELS_DIR / ".hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR)
os.environ["HF_HUB_CACHE"] = str(MODELS_DIR)

print(f"模型目录: {MODELS_DIR}")
print(f"HF 缓存: {MODELS_DIR / '.hf_cache'}")
print()


def download_faster_whisper(model_size="small"):
    """下载 faster-whisper 模型"""
    print(f"[1/4] 下载 faster-whisper-{model_size} (ASR)...")
    try:
        from faster_whisper import download_model
        download_model(model_size, output_dir=str(MODELS_DIR))
        print(f"  ✅ faster-whisper-{model_size} 完成")
    except Exception as e:
        print(f"  ⚠️ {e}, 尝试手动...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            f"Systran/faster-whisper-{model_size}",
            local_dir=str(MODELS_DIR / f"models--Systran--faster-whisper-{model_size}"),
            local_dir_use_symlinks=False,
        )
        print(f"  ✅ 完成")


def download_opus_mt():
    """下载 opus-mt-en-zh 翻译模型"""
    print("[2/4] 下载 opus-mt-en-zh (翻译)...")
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    model_name = "Helsinki-NLP/opus-mt-en-zh"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer.save_pretrained(str(MODELS_DIR / "opus-mt-en-zh"))
    model.save_pretrained(str(MODELS_DIR / "opus-mt-en-zh"))
    print(f"  ✅ opus-mt-en-zh 完成")


def download_sherpa_zipformer():
    """下载 Sherpa-ONNX Zipformer 流式模型"""
    print("[3/4] 下载 sherpa-onnx-zipformer-en (流式ASR备用)...")
    from huggingface_hub import snapshot_download
    snapshot_download(
        "csukuangfj/sherpa-onnx-streaming-zipformer-en-2023-02-21",
        local_dir=str(MODELS_DIR / "sherpa_zipformer_en"),
        local_dir_use_symlinks=False,
    )
    print(f"  ✅ sherpa-zipformer-en 完成")


def download_faster_whisper_tiny():
    """下载 tiny 模型（轻量快速备选）"""
    print("[4/4] 下载 faster-whisper-tiny (快速ASR)...")
    try:
        from faster_whisper import download_model
        download_model("tiny", output_dir=str(MODELS_DIR))
        print(f"  ✅ faster-whisper-tiny 完成")
    except Exception as e:
        print(f"  ⚠️ {e}, 尝试手动...")
        from huggingface_hub import snapshot_download
        snapshot_download(
            "Systran/faster-whisper-tiny",
            local_dir=str(MODELS_DIR / "models--Systran--faster-whisper-tiny"),
            local_dir_use_symlinks=False,
        )
        print(f"  ✅ 完成")


if __name__ == "__main__":
    # 检查已存在的模型
    existing = []
    if (MODELS_DIR / "models--Systran--faster-whisper-small").exists():
        existing.append("faster-whisper-small")
    if (MODELS_DIR / "models--Systran--faster-whisper-tiny").exists():
        existing.append("faster-whisper-tiny")
    if (MODELS_DIR / "opus-mt-en-zh").exists():
        existing.append("opus-mt-en-zh")
    if (MODELS_DIR / "sherpa_zipformer_en").exists():
        existing.append("sherpa-zipformer-en")

    if existing:
        print(f"已存在模型: {', '.join(existing)}")
        skip = input("是否跳过已存在的模型？(Y/n): ").strip().lower()
        if skip == 'n':
            existing = []

    if "faster-whisper-small" not in existing:
        download_faster_whisper("small")
    if "opus-mt-en-zh" not in existing:
        download_opus_mt()
    if "sherpa-zipformer-en" not in existing:
        download_sherpa_zipformer()
    if "faster-whisper-tiny" not in existing:
        download_faster_whisper_tiny()

    print()
    print("=" * 40)
    print("全部模型下载完成！")
    print(f"模型目录: {MODELS_DIR}")
    print("运行: python src/main.py")
