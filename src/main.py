"""
AI 同声传译助手 — 应用入口
完全离线运行，所有文件存放于项目根目录
用法: python src/main.py
"""
import os
import sys
from pathlib import Path

# ─── 环境初始化：强制所有缓存指向项目目录 ───
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
os.environ["HF_HOME"] = str(MODELS_DIR / ".hf_cache")
os.environ["TRANSFORMERS_CACHE"] = str(MODELS_DIR)
os.environ["HUGGINGFACE_HUB_CACHE"] = str(MODELS_DIR)
os.environ["HF_HUB_CACHE"] = str(MODELS_DIR)


def main():
    """启动 GUI 应用"""
    # 日志初始化
    from src.logger import setup_logger
    logger = setup_logger("AITranslator", "INFO")
    logger.info("=" * 40)
    logger.info("AI 同声传译助手 启动")
    logger.info(f"项目根目录: {PROJECT_ROOT}")
    logger.info(f"模型目录: {MODELS_DIR}")

    # 配置加载
    from src.config_manager import ConfigManager
    config = ConfigManager()
    config.load()
    logger.info("配置加载完成")

    # 检查模型文件
    mt_model = MODELS_DIR / "opus-mt-en-zh"
    if not (mt_model.exists() and any(mt_model.iterdir())):
        logger.warning("翻译模型未找到，首次使用需下载")
    logger.info("模型检查完成")

    # 启动 GUI
    from PyQt6.QtWidgets import QApplication
    from src.gui import MainWindow, SubtitleWindow

    app = QApplication(sys.argv)
    app.setApplicationName("AI 同声传译助手")

    # 主控制面板
    main_window = MainWindow()
    main_window.show()

    logger.info("GUI 已启动")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
