"""
日志系统 — 控制台输出 + 文件写入
日志文件仅存放于项目 logs/ 目录，保留 7 天
不记录原始音频数据
"""
import logging
import os
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"

# 日志格式
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logger(
    name: str = "AITranslator",
    level: str = "INFO",
    console: bool = True,
) -> logging.Logger:
    """
    初始化日志系统

    Args:
        name: 日志器名称
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        console: 是否输出到控制台

    Returns:
        配置好的 Logger 实例
    """
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # 文件输出 — 按天滚动
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOGS_DIR / f"app_{today}.log"

    fh = logging.FileHandler(str(log_file), encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # 文件记录全部级别
    fh.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
    logger.addHandler(fh)

    # 控制台输出
    if console:
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, level.upper(), logging.INFO))
        ch.setFormatter(logging.Formatter(LOG_FORMAT, DATE_FORMAT))
        logger.addHandler(ch)

    # 自动清理 7 天前的日志
    _cleanup_old_logs()

    return logger


def get_logger(name: str = "AITranslator") -> logging.Logger:
    """获取已配置的 logger（若不存在则创建）"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger


def _cleanup_old_logs(retention_days: int = 7):
    """清理过期日志文件"""
    import time
    now = time.time()
    cutoff = now - retention_days * 86400

    try:
        for f in LOGS_DIR.glob("app_*.log"):
            if f.stat().st_mtime < cutoff:
                f.unlink()
    except Exception:
        pass  # 清理失败不阻塞主流程
