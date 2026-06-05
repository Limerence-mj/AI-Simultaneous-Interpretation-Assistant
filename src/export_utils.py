"""
导出工具 — TXT / SRT 字幕文件生成
导出文件仅存放于项目 exports/ 目录
"""
from pathlib import Path
from typing import List

from src.state_manager import TranslationRecord

PROJECT_ROOT = Path(__file__).parent.parent
EXPORTS_DIR = PROJECT_ROOT / "exports"


def _format_srt_time(seconds: float) -> str:
    """将秒数转为 SRT 时间格式 HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def export_txt(records: List[TranslationRecord], output_path: Path = None) -> Path:
    """
    导出为纯文本 TXT 文件（每行一句中文）

    Args:
        records: 翻译记录列表
        output_path: 输出路径 (默认 exports/export_YYYY-MM-DD_HH-MM-SS.txt)

    Returns:
        导出文件的路径
    """
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = EXPORTS_DIR / f"export_{ts}.txt"

    lines = []
    for r in records:
        if r.final_translation:
            # 有时间戳则加上
            if r.start_time > 0:
                lines.append(
                    f"[{_format_srt_time(r.start_time)}] {r.final_translation}"
                )
            else:
                lines.append(r.final_translation)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return output_path


def export_srt(records: List[TranslationRecord], output_path: Path = None) -> Path:
    """
    导出为标准 SRT 字幕文件

    Args:
        records: 翻译记录列表
        output_path: 输出路径

    Returns:
        导出文件的路径
    """
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

    if output_path is None:
        from datetime import datetime
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_path = EXPORTS_DIR / f"export_{ts}.srt"

    blocks = []
    for i, r in enumerate(records, 1):
        if r.final_translation:
            start = _format_srt_time(r.start_time)
            end = _format_srt_time(r.end_time)
            text = r.final_translation
            # 如果有修正，附加原文说明
            if r.is_corrected and r.first_translation != r.final_translation:
                text += f"\n(修正前: {r.first_translation})"

            blocks.append(f"{i}\n{start} --> {end}\n{text}\n")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(blocks))

    return output_path
