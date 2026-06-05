"""
配置管理器 — 读写 config.json，持久化用户设置
文件仅存放于项目根目录
"""
import json
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.json"

DEFAULT_CONFIG: Dict[str, Any] = {
    "font_size": 24,
    "opacity": 0.6,
    "audio_device_id": None,
    "audio_source": "system",
    "source_language": "en",
    "target_language": "zh",
    "auto_correct": True,
    "tts_enabled": False,
    "tts_speed": 1.0,
    "tts_interrupt": True,
    "whisper_model": "small",
    "window_position": [None, None],
    "window_size": [800, 140],
    "window_monitor": 0,
    "hotwords_enabled": False,
    "log_level": "INFO",
}


class ConfigManager:
    """配置读写管理器"""

    def __init__(self, config_path: Path = DEFAULT_CONFIG_PATH):
        self._path = config_path
        self._data: Dict[str, Any] = {}

    def load(self) -> Dict[str, Any]:
        """加载配置，文件不存在或损坏时使用默认值"""
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
                # 补齐缺失的默认键
                for key, value in DEFAULT_CONFIG.items():
                    if key not in self._data:
                        self._data[key] = value
            except (json.JSONDecodeError, IOError):
                print("[Config] 配置文件损坏，使用默认值")
                self._data = dict(DEFAULT_CONFIG)
        else:
            self._data = dict(DEFAULT_CONFIG)
            self.save()  # 创建初始配置文件

        return self._data

    def save(self) -> None:
        """保存配置到文件"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def get(self, key: str, default: Any = None) -> Any:
        if not self._data:
            self.load()
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        if not self._data:
            self.load()
        self._data[key] = value

    def update(self, updates: Dict[str, Any]) -> None:
        if not self._data:
            self.load()
        self._data.update(updates)

    @property
    def data(self) -> Dict[str, Any]:
        if not self._data:
            self.load()
        return dict(self._data)
