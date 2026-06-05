"""
AI 同声传译助手 — 完整 GUI
主控制面板 + 悬浮字幕窗口 + 历史记录 + 系统托盘
兼容 PyQt5 / PyQt6
"""
import sys
import time
import threading
from pathlib import Path

import numpy as np

# PyQt5/PyQt6 兼容导入
_PYQT_VERSION = 0
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QComboBox, QCheckBox, QSlider, QDialog,
        QTableWidget, QTableWidgetItem, QSystemTrayIcon, QMenu,
        QMessageBox, QHeaderView, QGroupBox, QGridLayout, QTabWidget,
    )
    from PyQt6.QtCore import Qt, QTimer, QPoint, pyqtSignal, QThread
    from PyQt6.QtGui import QFont, QAction, QIcon, QColor, QMouseEvent
    _PYQT_VERSION = 6
    _HeaderResizeMode = QHeaderView.ResizeMode.Interactive
    def _global_pos(event): return event.globalPosition().toPoint()
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QComboBox, QCheckBox, QSlider, QDialog,
        QTableWidget, QTableWidgetItem, QSystemTrayIcon, QMenu,
        QMessageBox, QHeaderView, QGroupBox, QGridLayout, QTabWidget, QAction,
    )
    from PyQt5.QtCore import Qt, QTimer, QPoint, pyqtSignal, QThread
    from PyQt5.QtGui import QFont, QIcon, QColor, QMouseEvent
    _PYQT_VERSION = 5
    _HeaderResizeMode = QHeaderView.Interactive
    def _global_pos(event): return event.globalPos()

from src.state_manager import StateManager, AppStatus, TranslationRecord
from src.config_manager import ConfigManager
from src.logger import get_logger

PROJECT_ROOT = Path(__file__).parent.parent

logger = get_logger(__name__)


# ─── 主窗口样式表 (暗色现代风格) ───
_MAIN_STYLESHEET = """
QMainWindow {
    background-color: #1a1d23;
}
QWidget {
    font-family: "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 15px;
    color: #e0e0e0;
}
QGroupBox {
    font-size: 16px;
    font-weight: bold;
    color: #8ab4f8;
    background-color: #21252b;
    border: 1px solid #333842;
    border-radius: 12px;
    margin-top: 18px;
    padding: 22px 16px 16px 16px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 10px;
    color: #8ab4f8;
}
QLabel {
    color: #c8ccd4;
    background: transparent;
    border: none;
    font-size: 15px;
}
QComboBox {
    background-color: #2c313a;
    color: #e0e0e0;
    border: 1px solid #3e4452;
    border-radius: 8px;
    padding: 10px 16px;
    min-height: 24px;
    font-size: 14px;
}
QComboBox:hover { border-color: #5a9eff; }
QComboBox::drop-down {
    border: none;
    width: 28px;
}
QComboBox QAbstractItemView {
    background-color: #2c313a;
    color: #e0e0e0;
    selection-background-color: #3a6fc5;
    border: 1px solid #3e4452;
    border-radius: 6px;
    font-size: 14px;
}
QSlider::groove:horizontal {
    background: #2c313a;
    height: 8px;
    border-radius: 4px;
}
QSlider::handle:horizontal {
    background: #5a9eff;
    width: 20px;
    height: 20px;
    margin: -6px 0;
    border-radius: 10px;
}
QSlider::handle:horizontal:hover {
    background: #7ab4ff;
}
QCheckBox {
    color: #c8ccd4;
    spacing: 10px;
    font-size: 15px;
}
QCheckBox::indicator {
    width: 22px;
    height: 22px;
    border: 2px solid #3e4452;
    border-radius: 5px;
    background-color: #2c313a;
}
QCheckBox::indicator:checked {
    background-color: #5a9eff;
    border-color: #5a9eff;
}
QPushButton {
    background-color: #2c313a;
    color: #e0e0e0;
    border: 1px solid #3e4452;
    border-radius: 10px;
    padding: 12px 22px;
    font-size: 15px;
    font-weight: 500;
    min-height: 22px;
}
QPushButton:hover {
    background-color: #373d48;
    border-color: #5a9eff;
}
QPushButton:pressed {
    background-color: #1e2229;
}
QPushButton#startBtn {
    background-color: #2d6a4f;
    color: #ffffff;
    border: 1px solid #40916c;
    font-size: 18px;
    font-weight: bold;
    padding: 16px 40px;
    border-radius: 12px;
    min-height: 32px;
}
QPushButton#startBtn:hover {
    background-color: #40916c;
}
QPushButton#startBtn:disabled {
    background-color: #2c313a;
    color: #666;
    border-color: #333842;
}
QPushButton#stopBtn {
    background-color: #6b3333;
    color: #ffffff;
    border: 1px solid #944545;
    font-size: 18px;
    font-weight: bold;
    padding: 16px 40px;
    border-radius: 12px;
    min-height: 32px;
}
QPushButton#stopBtn:hover {
    background-color: #8b3a3a;
}
QPushButton#stopBtn:disabled {
    background-color: #2c313a;
    color: #666;
    border-color: #333842;
}
QTableWidget {
    background-color: #21252b;
    alternate-background-color: #282c34;
    color: #c8ccd4;
    gridline-color: #333842;
    border: 1px solid #333842;
    border-radius: 8px;
    font-size: 14px;
}
QHeaderView::section {
    background-color: #2c313a;
    color: #8ab4f8;
    padding: 8px;
    border: none;
    border-bottom: 2px solid #3e4452;
    font-weight: bold;
    font-size: 14px;
}
QStatusBar {
    background-color: #1a1d23;
    color: #8ab4f8;
}
"""


# ─── 字幕悬浮窗口 (保持原有设计，增加修正动画) ───

class SubtitleWindow(QWidget):
    """半透明、置顶、可拖拽、无边框字幕窗口"""

    subtitle_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.state = StateManager()
        self._opacity = 0.6
        self._font_size = 32
        self._last_version = 0
        self._last_text = ""
        self._dragging = False
        self._drag_start_pos = QPoint()

        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        self.setWindowTitle("AI 同传字幕")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        screen = QApplication.primaryScreen().availableGeometry()
        w, h = 960, 200
        self.resize(w, h)
        self.move((screen.width() - w) // 2, screen.height() - h - 100)

        self._label = QLabel("")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setFont(QFont("Microsoft YaHei", self._font_size))
        self._label.setStyleSheet("color: #FFFFFF; padding: 16px 20px;")

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 10, 20, 10)
        layout.addWidget(self._label)
        self.setLayout(layout)

        self._update_background()

        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)

    def _setup_timer(self):
        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh)
        self._timer.start(100)

    def _refresh(self):
        if self.state.subtitle_version != self._last_version:
            self._last_version = self.state.subtitle_version
            text = self.state.current_subtitle
            is_correction = bool(self._last_text and text and self._last_text != text)
            self._label.setText(text)
            self.subtitle_changed.emit(text)
            self._last_text = text
            if is_correction:
                self._flash_correction()

    def _flash_correction(self):
        self.setStyleSheet(
            "SubtitleWindow { background-color: rgba(255, 200, 0, 200); border-radius: 12px; }"
        )
        QTimer.singleShot(800, self._update_background)

    def _update_background(self):
        alpha = int(self._opacity * 255)
        self.setStyleSheet(
            f"SubtitleWindow {{ background-color: rgba(0, 0, 0, {alpha}); border-radius: 12px; }}"
        )

    def _show_menu(self, pos):
        menu = QMenu(self)
        font_menu = menu.addMenu("字体大小")
        for size in [24, 32, 40]:
            a = QAction(f"{size}px", self)
            a.triggered.connect(lambda checked, s=size: self.set_font_size(s))
            font_menu.addAction(a)
        op_menu = menu.addMenu("透明度")
        for op in [0.2, 0.4, 0.6, 0.8]:
            a = QAction(f"{int(op*100)}%", self)
            a.triggered.connect(lambda checked, o=op: self.set_opacity(o))
            op_menu.addAction(a)
        menu.addSeparator()
        menu.addAction(QAction("退出", self, triggered=QApplication.quit))
        menu.exec(self.mapToGlobal(pos))

    def set_font_size(self, size: int):
        self._font_size = size
        self._label.setFont(QFont("Microsoft YaHei", size))

    def set_opacity(self, opacity: float):
        self._opacity = max(0.1, min(1.0, opacity))
        self._update_background()

    def set_subtitle_direct(self, text: str):
        self._label.setText(text)

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start_pos = _global_pos(e) - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e: QMouseEvent):
        if self._dragging:
            self.move(e.globalPosition().toPoint() - self._drag_start_pos)

    def mouseReleaseEvent(self, e: QMouseEvent):
        self._dragging = False


# ─── 历史记录窗口 ───

class HistoryDialog(QDialog):
    """翻译历史记录表格窗口"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.state = StateManager()
        self.setWindowTitle("翻译历史记录")
        self.resize(800, 500)
        self._setup_ui()

    def _setup_ui(self):
        self.setStyleSheet(_MAIN_STYLESHEET)
        self.setMinimumSize(800, 500)
        layout = QVBoxLayout(self)

        self._table = QTableWidget()
        self._table.setColumnCount(5)
        self._table.setHorizontalHeaderLabels(["时间", "英文原文", "首次译文", "最终译文", "修正"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.horizontalHeader().setSectionResizeMode(_HeaderResizeMode)
        self._table.setColumnWidth(0, 70)
        self._table.setColumnWidth(1, 200)
        self._table.setColumnWidth(2, 150)
        self._table.setColumnWidth(3, 150)
        self._table.setColumnWidth(4, 50)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(self._table)

        # 按钮栏
        btn_layout = QHBoxLayout()
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._refresh)
        btn_layout.addWidget(refresh_btn)

        copy_btn = QPushButton("复制选中行")
        copy_btn.clicked.connect(self._copy_selected)
        btn_layout.addWidget(copy_btn)

        copy_all_btn = QPushButton("复制全部")
        copy_all_btn.clicked.connect(self._copy_all)
        btn_layout.addWidget(copy_all_btn)

        btn_layout.addStretch()

        self._stats_label = QLabel()
        btn_layout.addWidget(self._stats_label)

        layout.addLayout(btn_layout)
        self._refresh()

    def _format_time(self, seconds: float) -> str:
        m = int(seconds // 60)
        s = int(seconds % 60)
        return f"{m:02d}:{s:02d}"

    def _refresh(self):
        records = self.state.history
        self._table.setRowCount(len(records))

        corrections = 0
        for row, r in enumerate(records):
            self._table.setItem(row, 0, QTableWidgetItem(self._format_time(r.start_time)))
            self._table.setItem(row, 1, QTableWidgetItem(r.en_text[:80]))
            self._table.setItem(row, 2, QTableWidgetItem(r.first_translation[:60]))
            self._table.setItem(row, 3, QTableWidgetItem(r.final_translation[:60]))
            corrected = "✓" if r.is_corrected else ""
            self._table.setItem(row, 4, QTableWidgetItem(corrected))
            if r.is_corrected:
                corrections += 1
                # 修正行黄色背景
                for col in range(5):
                    self._table.item(row, col).setBackground(QColor(255, 255, 200))

        self._stats_label.setText(
            f"共 {len(records)} 句 · 修正 {corrections} 次"
        )
        self._table.scrollToBottom()

    def _copy_selected(self):
        rows = set(i.row() for i in self._table.selectedItems())
        lines = []
        for row in sorted(rows):
            r = self.state.history[row] if row < len(self.state.history) else None
            if r:
                lines.append(
                    f"[{self._format_time(r.start_time)}] EN: {r.en_text}  →  ZH: {r.final_translation}"
                )
        if lines:
            QApplication.clipboard().setText("\n".join(lines))

    def _copy_all(self):
        lines = []
        for r in self.state.history:
            lines.append(
                f"[{self._format_time(r.start_time)}] EN: {r.en_text}  →  ZH: {r.final_translation}"
            )
        QApplication.clipboard().setText("\n".join(lines))


# ─── 主控制面板 ───

class MainWindow(QMainWindow):
    """主控制面板"""

    status_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.state = StateManager()
        self.config_mgr = ConfigManager()
        self.config_mgr.load()

        self._pipeline = None       # Pipeline 实例
        self._audio_capture = None  # AudioCapture 实例
        self._running = False
        self._start_time = 0.0

        self.setWindowTitle("🎙️ AI 同声传译助手")
        self.setMinimumSize(560, 620)
        self.resize(580, 660)
        self.setStyleSheet(_MAIN_STYLESHEET)

        self._setup_ui()
        self._load_config()
        self._setup_tray()

        # 启动时刷新设备列表
        self._refresh_devices()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(0)
        layout.setContentsMargins(0, 0, 0, 0)

        # ── 标题 ──
        title = QLabel("  🎙️  AI 同声传译助手")
        title.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: #8ab4f8; "
            "padding: 14px 18px; background: transparent; border: none;"
        )
        title.setFixedHeight(52)
        layout.addWidget(title)

        # ── 标签页 ──
        tabs = QTabWidget()
        tabs.setStyleSheet("""
            QTabWidget::pane { border: none; background: #1a1d23; }
            QTabBar::tab {
                font-size: 14px; padding: 10px 24px;
                background: #21252b; color: #999; border: none;
                border-bottom: 2px solid transparent;
            }
            QTabBar::tab:selected {
                color: #8ab4f8; border-bottom: 2px solid #5a9eff;
                background: #282c34;
            }
            QTabBar::tab:hover { color: #c8ccd4; }
        """)
        layout.addWidget(tabs)

        # ─── Tab 1: 控制 ───
        tab1 = QWidget()
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.setSpacing(16)
        tab1_layout.setContentsMargins(24, 24, 24, 24)

        # 音频源
        audio_section = QLabel("音频来源")
        audio_section.setStyleSheet("font-size: 14px; color: #888; padding: 0;")
        tab1_layout.addWidget(audio_section)

        audio_row = QHBoxLayout()
        self._audio_source_combo = QComboBox()
        self._audio_source_combo.addItems(["系统音频", "麦克风"])
        self._audio_source_combo.setMinimumWidth(160)
        audio_row.addWidget(self._audio_source_combo)

        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(200)
        audio_row.addWidget(self._device_combo, 1)

        self._refresh_dev_btn = QPushButton("⟳")
        self._refresh_dev_btn.setFixedSize(36, 36)
        self._refresh_dev_btn.setToolTip("刷新设备")
        self._refresh_dev_btn.clicked.connect(self._refresh_devices)
        audio_row.addWidget(self._refresh_dev_btn)
        tab1_layout.addLayout(audio_row)

        # 分隔
        sep1 = QLabel()
        sep1.setFixedHeight(1)
        sep1.setStyleSheet("background: #333842;")
        tab1_layout.addWidget(sep1)

        # 开始按钮（最大最显眼）
        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("▶  开始翻译")
        self._start_btn.setObjectName("startBtn")
        self._start_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._start_btn.setMinimumHeight(60)
        self._start_btn.clicked.connect(self._toggle_running)
        btn_row.addWidget(self._start_btn, 3)

        self._stop_btn = QPushButton("⏹  停止")
        self._stop_btn.setObjectName("stopBtn")
        self._stop_btn.setEnabled(False)
        self._stop_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_btn.setMinimumHeight(60)
        self._stop_btn.clicked.connect(self._toggle_running)
        btn_row.addWidget(self._stop_btn, 1)

        tab1_layout.addLayout(btn_row)

        # 状态
        self._status_indicator = QLabel("⚫  就绪")
        self._status_indicator.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #999; background: transparent; padding-top: 8px;"
        )
        tab1_layout.addWidget(self._status_indicator)

        self._stats_label = QLabel("等待开始翻译")
        self._stats_label.setStyleSheet(
            "font-size: 14px; color: #666; background: transparent;"
        )
        tab1_layout.addWidget(self._stats_label)

        tab1_layout.addStretch()
        tabs.addTab(tab1, "控制")

        # ─── Tab 2: 设置 ───
        tab2 = QWidget()
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.setSpacing(16)
        tab2_layout.setContentsMargins(24, 24, 24, 24)

        # 字幕字体
        font_label = QLabel("字幕字体大小")
        font_label.setStyleSheet("font-size: 14px; color: #888;")
        tab2_layout.addWidget(font_label)
        self._font_combo = QComboBox()
        self._font_combo.addItems(["24px", "32px", "40px"])
        self._font_combo.setCurrentIndex(1)
        tab2_layout.addWidget(self._font_combo)

        # 透明度
        op_label = QLabel("字幕透明度")
        op_label.setStyleSheet("font-size: 14px; color: #888;")
        tab2_layout.addWidget(op_label)
        op_row = QHBoxLayout()
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(20, 80)
        self._opacity_slider.setValue(60)
        self._opacity_label = QLabel("60%")
        self._opacity_label.setStyleSheet("font-size: 16px; font-weight: bold; min-width: 40px;")
        self._opacity_slider.valueChanged.connect(
            lambda v: self._opacity_label.setText(f"{v}%")
        )
        op_row.addWidget(self._opacity_slider, 1)
        op_row.addWidget(self._opacity_label)
        tab2_layout.addLayout(op_row)

        # 修正
        sep2 = QLabel()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet("background: #333842;")
        tab2_layout.addWidget(sep2)

        self._auto_correct_cb = QCheckBox("启用自动修正")
        self._auto_correct_cb.setChecked(True)
        tab2_layout.addWidget(self._auto_correct_cb)

        # TTS
        tts_row = QHBoxLayout()
        self._tts_cb = QCheckBox("语音播报")
        self._tts_cb.toggled.connect(self._on_tts_toggled)
        tts_row.addWidget(self._tts_cb)
        tts_row.addSpacing(20)
        self._tts_speed_combo = QComboBox()
        self._tts_speed_combo.addItems(["0.8x", "1.0x", "1.2x", "1.5x"])
        self._tts_speed_combo.setCurrentIndex(1)
        self._tts_speed_combo.setEnabled(False)
        self._tts_speed_combo.setFixedWidth(90)
        tts_row.addWidget(self._tts_speed_combo)
        tts_row.addStretch()
        tab2_layout.addLayout(tts_row)

        tab2_layout.addStretch()
        tabs.addTab(tab2, "设置")

        # ─── 辅助按钮（标签页下方） ───
        aux_row = QHBoxLayout()
        aux_row.setContentsMargins(24, 8, 24, 12)

        self._history_btn = QPushButton("📋 历史")
        self._history_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._history_btn.clicked.connect(self._show_history)
        aux_row.addWidget(self._history_btn)

        self._export_btn = QPushButton("💾 导出")
        self._export_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._export_btn.clicked.connect(self._export)
        aux_row.addWidget(self._export_btn)

        self._subtitle_btn = QPushButton("📺 字幕")
        self._subtitle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self._subtitle_btn.clicked.connect(self._toggle_subtitle)
        aux_row.addWidget(self._subtitle_btn)

        aux_row.addStretch()
        layout.addLayout(aux_row)

    # ─── 设备枚举 ───

    def _refresh_devices(self):
        try:
            from src.audio_capture import AudioCapture
            devices = AudioCapture.list_devices()
            self._device_combo.clear()

            audio_source = self._audio_source_combo.currentText()
            if "系统" in audio_source:
                target_type = "loopback"
            else:
                target_type = "input"

            filtered = [d for d in devices if d.device_type == target_type]
            if not filtered:
                filtered = devices  # fallback

            for d in filtered:
                self._device_combo.addItem(f"[{d.device_type}] {d.name}", d.id)

            logger.info(f"设备列表刷新: {len(filtered)} 个 {target_type} 设备")
        except Exception as e:
            logger.error(f"设备枚举失败: {e}")

    # ─── 启动/停止 ───

    def _toggle_running(self):
        if self._running:
            self._stop_pipeline()
        else:
            self._start_pipeline()

    def _start_pipeline(self):
        """启动翻译管道"""
        from src.pipeline import Pipeline
        from src.audio_capture import AudioCapture
        from src.tts_engine import TTSEngine

        try:
            # 初始化管道 + 翻译协调器
            self._pipeline = Pipeline(model_size="small", device="cpu")
            self._pipeline.start()

            from src.translator_coordinator import TranslationCoordinator
            self._coordinator = TranslationCoordinator()
            self._coordinator.initialize()
            # 复用 pipeline 的 ASR 模型
            self._coordinator.state.set_status(AppStatus.RUNNING)

            # 初始化 TTS（如果勾选）
            if self._tts_cb.isChecked():
                speeds = [0.8, 1.0, 1.2, 1.5]
                spd = speeds[self._tts_speed_combo.currentIndex()]
                self._tts_engine = TTSEngine(speed=spd)
                if self._tts_engine.is_available:
                    self._tts_engine.start()
                    self._coordinator.set_tts(self._tts_engine)
                    logger.info(f"TTS 已启动 (speed={spd}x)")
                else:
                    self._tts_cb.setChecked(False)
                    self._tts_cb.setEnabled(False)
                    self._tts_cb.setText("语音播报 (不可用)")
            else:
                self._tts_engine = None

            logger.info("Pipeline 模型加载完成")

            # 音频捕获
            self._audio_capture = AudioCapture()
            device_id = self._device_combo.currentData()
            self._audio_capture.start(device_id, self._on_audio_chunk)
            logger.info(f"音频捕获已启动 (device={device_id})")

            self._running = True
            self._start_time = time.time()

            # 更新 UI
            self._start_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
            self._status_indicator.setText("🟢  运行中")
            self._status_indicator.setStyleSheet(
                "font-size: 14px; font-weight: bold; color: #4ade80; background: transparent; border: none;"
            )

            self.state.set_status(AppStatus.RUNNING)

            # 定时刷新统计
            self._stats_timer = QTimer()
            self._stats_timer.timeout.connect(self._update_stats)
            self._stats_timer.start(1000)

            logger.info("翻译管道已启动")

        except Exception as e:
            logger.error(f"启动失败: {e}")
            QMessageBox.critical(self, "启动失败", str(e))

    def _on_tts_toggled(self, checked):
        """TTS 开关切换"""
        self._tts_speed_combo.setEnabled(checked)
        if not checked and hasattr(self, '_tts_engine') and self._tts_engine:
            self._tts_engine.stop()
            self._tts_engine = None

    def _stop_pipeline(self):
        """停止翻译管道"""
        self._running = False

        if self._audio_capture:
            self._audio_capture.stop()
        if hasattr(self, '_stats_timer'):
            self._stats_timer.stop()

        # 停止 TTS
        if hasattr(self, '_tts_engine') and self._tts_engine:
            self._tts_engine.stop()
            self._tts_engine = None

        # Flush 剩余结果
        if self._pipeline:
            self._pipeline.finish()

        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_indicator.setText("⚫  已停止")
        self._status_indicator.setStyleSheet(
            "font-size: 14px; font-weight: bold; color: #999; background: transparent; border: none;"
        )

        self.state.set_status(AppStatus.STOPPED)
        logger.info("翻译管道已停止")

    def _on_audio_chunk(self, audio_chunk: np.ndarray):
        """音频回调 → VAD → ASR → MT → TTS"""
        if not self._running or self._pipeline is None:
            return
        try:
            asr_result = self._pipeline.feed(audio_chunk)
            if asr_result is not None and hasattr(self, '_coordinator'):
                self._coordinator.process(asr_result)
        except Exception as e:
            logger.error(f"处理音频块失败: {e}")

    def _update_stats(self):
        """更新统计显示"""
        elapsed = time.time() - self._start_time if self._start_time else 0
        m, s = int(elapsed // 60), int(elapsed % 60)
        stats = self.state
        self._stats_label.setText(
            f"已翻译 {stats.translated_count} 句 · "
            f"修正 {stats.correction_count} 次 · "
            f"延迟 {stats.avg_latency_ms:.0f}ms · "
            f"运行 {m:02d}:{s:02d}"
        )

    # ─── 历史记录 ───

    def _show_history(self):
        dialog = HistoryDialog(self)
        dialog.exec()

    # ─── 导出 ───

    def _export(self):
        from src.export_utils import export_txt, export_srt

        records = self.state.history
        if not records:
            QMessageBox.information(self, "提示", "暂无翻译记录可导出")
            return

        txt_path = export_txt(records)
        srt_path = export_srt(records)
        QMessageBox.information(
            self, "导出完成",
            f"已导出:\n{txt_path}\n{srt_path}"
        )
        logger.info(f"已导出: {txt_path}, {srt_path}")

    # ─── 字幕窗口 ───

    def _toggle_subtitle(self):
        if not hasattr(self, '_subtitle_window') or self._subtitle_window is None:
            self._subtitle_window = SubtitleWindow()
            self._subtitle_window.show()
            self._subtitle_btn.setText("📺 隐藏")
        else:
            if self._subtitle_window.isVisible():
                self._subtitle_window.hide()
                self._subtitle_btn.setText("📺 字幕")
            else:
                self._subtitle_window.show()
                self._subtitle_btn.setText("📺 隐藏")

    # ─── 系统托盘 ───

    def _setup_tray(self):
        self._tray = QSystemTrayIcon(self)
        self._tray.setToolTip("AI 同声传译助手")

        tray_menu = QMenu()
        tray_menu.addAction("显示主面板", self.show)
        tray_menu.addAction("显示/隐藏字幕", self._toggle_subtitle)
        tray_menu.addSeparator()
        tray_menu.addAction("退出", QApplication.quit)
        self._tray.setContextMenu(tray_menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show()
            self.raise_()

    def closeEvent(self, event):
        """关闭时保存配置并最小化到托盘"""
        self._save_config()
        event.ignore()
        self.hide()
        self._tray.showMessage(
            "AI 同声传译助手", "已最小化到系统托盘", QSystemTrayIcon.MessageIcon.Information, 2000
        )

    # ─── 配置 ───

    def _load_config(self):
        cfg = self.config_mgr.data
        font_map = {24: 0, 32: 1, 40: 2}
        self._font_combo.setCurrentIndex(font_map.get(cfg.get("font_size", 24), 1))
        self._opacity_slider.setValue(int(cfg.get("opacity", 0.6) * 100))
        self._auto_correct_cb.setChecked(cfg.get("auto_correct", True))

    def _save_config(self):
        font_sizes = [24, 32, 40]
        self.config_mgr.update({
            "font_size": font_sizes[self._font_combo.currentIndex()],
            "opacity": self._opacity_slider.value() / 100.0,
            "auto_correct": self._auto_correct_cb.isChecked(),
            "audio_source": "system" if "系统" in self._audio_source_combo.currentText() else "microphone",
        })
        self.config_mgr.save()
