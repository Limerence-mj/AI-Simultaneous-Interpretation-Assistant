"""
AI 同声传译助手 — 完整 GUI
主控制面板 + 悬浮字幕窗口 + 历史记录 + 系统托盘
兼容 PyQt5 / PyQt6
"""
import sys
import time
import threading
from pathlib import Path
from typing import Callable

import numpy as np

# PyQt5/PyQt6 兼容导入
_PYQT_VERSION = 0
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QComboBox, QCheckBox, QSlider, QDialog,
        QTableWidget, QTableWidgetItem, QSystemTrayIcon, QMenu,
        QMessageBox, QHeaderView, QGroupBox, QGridLayout, QTabWidget,
        QScrollArea,
    )
    from PyQt6.QtCore import Qt, QTimer, QPoint, pyqtSignal, QThread
    from PyQt6.QtGui import QFont, QAction, QIcon, QColor, QMouseEvent
    _PYQT_VERSION = 6
    _HeaderResizeMode = QHeaderView.ResizeMode.Interactive
    def _global_pos(event): return event.globalPosition().toPoint()
    _WA_TranslucentBackground = Qt.WidgetAttribute.WA_TranslucentBackground
    _WA_ShowWithoutActivating = Qt.WidgetAttribute.WA_ShowWithoutActivating
except ImportError:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QComboBox, QCheckBox, QSlider, QDialog,
        QTableWidget, QTableWidgetItem, QSystemTrayIcon, QMenu,
        QMessageBox, QHeaderView, QGroupBox, QGridLayout, QTabWidget, QAction,
        QScrollArea,
    )
    from PyQt5.QtCore import Qt, QTimer, QPoint, pyqtSignal, QThread
    from PyQt5.QtGui import QFont, QIcon, QColor, QMouseEvent
    _PYQT_VERSION = 5
    _HeaderResizeMode = QHeaderView.Interactive
    def _global_pos(event): return event.globalPos()
    _WA_TranslucentBackground = Qt.WA_TranslucentBackground
    _WA_ShowWithoutActivating = Qt.WA_ShowWithoutActivating

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
    # 默认提示文字
    DEFAULT_TEXT = "🎙️ AI 同声传译助手 — 就绪"

    def __init__(self):
        super().__init__()
        self.state = StateManager()
        self._opacity = 0.6
        self._font_size = 32
        self._last_version = 0
        self._last_text = ""
        self._dragging = False
        self._drag_start_pos = QPoint()
        self._flash_timer_id = None  # 修正闪烁定时器 ID

        self._setup_ui()
        self._setup_timer()

    def _setup_ui(self):
        self.setWindowTitle("AI 同传字幕")
        self.setWindowFlags(Qt.WindowType.WindowStaysOnTopHint|Qt.WindowType.FramelessWindowHint|Qt.WindowType.Tool)
        self.setAttribute(_WA_ShowWithoutActivating, True)
        self.setAttribute(_WA_TranslucentBackground, True)
        screen = QApplication.primaryScreen().availableGeometry()
        w, h = 1100, 320; self.resize(w, h)
        self.move((screen.width()-w)//2, screen.height()-h-100)
        self._en_label = QLabel("")
        self._en_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self._en_label.setWordWrap(True)
        self._en_label.setFont(QFont("Microsoft YaHei", int(self._font_size*0.75)))
        self._en_label.setStyleSheet("color: rgba(255,220,50,200); padding: 4px 20px; background: transparent;")
        self._zh_label = QLabel(self.DEFAULT_TEXT)
        self._zh_label.setAlignment(Qt.AlignmentFlag.AlignCenter); self._zh_label.setWordWrap(True)
        self._zh_label.setFont(QFont("Microsoft YaHei", self._font_size))
        self._zh_label.setStyleSheet("color: rgba(255,220,50,255); padding: 4px 20px; background: transparent;")
        layout = QVBoxLayout(); layout.setContentsMargins(20,10,20,10)
        layout.addWidget(self._en_label); layout.addWidget(self._zh_label); self.setLayout(layout)
        self._apply_background(self._opacity)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)

    def _setup_timer(self):
        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)   # 30fps 刷新

    def _refresh(self):
        if self.state.subtitle_version != self._last_version:
            self._last_version = self.state.subtitle_version
            zh = self.state.current_subtitle or ""
            en = self.state.current_en_text or ""
            if not zh: zh = self.DEFAULT_TEXT
            is_correction = bool(self._last_text and zh and self._last_text != zh and self._last_text != self.DEFAULT_TEXT)
            self._en_label.setText(en); self._zh_label.setText(zh)
            self.subtitle_changed.emit(zh); self._last_text = zh
            if is_correction: self._flash_correction()

    def _flash_correction(self):
        self._apply_background(self._opacity, flash=True)
        if self._flash_timer_id is not None: self.killTimer(self._flash_timer_id)
        self._flash_timer_id = self.startTimer(800)

    def timerEvent(self, event):
        if self._flash_timer_id is not None and event.timerId() == self._flash_timer_id:
            self.killTimer(self._flash_timer_id); self._flash_timer_id = None
            self._flash = False; self._update_label_style()
        else: super().timerEvent(event)

    def _apply_background(self, opacity: float, flash: bool = False):
        self._text_opacity = max(0.1, min(1.0, opacity))
        self._flash = flash; self._update_label_style()

    def _update_label_style(self):
        alpha = int(getattr(self, '_text_opacity', 0.6)*255)
        flash = getattr(self, '_flash', False)
        if flash:
            zh_c = f"rgba(255,240,80,{min(alpha+40,255)})"; en_c = f"rgba(255,240,80,{min(alpha+40,255)})"
        else:
            zh_c = f"rgba(255,220,50,{alpha})"; en_c = f"rgba(255,220,50,{alpha})"
        self._zh_label.setStyleSheet(f"color: {zh_c}; padding: 4px 20px; background: transparent; font-size: {self._font_size}px;")
        self._en_label.setStyleSheet(f"color: {en_c}; padding: 4px 20px; background: transparent; font-size: {int(self._font_size*0.75)}px;")

    def _update_background(self): self._update_label_style()

    def _show_menu(self, pos):
        menu = QMenu(self)
        menu.setStyleSheet("QMenu { background-color: #2c313a; color: #e0e0e0; border: 1px solid #3e4452; } QMenu::item:selected { background-color: #3a6fc5; }")
        fm = menu.addMenu("字体大小")
        for size in [24,32,40]:
            a = QAction(f"{size}px", self); a.setData(size); a.triggered.connect(self._on_font_action); fm.addAction(a)
        om = menu.addMenu("透明度")
        for op in [0.2,0.4,0.6,0.8]:
            a = QAction(f"{int(op*100)}%", self); a.setData(op); a.triggered.connect(self._on_opacity_action); om.addAction(a)
        menu.addSeparator(); menu.addAction(QAction("退出", self, triggered=QApplication.quit))
        menu.exec(self.mapToGlobal(pos))

    def _on_font_action(self):
        s = self.sender().data()
        if s: self.set_font_size(s)
    def _on_opacity_action(self):
        o = self.sender().data()
        if o: self.set_opacity(o)

    def set_font_size(self, size: int):
        self._font_size = size; self._update_label_style()
    def set_opacity(self, opacity: float):
        self._opacity = max(0.1, min(1.0, opacity)); self._apply_background(self._opacity)
    def set_subtitle_direct(self, text: str):
        self._zh_label.setText(text)

    # ─── 鼠标拖拽（兼容 PyQt5/PyQt6） ───

    def mousePressEvent(self, e: QMouseEvent):
        if e.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start_pos = _global_pos(e) - self.frameGeometry().topLeft()

    def mouseMoveEvent(self, e: QMouseEvent):
        if self._dragging:
            # 使用兼容的 _global_pos 而不是直接调用 PyQt6 的 globalPosition()
            self.move(_global_pos(e) - self._drag_start_pos)

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


# ─── 后台初始化线程 ───

class PipelineInitWorker(QThread):
    """后台线程：加载模型和初始化管道，避免阻塞 UI"""
    progress_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, on_subtitle: Callable = None):
        super().__init__()
        self._on_subtitle_cb = on_subtitle
        self.pipeline = None
        self.coordinator = None

    def run(self):
        try:
            from src.pipeline import Pipeline
            from src.translator_coordinator import TranslationCoordinator
            self.progress_signal.emit("正在加载语音识别模型 (Whisper)...")
            self.pipeline = Pipeline(model_size="tiny", device="cpu",
                                     on_result=self._on_subtitle_cb)
            self.pipeline.start()
            self.progress_signal.emit("正在加载翻译模型 (opus-mt)...")
            self.coordinator = TranslationCoordinator()
            self.coordinator.initialize()
            self.finished_signal.emit(True, "")
        except Exception as e:
            import traceback; traceback.print_exc()
            self.finished_signal.emit(False, str(e))


# ─── 主控制面板 ───

class MainWindow(QMainWindow):
    """主控制面板"""

    status_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.state = StateManager()
        self.config_mgr = ConfigManager()
        self.config_mgr.load()

        self._pipeline = None       # StreamingPipeline 实例
        self._audio_capture = None  # AudioCapture 实例
        self._tts_engine = None     # TTS 引擎
        self._running = False
        self._start_time = 0.0
        self._init_thread = None    # 后台初始化线程

        self.setWindowTitle("🎙️ AI 同声传译助手")
        self.setMinimumSize(600, 750)
        self.resize(620, 850)
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
        self._audio_source_combo.activated.connect(self._refresh_devices)
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

        # ─── Tab 2: 设置（可滚动） ───
        scroll2 = QScrollArea()
        scroll2.setWidgetResizable(True)
        scroll2.setStyleSheet("QScrollArea { border: none; background: #1a1d23; }")
        tab2 = QWidget()
        tab2.setStyleSheet("background: #1a1d23;")
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.setSpacing(16)
        tab2_layout.setContentsMargins(24, 24, 24, 24)

        # ── 字幕字体 ──
        font_label = QLabel("字幕字体大小"); font_label.setStyleSheet("font-size: 14px; color: #888;")
        tab2_layout.addWidget(font_label)
        self._font_combo = QComboBox(); self._font_combo.addItems(["24px","32px","40px"]); self._font_combo.setCurrentIndex(1)
        self._font_combo.currentIndexChanged.connect(self._on_font_combo_changed)
        tab2_layout.addWidget(self._font_combo)

        # ── 透明度 ──
        op_label = QLabel("字幕透明度"); op_label.setStyleSheet("font-size: 14px; color: #888;")
        tab2_layout.addWidget(op_label)
        op_row = QHBoxLayout()
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal); self._opacity_slider.setRange(20,80); self._opacity_slider.setValue(60)
        self._opacity_label = QLabel("60%"); self._opacity_label.setStyleSheet("font-size: 16px; font-weight: bold; min-width: 40px;")
        self._opacity_slider.valueChanged.connect(self._on_opacity_slider_changed)
        op_row.addWidget(self._opacity_slider,1); op_row.addWidget(self._opacity_label)
        tab2_layout.addLayout(op_row)

        # ── 修正 ──
        sep2 = QLabel(); sep2.setFixedHeight(1); sep2.setStyleSheet("background: #333842;")
        tab2_layout.addWidget(sep2)
        self._auto_correct_cb = QCheckBox("启用自动修正"); self._auto_correct_cb.setChecked(True)
        tab2_layout.addWidget(self._auto_correct_cb)
        tab2_layout.addStretch()
        scroll2.setWidget(tab2)
        tabs.addTab(scroll2, "设置")

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
                # 用中文标签区分设备类型
                if d.device_type == "loopback":
                    label = f"🔊 系统音频 — {d.name}"
                else:
                    label = f"🎤 麦克风 — {d.name}"
                self._device_combo.addItem(label, d.id)

            logger.info(f"设备列表刷新: {len(filtered)} 个 {target_type} 设备")
        except Exception as e:
            logger.error(f"设备枚举失败: {e}")

    # ─── 启动/停止 ───

    def _toggle_running(self):
        if self._running:
            self._stop_pipeline()
        elif self._init_thread is None:
            self._start_pipeline()

    def _start_pipeline(self):
        """启动翻译管道"""
        from src.audio_capture import AudioCapture

        self._start_btn.setEnabled(False); self._stop_btn.setEnabled(False)
        self._status_indicator.setText("🟡  正在加载模型...")
        self._status_indicator.setStyleSheet("font-size: 18px; font-weight: bold; color: #facc15; background: transparent; padding-top: 8px;")
        self._init_thread = PipelineInitWorker(on_subtitle=self._on_subtitle_update)
        self._init_thread.progress_signal.connect(self._on_init_progress)
        self._init_thread.finished_signal.connect(self._on_init_finished)
        self._init_thread.start()

    def _on_init_progress(self, msg: str):
        """后台线程进度回调"""
        logger.info(msg)
        self._status_indicator.setText(f"🟡  {msg}")

    def _on_init_finished(self, success: bool, error_msg: str):
        """后台线程完成回调（在主线程执行）"""
        if not success:
            self._init_thread = None
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._status_indicator.setText("❌  启动失败")
            self._status_indicator.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #f87171; background: transparent; padding-top: 8px;"
            )
            QMessageBox.critical(self, "启动失败", error_msg)
            return

        self._pipeline = self._init_thread.pipeline
        self._coordinator = self._init_thread.coordinator
        self._tts_engine = None  # TTS 扩展接口暂不启用
        self._init_thread = None
        # 启动音频捕获
        from src.audio_capture import AudioCapture
        try:
            self._audio_capture = AudioCapture()
            device_id = self._device_combo.currentData()
            if device_id is None:
                # 没有设备选中时，尝试默认设备
                devices = AudioCapture.list_devices()
                loopback_devices = [d for d in devices if d.device_type == "loopback"]
                if loopback_devices:
                    device_id = loopback_devices[0].id
                elif devices:
                    device_id = devices[0].id
                else:
                    raise RuntimeError("未检测到可用音频设备")

            self._audio_capture.start(device_id, self._on_audio_chunk)
            logger.info(f"音频捕获已启动 (device={device_id})")
        except Exception as e:
            logger.error(f"音频启动失败: {e}")
            self._pipeline = None
            self._tts_engine = None
            self._start_btn.setEnabled(True)
            self._stop_btn.setEnabled(False)
            self._status_indicator.setText("❌  音频设备错误")
            self._status_indicator.setStyleSheet(
                "font-size: 18px; font-weight: bold; color: #f87171; background: transparent; padding-top: 8px;"
            )
            QMessageBox.critical(self, "音频设备错误", str(e))
            return

        self._running = True
        self._start_time = time.time()
        self.state.set_status(AppStatus.RUNNING)

        # 更新 UI
        self._start_btn.setEnabled(False)
        self._stop_btn.setEnabled(True)
        self._status_indicator.setText("🟢  运行中")
        self._status_indicator.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #4ade80; background: transparent; padding-top: 8px;"
        )

        # 定时刷新统计
        self._stats_timer = QTimer()
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start(1000)

        logger.info("翻译管道已启动")

    def _on_font_combo_changed(self, idx):
        sizes = [24, 32, 40]
        if 0 <= idx < len(sizes):
            w = getattr(self, '_subtitle_window', None)
            if w is not None: w.set_font_size(sizes[idx])

    def _on_opacity_slider_changed(self, value):
        self._opacity_label.setText(f"{value}%")
        w = getattr(self, '_subtitle_window', None)
        if w is not None: w.set_opacity(value / 100.0)

    def _stop_pipeline(self):
        """停止翻译管道"""
        self._running = False

        if hasattr(self, '_stats_timer'):
            try:
                self._stats_timer.stop()
            except Exception:
                pass

        if self._audio_capture:
            try:
                self._audio_capture.stop()
            except Exception:
                pass
            self._audio_capture = None

        if self._pipeline:
            try:
                self._pipeline.finish()
            except Exception:
                pass
            self._pipeline = None

        self._coordinator = None

        self._start_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status_indicator.setText("⚫  已停止")
        self._status_indicator.setStyleSheet(
            "font-size: 18px; font-weight: bold; color: #999; background: transparent; padding-top: 8px;"
        )

        self.state.set_status(AppStatus.STOPPED)
        logger.info("翻译管道已停止")

    def _on_audio_chunk(self, audio_chunk: np.ndarray):
        """音频回调 → VAD → ASR 队列 → 翻译"""
        if not self._running or self._pipeline is None:
            return
        try:
            # 周期性能量日志
            if not hasattr(self, '_chunk_count'):
                self._chunk_count = 0
                self._last_report = time.time()
            self._chunk_count += 1
            if time.time() - self._last_report >= 2.0:
                energy = float(np.sqrt(np.mean(audio_chunk ** 2)))
                logger.info(f"[音频] {self._chunk_count}块, 能量={energy:.4f}")
                self._chunk_count = 0
                self._last_report = time.time()

            self._pipeline.feed(audio_chunk)
        except Exception as e:
            logger.error(f"音频处理失败: {e}")

    def _on_subtitle_update(self, result):
        """字幕回调（兼容 API 模式字符串 和 本地模式 ASRResult）"""
        if not self._running:
            return
        if isinstance(result, str):
            self.state.set_subtitle(result)
            return
        # 本地模式：收到 ASRResult
        if self._coordinator is None:
            return
        try:
            logger.info(f"[翻译] EN: {result.en_text[:80]}")
            mt_result = self._coordinator.process(result)
            logger.info(f"[翻译] ZH: {mt_result.zh_text[:80]}")
        except Exception as e:
            logger.error(f"翻译处理失败: {e}")

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
            sizes = [24, 32, 40]; idx = self._font_combo.currentIndex()
            if 0 <= idx < len(sizes): self._subtitle_window.set_font_size(sizes[idx])
            self._subtitle_window.set_opacity(self._opacity_slider.value() / 100.0)
            self._subtitle_window.show(); self._subtitle_btn.setText("📺 隐藏")
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
        tray_menu.addAction("退出", self._quit_app)
        self._tray.setContextMenu(tray_menu)
        self._tray.activated.connect(self._on_tray_activated)
        self._tray.show()

    def _on_tray_activated(self, reason):
        if reason == QSystemTrayIcon.ActivationReason.DoubleClick:
            self.show()
            self.raise_()

    def _quit_app(self):
        """彻底退出程序"""
        # 停止翻译管道
        if self._running:
            self._stop_pipeline()
        # 保存配置
        self._save_config()
        # 关闭字幕窗口
        if hasattr(self, '_subtitle_window') and self._subtitle_window:
            try:
                self._subtitle_window.close()
            except Exception:
                pass
            self._subtitle_window = None
        # 隐藏托盘图标
        if hasattr(self, '_tray'):
            try:
                self._tray.hide()
            except Exception:
                pass
        QApplication.quit()

    def closeEvent(self, event):
        """关闭主窗口 → 彻底退出程序"""
        self._quit_app()

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
