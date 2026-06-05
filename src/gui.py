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
        QMessageBox, QHeaderView, QGroupBox, QGridLayout,
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
        QMessageBox, QHeaderView, QGroupBox, QGridLayout, QAction,
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


# ─── 字幕悬浮窗口 (保持原有设计，增加修正动画) ───

class SubtitleWindow(QWidget):
    """半透明、置顶、可拖拽、无边框字幕窗口"""

    subtitle_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.state = StateManager()
        self._opacity = 0.6
        self._font_size = 24
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
        w, h = 800, 120
        self.resize(w, h)
        self.move((screen.width() - w) // 2, screen.height() - h - 60)

        self._label = QLabel("")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setFont(QFont("Microsoft YaHei", self._font_size))
        self._label.setStyleSheet("color: #FFFFFF; padding: 10px;")

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
        for size in [18, 24, 32]:
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
        self.setFixedSize(420, 480)

        self._setup_ui()
        self._load_config()
        self._setup_tray()

        # 启动时刷新设备列表
        self._refresh_devices()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)

        # ── 音频源 ──
        audio_group = QGroupBox("音频设置")
        audio_layout = QGridLayout(audio_group)

        audio_layout.addWidget(QLabel("音频源:"), 0, 0)
        self._audio_source_combo = QComboBox()
        self._audio_source_combo.addItems(["系统音频 (Loopback)", "麦克风"])
        audio_layout.addWidget(self._audio_source_combo, 0, 1)

        audio_layout.addWidget(QLabel("设备:"), 1, 0)
        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(250)
        audio_layout.addWidget(self._device_combo, 1, 1)

        self._refresh_dev_btn = QPushButton("⟳")
        self._refresh_dev_btn.setFixedWidth(35)
        self._refresh_dev_btn.clicked.connect(self._refresh_devices)
        audio_layout.addWidget(self._refresh_dev_btn, 1, 2)

        layout.addWidget(audio_group)

        # ── 字幕设置 ──
        subtitle_group = QGroupBox("字幕设置")
        sub_layout = QGridLayout(subtitle_group)

        sub_layout.addWidget(QLabel("字体:"), 0, 0)
        font_widget = QWidget()
        font_h = QHBoxLayout(font_widget)
        font_h.setContentsMargins(0, 0, 0, 0)
        self._font_combo = QComboBox()
        self._font_combo.addItems(["小 (18px)", "中 (24px)", "大 (32px)"])
        self._font_combo.setCurrentIndex(1)
        font_h.addWidget(self._font_combo)
        sub_layout.addWidget(font_widget, 0, 1)

        sub_layout.addWidget(QLabel("透明度:"), 1, 0)
        self._opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self._opacity_slider.setRange(20, 80)
        self._opacity_slider.setValue(60)
        self._opacity_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self._opacity_slider.setTickInterval(20)
        self._opacity_label = QLabel("60%")
        self._opacity_slider.valueChanged.connect(
            lambda v: self._opacity_label.setText(f"{v}%")
        )
        sub_layout.addWidget(self._opacity_slider, 1, 1)
        sub_layout.addWidget(self._opacity_label, 1, 2)

        layout.addWidget(subtitle_group)

        # ── 功能选项 ──
        options_group = QGroupBox("功能选项")
        opt_layout = QVBoxLayout(options_group)
        self._auto_correct_cb = QCheckBox("自动修正 (Reshoot + 上下文修正)")
        self._auto_correct_cb.setChecked(True)
        opt_layout.addWidget(self._auto_correct_cb)

        tts_widget = QWidget()
        tts_h = QHBoxLayout(tts_widget)
        tts_h.setContentsMargins(0, 0, 0, 0)
        self._tts_cb = QCheckBox("语音播报 (TTS)")
        self._tts_cb.setChecked(False)
        self._tts_cb.toggled.connect(self._on_tts_toggled)
        tts_h.addWidget(self._tts_cb)
        tts_h.addWidget(QLabel("语速:"))
        self._tts_speed_combo = QComboBox()
        self._tts_speed_combo.addItems(["0.8x", "1.0x", "1.2x", "1.5x"])
        self._tts_speed_combo.setCurrentIndex(1)
        self._tts_speed_combo.setEnabled(False)
        tts_h.addWidget(self._tts_speed_combo)
        tts_h.addStretch()
        opt_layout.addWidget(tts_widget)
        layout.addWidget(options_group)

        # ── 控制按钮 ──
        btn_layout = QHBoxLayout()
        self._start_btn = QPushButton("▶ 开始翻译")
        self._start_btn.setStyleSheet(
            "QPushButton { background-color: #4CAF50; color: white; font-size: 14px; "
            "padding: 8px 20px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #45a049; }"
        )
        self._start_btn.clicked.connect(self._toggle_running)
        btn_layout.addWidget(self._start_btn)

        self._stop_btn = QPushButton("⏹ 停止")
        self._stop_btn.setEnabled(False)
        self._stop_btn.clicked.connect(self._toggle_running)
        btn_layout.addWidget(self._stop_btn)
        layout.addLayout(btn_layout)

        # ── 状态栏 ──
        status_group = QGroupBox("运行状态")
        status_layout = QGridLayout(status_group)
        self._status_indicator = QLabel("⚫ 未启动")
        status_layout.addWidget(self._status_indicator, 0, 0)

        self._stats_label = QLabel("已翻译 0 句 · 修正 0 次 · 延迟 --ms")
        status_layout.addWidget(self._stats_label, 1, 0)
        layout.addWidget(status_group)

        # ── 辅助按钮 ──
        aux_layout = QHBoxLayout()
        self._history_btn = QPushButton("📋 历史记录")
        self._history_btn.clicked.connect(self._show_history)
        aux_layout.addWidget(self._history_btn)

        self._export_btn = QPushButton("💾 导出")
        self._export_btn.clicked.connect(self._export)
        aux_layout.addWidget(self._export_btn)

        self._subtitle_btn = QPushButton("📺 显示/隐藏字幕")
        self._subtitle_btn.clicked.connect(self._toggle_subtitle)
        aux_layout.addWidget(self._subtitle_btn)
        layout.addLayout(aux_layout)

        # ── 底部 ──
        layout.addStretch()
        layout.addWidget(QLabel("v1.0 · 完全离线 · MIT"))

    # ─── 设备枚举 ───

    def _refresh_devices(self):
        try:
            from src.audio_capture import AudioCapture
            devices = AudioCapture.list_devices()
            self._device_combo.clear()

            audio_source = self._audio_source_combo.currentText()
            if "Loopback" in audio_source or "系统" in audio_source:
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
            self._status_indicator.setText("🟢 运行中")
            self._status_indicator.setStyleSheet("color: green; font-weight: bold;")

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
        self._status_indicator.setText("⚫ 已停止")
        self._status_indicator.setStyleSheet("")

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
            self._subtitle_btn.setText("📺 隐藏字幕")
        else:
            if self._subtitle_window.isVisible():
                self._subtitle_window.hide()
                self._subtitle_btn.setText("📺 显示字幕")
            else:
                self._subtitle_window.show()
                self._subtitle_btn.setText("📺 隐藏字幕")

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
        font_map = {18: 0, 24: 1, 32: 2}
        self._font_combo.setCurrentIndex(font_map.get(cfg.get("font_size", 24), 1))
        self._opacity_slider.setValue(int(cfg.get("opacity", 0.6) * 100))
        self._auto_correct_cb.setChecked(cfg.get("auto_correct", True))

    def _save_config(self):
        font_sizes = [18, 24, 32]
        self.config_mgr.update({
            "font_size": font_sizes[self._font_combo.currentIndex()],
            "opacity": self._opacity_slider.value() / 100.0,
            "auto_correct": self._auto_correct_cb.isChecked(),
            "audio_source": "system" if "Loopback" in self._audio_source_combo.currentText() else "microphone",
        })
        self.config_mgr.save()
