"""
字幕悬浮窗口 — SubtitleWindow
基于 PyQt6：半透明、置顶、无边框、可拖拽
"""
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QApplication, QMenu
from PyQt6.QtCore import Qt, QPoint, QTimer, pyqtSignal
from PyQt6.QtGui import QFont, QAction, QMouseEvent

from src.state_manager import StateManager, AppStatus


class SubtitleWindow(QWidget):
    """悬浮字幕窗口"""

    # 信号：字幕文本变化
    subtitle_changed = pyqtSignal(str)

    def __init__(self):
        super().__init__()

        self.state = StateManager()

        # ─── 窗口属性 ───
        self.setWindowTitle("AI 同传字幕")
        self.setWindowFlags(
            Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)

        # 默认大小和位置
        screen = QApplication.primaryScreen().availableGeometry()
        self._default_w = int(screen.width() * 0.6)
        self._default_h = 120
        self.resize(self._default_w, self._default_h)

        # 底部居中
        x = (screen.width() - self._default_w) // 2
        y = screen.height() - self._default_h - 60
        self.move(x, y)

        # ─── UI ───
        self._label = QLabel("")
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._label.setWordWrap(True)
        self._label.setFont(QFont("Microsoft YaHei", 24))
        self._label.setStyleSheet("color: #FFFFFF; padding: 10px;")

        layout = QVBoxLayout()
        layout.setContentsMargins(20, 10, 20, 10)
        layout.addWidget(self._label)
        self.setLayout(layout)

        # 背景样式
        self._opacity = 0.6
        self._update_background()

        # ─── 拖拽状态 ───
        self._dragging = False
        self._drag_start_pos = QPoint()

        # ─── 定时刷新字幕 ───
        self._timer = QTimer()
        self._timer.timeout.connect(self._refresh_subtitle)
        self._timer.start(100)  # 每 100ms 检查

        # 上次版本号
        self._last_version = 0

        # ─── 右键菜单 ───
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_menu)

    # ─── 背景 ───

    def _update_background(self):
        """更新半透明背景"""
        alpha = int(self._opacity * 255)
        self.setStyleSheet(f"""
            SubtitleWindow {{
                background-color: rgba(0, 0, 0, {alpha});
                border-radius: 12px;
            }}
        """)

    # ─── 字幕刷新 ───

    def _refresh_subtitle(self):
        """定时检查并更新字幕"""
        if self.state.subtitle_version != self._last_version:
            self._last_version = self.state.subtitle_version
            text = self.state.current_subtitle
            self._label.setText(text)
            self.subtitle_changed.emit(text)

    # ─── 拖拽 ───

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = True
            self._drag_start_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._dragging:
            self.move(event.globalPosition().toPoint() - self._drag_start_pos)
            event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            event.accept()

    # ─── 右键菜单 ───

    def _show_menu(self, pos):
        menu = QMenu(self)

        # 字体大小
        font_menu = menu.addMenu("字体大小")
        for size in [18, 24, 32]:
            action = QAction(f"{size}px", self)
            action.triggered.connect(lambda checked, s=size: self.set_font_size(s))
            font_menu.addAction(action)

        # 透明度
        opacity_menu = menu.addMenu("透明度")
        for op in [0.2, 0.4, 0.6, 0.8]:
            action = QAction(f"{int(op*100)}%", self)
            action.triggered.connect(lambda checked, o=op: self.set_opacity(o))
            opacity_menu.addAction(action)

        menu.addSeparator()
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(QApplication.quit)
        menu.addAction(exit_action)

        menu.exec(self.mapToGlobal(pos))

    # ─── 公开接口 ───

    def set_font_size(self, size: int):
        """设置字体大小"""
        self._label.setFont(QFont("Microsoft YaHei", size))

    def set_opacity(self, opacity: float):
        """设置背景透明度 (0.0~1.0)"""
        self._opacity = max(0.1, min(1.0, opacity))
        self._update_background()

    def set_subtitle_direct(self, text: str):
        """直接设置字幕文本（不经过 StateManager）"""
        self._label.setText(text)

    def show(self):
        """显示窗口"""
        super().show()

    def close(self):
        """关闭"""
        self._timer.stop()
        super().close()
