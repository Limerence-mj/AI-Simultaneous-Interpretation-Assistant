# AI 同声传译助手

**AI Simultaneous Interpretation Assistant**

一款完全离线运行的桌面端AI同声传译助手，能够实时捕获系统音频或麦克风输入，将英文语音流逐句识别并翻译成中文，以可拖拽、半透明悬浮字幕形式呈现在屏幕前端，并具备基于上下文的自动修正能力。

---

## ✨ 核心特性

- **完全离线**：所有模型本地运行，零云端依赖，保护隐私
- **实时翻译**：端到端延迟 GPU < 3s / CPU < 6s
- **上下文自修正**：随上下文积累自动纠正代词、名词翻译错误
- **悬浮字幕**：半透明、置顶、可拖拽、跨显示器
- **语音播报**（v2.0）：可选 TTS 语音朗读翻译结果
- **字幕导出**：支持 TXT / SRT 格式

---

## 📋 系统要求

| 项目 | 最低要求 | 推荐配置 |
|------|----------|----------|
| 操作系统 | Windows 10 1809+ / macOS 12.0+ | Windows 11 / macOS 14+ |
| CPU | Intel i5 8代+ / AMD Ryzen 5+ | Intel i7 10代+ |
| 内存 | 8 GB | 16 GB |
| GPU（可选） | NVIDIA GTX 1060 6GB+ | NVIDIA RTX 3060+ |
| Python | 3.10+ | 3.11+ |
| 磁盘空间 | ~4 GB（含模型） | ~5 GB |

---

## 🚀 快速开始

### 1. 下载项目

```bash
git clone https://github.com/Limerence-mj/AI-Simultaneous-Interpretation-Assistant.git
cd AI-Simultaneous-Interpretation-Assistant
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 下载模型

```bash
python scripts/download_models.py
```

模型文件将存放于 `models/` 目录（约 3GB）。

### 4. 运行

```bash
python src/main.py
```

---

## 📦 依赖说明

### 第三方库

| 库 | 版本 | 用途 | 许可证 | 是否原创 |
|----|------|------|--------|----------|
| faster-whisper | >=0.10.0 | 语音识别引擎（Whisper CTranslate2） | MIT | 否 |
| transformers | >=4.30.0 | 翻译模型加载与推理 | Apache 2.0 | 否 |
| torch | >=2.0.0 | 深度学习推理框架 | BSD | 否 |
| sentencepiece | >=0.1.99 | 翻译模型分词器 | Apache 2.0 | 否 |
| sounddevice | >=0.4.6 | 跨平台音频捕获 | MIT | 否 |
| numpy | >=1.24.0 | 音频数据处理 | BSD | 否 |
| silero-vad | >=5.0 | 语音活动检测 | MIT | 否 |
| pydub | >=0.25.0 | 音频文件格式处理 | MIT | 否 |
| PyQt6 | >=6.5.0 | 图形用户界面 | GPL | 否 |
| pyttsx3 | >=2.90.0 | 离线语音合成（v2.0） | MPL 2.0 | 否 |
| pyinstaller | >=5.13.0 | 应用打包分发 | GPL | 否 |
| psutil | >=5.9.0 | 系统资源监控 | BSD | 否 |

### 模型文件

| 模型 | 大小 | 用途 | 来源 | 许可证 |
|------|------|------|------|--------|
| faster-whisper-small | ~1.8 GB | 英文语音识别 | Systran/faster-whisper-small | MIT |
| faster-whisper-tiny | ~700 MB | 语音识别（降级备选） | Systran/faster-whisper-tiny | MIT |
| opus-mt-en-zh | ~600 MB | 英→中翻译 | Helsinki-NLP/opus-mt-en-zh | Apache 2.0 |
| silero-vad | ~2 MB | 语音活动检测 | snakers4/silero-vad | MIT |

### 原创功能部分

以下为本项目独立设计与实现的原创功能：

- **流式音频处理管线**：VAD → ASR → MT → TTS 的多线程流式调度架构，支持实时音频流的逐句切分、识别、翻译和播报
- **上下文修正算法**：包含断句合并重译（Reshoot）和滑动窗口上下文回顾翻译（Context-aware Re-translation）两种自修正机制，可随上下文积累自动纠正早期翻译错误
- **翻译协调器**（TranslationCoordinator）：管理上下文窗口、触发翻译、比较修正、推送状态更新的核心调度模块
- **悬浮字幕 UI 架构**：基于 PyQt6 的跨显示器、半透明、可拖拽字幕窗口，含修正闪烁动画和位置记忆
- **状态管理器**（StateManager）：线程安全的单例状态管理，实现 UI 与处理管线的解耦
- **降级策略体系**：GPU→CPU、small→tiny、MT崩溃→显示原文、VAD异常→固定切句 等多层降级保障

---

## 📁 项目结构

```
AI-Simultaneous-Interpretation-Assistant/
├── src/                        # 源代码
│   ├── __init__.py
│   ├── main.py                 # 应用入口
│   ├── audio_capture.py        # 音频捕获模块
│   ├── vad.py                  # 语音活动检测
│   ├── asr_engine.py           # 语音识别引擎
│   ├── mt_engine.py            # 机器翻译引擎
│   ├── tts_engine.py           # 语音合成引擎 (v2.0)
│   ├── translator_coordinator.py  # 翻译调度与修正
│   ├── state_manager.py        # 状态管理器
│   ├── config_manager.py       # 配置读写
│   ├── logger.py               # 日志模块
│   └── gui.py                  # 图形界面
├── tests/                      # 测试
│   ├── __init__.py
│   ├── test_env.py             # 环境与模型加载测试
│   ├── test_audio_pipeline.py  # 音频管道测试
│   └── ...
├── scripts/                    # 工具脚本
│   └── download_models.py      # 模型下载
├── models/                     # 模型文件（不纳入版本控制）
├── logs/                       # 运行日志
├── exports/                    # 导出字幕
├── config.json                 # 用户配置
├── requirements.txt            # Python 依赖
├── README.md                   # 项目说明
├── 设计文档.md                 # 完整设计文档
└── .gitignore
```

---

## 🧪 运行测试

```bash
# 环境测试
python tests/test_env.py

# 音频管道测试
python tests/test_audio_pipeline.py

# 全部测试
python -m pytest tests/
```

---

## 🔒 隐私说明

- 所有处理完全在本地完成，**不发起任何网络请求**
- 音频数据仅在内存中暂存处理，**不落盘持久化**
- 日志文件**不包含**原始音频数据和完整翻译文本

---

## 📄 许可证

本项目代码采用 MIT 许可证。所依赖的第三方模型和库各按其原始许可证分发。

---

> 项目始于 2026-06-05 · 持续开发中
