# 音频标准化工具

[![许可证: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![English](https://img.shields.io/badge/English-README-blue)](README.md)
[![Русский](https://img.shields.io/badge/Русский-README-blue)](README.ru.md)
[![中文](https://img.shields.io/badge/中文-README-blue)](README.zh.md)

一个支持多语言和批量处理的视频文件音频标准化多功能应用程序。

## 功能特点

- 🎯 自动将音频标准化至专业水平
- 📁 支持单个文件和整个文件夹的处理
- 🌐 支持10种界面语言：
  - 英语 (en)
  - 俄语 (ru)
  - 西班牙语 (es)
  - 法语 (fr)
  - 德语 (de)
  - 意大利语 (it)
  - 葡萄牙语 (pt)
  - 日语 (ja)
  - 中文 (zh)
  - 韩语 (ko)
- ⚡ 多线程文件处理
- 📊 进度显示
- 🚫 支持取消操作
- 🎨 现代深色主题界面

## 系统要求

- Python 3.8 或更高版本
- FFmpeg
- 以下Python包：
  - customtkinter
  - pillow
  - moviepy

## 安装说明

1. 安装FFmpeg：
   - Windows: [下载FFmpeg](https://ffmpeg.org/download.html)
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

2. 安装所需的Python包：
```bash
pip install -r requirements.txt
```

## 使用方法

1. 启动应用程序：
```bash
python normalize_audio.py
```

2. 选择操作模式：
   - "单个文件" - 处理单个文件
   - "文件夹" - 处理文件夹中的所有视频文件

3. 选择要处理的文件或文件夹

4. 选择保存处理后的文件的文件夹

5. 点击"标准化"开始处理

## 支持的格式

- MP4 (.mp4)
- AVI (.avi)
- MKV (.mkv)
- MOV (.mov)

## 处理特点

- 自动分析音频参数
- 标准化至标准响度水平 (-23 LUFS)
- 动态标准化实现均匀音效
- 保持原始视频流
- 高质量音频 (320k比特率, 48kHz)

## 许可证

MIT许可证 