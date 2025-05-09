# Нормализатор аудио

[![Лицензия: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![English](https://img.shields.io/badge/English-README-blue)](README.md)
[![Русский](https://img.shields.io/badge/Русский-README-blue)](README.ru.md)
[![中文](https://img.shields.io/badge/中文-README-blue)](README.zh.md)

Многофункциональное приложение для нормализации аудио в видеофайлах с поддержкой множества языков и пакетной обработки.

## Возможности

- 🎯 Автоматическая нормализация аудио до профессионального уровня
- 📁 Обработка как отдельных файлов, так и целых папок
- 🌐 Поддержка 10 языков интерфейса:
  - Английский (en)
  - Русский (ru)
  - Испанский (es)
  - Французский (fr)
  - Немецкий (de)
  - Итальянский (it)
  - Португальский (pt)
  - Японский (ja)
  - Китайский (zh)
  - Корейский (ko)
- ⚡ Многопоточная обработка файлов
- 📊 Отображение прогресса обработки
- 🚫 Возможность отмены операции
- 🎨 Современный интерфейс с темной темой

## Требования

- Python 3.8 или выше
- FFmpeg
- Следующие Python-пакеты:
  - customtkinter
  - pillow
  - moviepy

## Установка

1. Установите FFmpeg:
   - Windows: [Скачать FFmpeg](https://ffmpeg.org/download.html)
   - Linux: `sudo apt-get install ffmpeg`
   - macOS: `brew install ffmpeg`

2. Установите необходимые Python-пакеты:
```bash
pip install -r requirements.txt
```

## Использование

1. Запустите приложение:
```bash
python normalize_audio.py
```

2. Выберите режим работы:
   - "Один файл" - для обработки одного файла
   - "Папка" - для обработки всех видеофайлов в папке

3. Выберите файл или папку для обработки

4. Выберите папку для сохранения обработанных файлов

5. Нажмите "Нормализовать" для начала обработки

## Поддерживаемые форматы

- MP4 (.mp4)
- AVI (.avi)
- MKV (.mkv)
- MOV (.mov)

## Особенности обработки

- Автоматический анализ аудио параметров
- Нормализация до стандартного уровня громкости (-23 LUFS)
- Динамическая нормализация для равномерного звучания
- Сохранение оригинального видео потока
- Высокое качество аудио (320k битрейт, 48kHz)

## Лицензия

MIT License 