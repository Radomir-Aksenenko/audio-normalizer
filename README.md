# Audio Normalizer для видео файлов

Приложение для автоматической нормализации звука в видео файлах с использованием современных алгоритмов обработки аудио и нейронных сетей.

## Возможности

- 🎵 **EBU R128 нормализация** - стандарт вещания для сбалансированного звука
- 🧠 **Нейронные сети** - подавление шумов с помощью RNNoise и других алгоритмов
- ⚡ **GPU ускорение** - использование PyTorch для быстрой обработки
- 📁 **Пакетная обработка** - обработка множества файлов одновременно
- 🎛️ **Гибкие настройки** - различные режимы нормализации
- 🖥️ **Удобный интерфейс** - графический интерфейс для простого использования

## Установка

### Требования

- Python 3.8+
- FFmpeg (последняя версия)
- CUDA (опционально, для GPU ускорения)

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Установка FFmpeg

**Windows:**
1. Скачайте FFmpeg с [официального сайта](https://ffmpeg.org/download.html)
2. Добавьте путь к FFmpeg в переменную PATH

**Linux/macOS:**
```bash
# Ubuntu/Debian
sudo apt update && sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## Использование

### Командная строка

```python
from audio_normalizer import AudioNormalizer

# Создание экземпляра нормализатора
normalizer = AudioNormalizer(
    method='ebu_r128',  # или 'rms', 'peak'
    target_loudness=-23.0,  # LUFS
    use_gpu=True,  # использовать GPU если доступно
    noise_reduction=True  # включить подавление шумов
)

# Обработка одного файла
normalizer.normalize_file('input.mp4', 'output.mp4')

# Пакетная обработка
normalizer.normalize_batch(['file1.mp4', 'file2.mp4'], output_dir='normalized/')
```

### Графический интерфейс

```bash
python gui.py
```

## Алгоритмы нормализации

### EBU R128 (рекомендуется)
- Стандарт европейского вещательного союза
- Целевая громкость: -23 LUFS
- Лучший выбор для большинства случаев

### RMS нормализация
- Основана на среднеквадратичном значении
- Хорошо подходит для музыки

### Peak нормализация
- Нормализация по пиковым значениям
- Быстрая, но менее точная

## Подавление шумов

Приложение использует несколько методов подавления шумов:

1. **RNNoise** - рекуррентная нейронная сеть для подавления фонового шума
2. **Spectral Gating** - спектральное стробирование через noisereduce
3. **Traditional DSP** - классические методы цифровой обработки сигналов

## Примеры использования

### Нормализация диалогов в фильме
```python
normalizer = AudioNormalizer(
    method='ebu_r128',
    target_loudness=-23.0,
    speech_enhancement=True,
    noise_reduction=True
)
normalizer.normalize_file('movie.mp4', 'movie_normalized.mp4')
```

### Обработка подкастов
```python
normalizer = AudioNormalizer(
    method='rms',
    target_loudness=-20.0,
    speech_enhancement=True,
    dynamic_range_compression=True
)
normalizer.normalize_file('podcast.mp4', 'podcast_normalized.mp4')
```

## Конфигурация

Создайте файл `config.yaml` для настройки параметров по умолчанию:

```yaml
audio:
  method: "ebu_r128"
  target_loudness: -23.0
  sample_rate: 48000
  
processing:
  use_gpu: true
  batch_size: 8
  num_workers: 4
  
noise_reduction:
  enabled: true
  method: "rnnoise"
  strength: 0.8
```

## Производительность

- **CPU**: ~0.5x реального времени на современном процессоре
- **GPU**: ~2-5x реального времени с CUDA
- **Память**: ~500MB для обработки HD видео

## Поддерживаемые форматы

**Входные форматы:**
- MP4, AVI, MKV, MOV, WMV
- MP3, WAV, FLAC, AAC, OGG

**Выходные форматы:**
- MP4 (рекомендуется)
- MKV, AVI
- Аудио: AAC, MP3, FLAC

## Устранение неполадок

### FFmpeg не найден
```
Ошибка: FFmpeg не найден в PATH
Решение: Установите FFmpeg и добавьте в PATH
```

### Ошибки CUDA
```
Ошибка: CUDA out of memory
Решение: Уменьшите batch_size или отключите GPU
```

### Медленная обработка
```
Проблема: Обработка занимает много времени
Решение: Включите GPU ускорение или уменьшите качество
```

## Лицензия

MIT License - см. файл LICENSE

## Благодарности

- [ffmpeg-normalize](https://github.com/slhck/ffmpeg-normalize) - основа для нормализации
- [RNNoise](https://github.com/xiph/rnnoise) - подавление шумов
- [noisereduce](https://github.com/timsainb/noisereduce) - дополнительные алгоритмы

## Поддержка

Если у вас есть вопросы или предложения, создайте issue в репозитории.