#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI интерфейс для Audio Normalizer
Командная строка для нормализации аудио в видео файлах
"""

import argparse
import sys
import os
from pathlib import Path
from typing import List
import glob

from audio_normalizer import AudioNormalizer


def find_media_files(paths: List[str], recursive: bool = False) -> List[Path]:
    """
    Поиск медиа файлов в указанных путях
    
    Args:
        paths: Список путей (файлы или директории)
        recursive: Рекурсивный поиск в поддиректориях
        
    Returns:
        Список найденных медиа файлов
    """
    media_files = []
    supported_extensions = AudioNormalizer.SUPPORTED_VIDEO_FORMATS | AudioNormalizer.SUPPORTED_AUDIO_FORMATS
    
    for path_str in paths:
        path = Path(path_str)
        
        if path.is_file():
            if path.suffix.lower() in supported_extensions:
                media_files.append(path)
            else:
                print(f"Предупреждение: Неподдерживаемый формат файла: {path}")
        
        elif path.is_dir():
            pattern = "**/*" if recursive else "*"
            for ext in supported_extensions:
                media_files.extend(path.glob(f"{pattern}{ext}"))
                media_files.extend(path.glob(f"{pattern}{ext.upper()}"))
        
        else:
            # Попробуем как glob паттерн
            matches = glob.glob(path_str, recursive=recursive)
            for match in matches:
                match_path = Path(match)
                if match_path.is_file() and match_path.suffix.lower() in supported_extensions:
                    media_files.append(match_path)
    
    return list(set(media_files))  # Убираем дубликаты


def main():
    parser = argparse.ArgumentParser(
        description="Audio Normalizer - нормализация звука в видео файлах",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  %(prog)s video.mp4                           # Нормализация одного файла
  %(prog)s *.mp4 -o output/                    # Пакетная обработка
  %(prog)s folder/ -r -m rms -t -16           # Рекурсивная обработка с RMS
  %(prog)s video.mp4 --noise-reduction        # С подавлением шумов
  %(prog)s video.mp4 --gpu --speech-enhance   # GPU + улучшение речи

Поддерживаемые форматы:
  Видео: .mp4, .avi, .mkv, .mov, .wmv, .flv, .webm
  Аудио: .mp3, .wav, .flac, .aac, .ogg, .m4a
        """
    )
    
    # Основные аргументы
    parser.add_argument('input', nargs='+', 
                       help='Входные файлы или директории')
    
    parser.add_argument('-o', '--output', 
                       help='Выходная директория (по умолчанию рядом с исходными файлами)')
    
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='Рекурсивный поиск файлов в директориях')
    
    # Параметры нормализации
    norm_group = parser.add_argument_group('Параметры нормализации')
    
    norm_group.add_argument('-m', '--method', 
                           choices=['ebu_r128', 'rms', 'peak'], 
                           default='ebu_r128',
                           help='Метод нормализации (по умолчанию: ebu_r128)')
    
    norm_group.add_argument('-t', '--target', type=float, default=-23.0,
                           help='Целевая громкость в LUFS/dB (по умолчанию: -23.0)')
    
    norm_group.add_argument('--no-preserve-video', action='store_true',
                           help='Не сохранять видео поток без изменений')
    
    # Дополнительная обработка
    enhance_group = parser.add_argument_group('Улучшение качества')
    
    enhance_group.add_argument('--noise-reduction', action='store_true',
                              help='Включить подавление шумов')
    
    enhance_group.add_argument('--speech-enhance', action='store_true',
                              help='Улучшение речи')
    
    enhance_group.add_argument('--dynamic-compression', action='store_true',
                              help='Сжатие динамического диапазона')
    
    # Технические параметры
    tech_group = parser.add_argument_group('Технические параметры')
    
    tech_group.add_argument('--gpu', action='store_true',
                           help='Использовать GPU для ускорения')
    
    tech_group.add_argument('--no-gpu', action='store_true',
                           help='Принудительно использовать CPU')
    
    tech_group.add_argument('--format', default='mp4',
                           help='Формат выходного файла (по умолчанию: mp4)')
    
    tech_group.add_argument('--audio-codec', default='aac',
                           help='Аудио кодек (по умолчанию: aac)')
    
    tech_group.add_argument('--audio-bitrate', default='192k',
                           help='Битрейт аудио (по умолчанию: 192k)')
    
    tech_group.add_argument('--sample-rate', type=int, default=48000,
                           help='Частота дискретизации (по умолчанию: 48000)')
    
    tech_group.add_argument('--temp-dir',
                           help='Директория для временных файлов')
    
    # Управление выводом
    output_group = parser.add_argument_group('Управление выводом')
    
    output_group.add_argument('-v', '--verbose', action='store_true',
                             help='Подробный вывод')
    
    output_group.add_argument('-q', '--quiet', action='store_true',
                             help='Тихий режим')
    
    output_group.add_argument('--info', action='store_true',
                             help='Показать информацию о файлах без обработки')
    
    output_group.add_argument('--estimate-time', action='store_true',
                             help='Оценить время обработки')
    
    output_group.add_argument('--system-info', action='store_true',
                             help='Показать информацию о системе')
    
    args = parser.parse_args()
    
    # Проверка конфликтующих аргументов
    if args.gpu and args.no_gpu:
        parser.error("--gpu и --no-gpu не могут использоваться одновременно")
    
    if args.verbose and args.quiet:
        parser.error("--verbose и --quiet не могут использоваться одновременно")
    
    # Настройка уровня вывода
    verbose = args.verbose and not args.quiet
    
    try:
        # Создание нормализатора
        normalizer = AudioNormalizer(
            method=args.method,
            target_loudness=args.target,
            use_gpu=args.gpu and not args.no_gpu,
            noise_reduction=args.noise_reduction,
            speech_enhancement=args.speech_enhance,
            dynamic_range_compression=args.dynamic_compression,
            preserve_video=not args.no_preserve_video,
            output_format=args.format,
            audio_codec=args.audio_codec,
            audio_bitrate=args.audio_bitrate,
            sample_rate=args.sample_rate,
            temp_dir=args.temp_dir,
            verbose=verbose
        )
        
        # Показать информацию о системе
        if args.system_info:
            print("Информация о системе:")
            for key, value in normalizer.get_system_info().items():
                print(f"  {key}: {value}")
            print()
        
        # Поиск медиа файлов
        media_files = find_media_files(args.input, args.recursive)
        
        if not media_files:
            print("Ошибка: Медиа файлы не найдены")
            return 1
        
        print(f"Найдено файлов: {len(media_files)}")
        
        if verbose:
            for file in media_files:
                print(f"  {file}")
        
        # Показать информацию о файлах
        if args.info:
            print("\nИнформация о файлах:")
            for file in media_files:
                try:
                    info = normalizer.get_audio_info(file)
                    print(f"\n{file}:")
                    print(f"  Длительность: {info['duration']:.2f} сек")
                    print(f"  Битрейт: {info['bitrate']} bps")
                    print(f"  Частота: {info['sample_rate']} Hz")
                    print(f"  Каналы: {info['channels']}")
                    print(f"  Кодек: {info['codec']}")
                    print(f"  Формат: {info['format']}")
                except Exception as e:
                    print(f"\n{file}: Ошибка получения информации - {e}")
            return 0
        
        # Оценка времени обработки
        if args.estimate_time:
            estimated_time = normalizer.estimate_processing_time(media_files)
            print(f"\nОценочное время обработки: {estimated_time:.1f} сек ({estimated_time/60:.1f} мин)")
            
            response = input("Продолжить обработку? (y/N): ")
            if response.lower() not in ['y', 'yes', 'да', 'д']:
                print("Обработка отменена")
                return 0
        
        # Обработка файлов
        if len(media_files) == 1:
            # Одиночный файл
            input_file = media_files[0]
            
            if args.output:
                output_dir = Path(args.output)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / normalizer._generate_output_filename(input_file)
            else:
                output_file = None
            
            result = normalizer.normalize_file(input_file, output_file)
            print(f"\nОбработка завершена: {result}")
        
        else:
            # Пакетная обработка
            output_dir = args.output
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            results = normalizer.normalize_batch(media_files, output_dir)
            
            print(f"\nПакетная обработка завершена:")
            print(f"  Обработано файлов: {len(results)}")
            print(f"  Ошибок: {len(media_files) - len(results)}")
            
            if verbose:
                print("\nОбработанные файлы:")
                for result in results:
                    print(f"  {result}")
        
        return 0
        
    except KeyboardInterrupt:
        print("\nОбработка прервана пользователем")
        return 1
    
    except Exception as e:
        print(f"Ошибка: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())