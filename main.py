#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Normalizer - Главный исполняемый файл
Приложение для нормализации аудио в видео файлах
"""

import sys
import os
import argparse
from pathlib import Path

# Добавляем текущую директорию в путь для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from audio_normalizer import AudioNormalizer
    from cli import main as cli_main
    from gui import AudioNormalizerGUI
except ImportError as e:
    print(f"Ошибка импорта модулей: {e}")
    print("Убедитесь, что все необходимые файлы находятся в текущей директории")
    sys.exit(1)

try:
    import tkinter as tk
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    print("Предупреждение: Tkinter недоступен, GUI интерфейс будет отключен")


def show_banner():
    """
    Показать баннер приложения
    """
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    Audio Normalizer v1.0                    ║
║              Нормализация аудио в видео файлах               ║
║                                                              ║
║  Возможности:                                                ║
║  • EBU R128, RMS, Peak нормализация                         ║
║  • Подавление шумов с помощью нейронных сетей               ║
║  • GPU ускорение (при наличии)                              ║
║  • Пакетная обработка файлов                                ║
║  • Графический и консольный интерфейс                       ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_dependencies():
    """
    Проверка основных зависимостей
    """
    print("Проверка зависимостей...")
    
    dependencies = {
        'FFmpeg': _check_ffmpeg,
        'ffmpeg-normalize': _check_ffmpeg_normalize,
    }
    
    missing_deps = []
    
    for name, check_func in dependencies.items():
        try:
            if check_func():
                print(f"  ✓ {name}: Доступен")
            else:
                print(f"  ✗ {name}: Недоступен")
                missing_deps.append(name)
        except Exception as e:
            print(f"  ✗ {name}: Ошибка проверки - {e}")
            missing_deps.append(name)
    
    if missing_deps:
        print(f"\nОшибка: Отсутствуют критически важные зависимости: {', '.join(missing_deps)}")
        print("\nДля установки зависимостей:")
        print("1. Установите FFmpeg: https://ffmpeg.org/download.html")
        print("2. Установите Python пакеты: pip install -r requirements.txt")
        return False
    
    print("Все основные зависимости доступны!\n")
    return True


def _check_ffmpeg():
    """
    Проверка FFmpeg
    """
    import subprocess
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def _check_ffmpeg_normalize():
    """
    Проверка ffmpeg-normalize
    """
    import subprocess
    try:
        # Сначала пробуем прямую команду
        result = subprocess.run(['ffmpeg-normalize', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            return True
    except:
        pass
    
    try:
        # Если не получилось, пробуем через python -m
        result = subprocess.run([sys.executable, '-m', 'ffmpeg_normalize', '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def run_gui():
    """
    Запуск графического интерфейса
    """
    if not GUI_AVAILABLE:
        print("Ошибка: Tkinter недоступен. Используйте консольный интерфейс.")
        return False
    
    try:
        print("Запуск графического интерфейса...")
        root = tk.Tk()
        app = AudioNormalizerGUI(root)
        root.mainloop()
        return True
    except Exception as e:
        print(f"Ошибка запуска GUI: {e}")
        return False


def run_cli(args=None):
    """
    Запуск консольного интерфейса
    """
    try:
        print("Запуск консольного интерфейса...")
        return cli_main(args)
    except Exception as e:
        print(f"Ошибка запуска CLI: {e}")
        return False


def run_interactive_mode():
    """
    Запуск интерактивного режима
    """
    print("Интерактивный режим Audio Normalizer")
    print("Введите 'help' для получения справки, 'quit' для выхода\n")
    
    normalizer = None
    
    while True:
        try:
            command = input("audio-normalizer> ").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                print("До свидания!")
                break
            
            elif command in ['help', 'h', '?']:
                print_interactive_help()
            
            elif command == 'init':
                try:
                    normalizer = AudioNormalizer(verbose=True)
                    print("Нормализатор инициализирован успешно")
                except Exception as e:
                    print(f"Ошибка инициализации: {e}")
            
            elif command == 'info':
                if normalizer:
                    info = normalizer.get_system_info()
                    print("Информация о системе:")
                    for key, value in info.items():
                        print(f"  {key}: {value}")
                else:
                    print("Сначала инициализируйте нормализатор командой 'init'")
            
            elif command.startswith('normalize '):
                if normalizer:
                    file_path = command[10:].strip()
                    if file_path:
                        try:
                            output_path = f"normalized_{Path(file_path).name}"
                            result = normalizer.normalize_file(file_path, output_path)
                            print(f"Файл нормализован: {result}")
                        except Exception as e:
                            print(f"Ошибка нормализации: {e}")
                    else:
                        print("Укажите путь к файлу")
                else:
                    print("Сначала инициализируйте нормализатор командой 'init'")
            
            elif command == 'gui':
                if run_gui():
                    break
            
            elif command == 'test':
                print("Запуск тестов...")
                os.system(f"{sys.executable} test_audio_normalizer.py")
            
            elif command == '':
                continue
            
            else:
                print(f"Неизвестная команда: {command}")
                print("Введите 'help' для получения справки")
        
        except KeyboardInterrupt:
            print("\nДо свидания!")
            break
        except EOFError:
            print("\nДо свидания!")
            break
        except Exception as e:
            print(f"Ошибка: {e}")


def print_interactive_help():
    """
    Вывод справки для интерактивного режима
    """
    help_text = """
Доступные команды:
  init                    - Инициализировать нормализатор
  info                    - Показать информацию о системе
  normalize <файл>        - Нормализовать указанный файл
  gui                     - Запустить графический интерфейс
  test                    - Запустить тесты
  help, h, ?              - Показать эту справку
  quit, exit, q           - Выйти из программы

Примеры:
  normalize video.mp4     - Нормализовать файл video.mp4
  normalize "C:\\path\\to\\file.avi"  - Нормализовать файл по полному пути
    """
    print(help_text)


def main():
    """
    Главная функция приложения
    """
    parser = argparse.ArgumentParser(
        description='Audio Normalizer - Нормализация аудио в видео файлах',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python main.py                          # Интерактивный режим
  python main.py --gui                    # Графический интерфейс
  python main.py input.mp4               # Нормализация одного файла
  python main.py input.mp4 -o output.mp4 # Нормализация с указанием выходного файла
  python main.py --batch folder/          # Пакетная обработка папки
  python main.py --test                   # Запуск тестов
        """
    )
    
    parser.add_argument('input', nargs='?', help='Входной файл или папка')
    parser.add_argument('-o', '--output', help='Выходной файл или папка')
    parser.add_argument('--gui', action='store_true', help='Запустить графический интерфейс')
    parser.add_argument('--batch', action='store_true', help='Пакетная обработка папки')
    parser.add_argument('--test', action='store_true', help='Запустить тесты')
    parser.add_argument('--interactive', '-i', action='store_true', help='Интерактивный режим')
    parser.add_argument('--no-banner', action='store_true', help='Не показывать баннер')
    parser.add_argument('--check-deps', action='store_true', help='Только проверить зависимости')
    
    # Добавляем аргументы из CLI модуля
    parser.add_argument('--method', choices=['ebu_r128', 'rms', 'peak'], 
                       default='ebu_r128', help='Метод нормализации')
    parser.add_argument('--target-loudness', type=float, default=-23.0,
                       help='Целевая громкость в LUFS (для EBU R128)')
    parser.add_argument('--noise-reduction', action='store_true',
                       help='Включить подавление шумов')
    parser.add_argument('--gpu', action='store_true',
                       help='Использовать GPU ускорение')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Подробный вывод')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Тихий режим')
    
    args = parser.parse_args()
    
    # Показать баннер
    if not args.no_banner and not args.quiet:
        show_banner()
    
    # Проверка зависимостей
    if args.check_deps:
        return 0 if check_dependencies() else 1
    
    if not args.quiet:
        if not check_dependencies():
            return 1
    
    # Запуск тестов
    if args.test:
        print("Запуск тестов...")
        exit_code = os.system(f"{sys.executable} test_audio_normalizer.py")
        return exit_code >> 8 if os.name == 'nt' else exit_code
    
    # Графический интерфейс
    if args.gui:
        return 0 if run_gui() else 1
    
    # Интерактивный режим
    if args.interactive or (not args.input and not args.gui and not args.test):
        run_interactive_mode()
        return 0
    
    # Консольный интерфейс с аргументами
    if args.input:
        # Передаем аргументы в CLI модуль
        cli_args = []
        
        if args.batch:
            cli_args.extend(['--batch', args.input])
        else:
            cli_args.append(args.input)
        
        if args.output:
            cli_args.extend(['-o', args.output])
        
        cli_args.extend(['--method', args.method])
        cli_args.extend(['--target-loudness', str(args.target_loudness)])
        
        if args.noise_reduction:
            cli_args.append('--noise-reduction')
        
        if args.gpu:
            cli_args.append('--gpu')
        
        if args.verbose:
            cli_args.append('--verbose')
        
        if args.quiet:
            cli_args.append('--quiet')
        
        return 0 if run_cli(cli_args) else 1
    
    # Если никаких аргументов не передано, запускаем интерактивный режим
    run_interactive_mode()
    return 0


if __name__ == '__main__':
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nПрограмма прервана пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        sys.exit(1)