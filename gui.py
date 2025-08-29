#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GUI интерфейс для Audio Normalizer
Графический интерфейс для нормализации аудио в видео файлах
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
from pathlib import Path
from typing import List, Optional
import time

try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

from audio_normalizer import AudioNormalizer


class AudioNormalizerGUI:
    """
    Графический интерфейс для Audio Normalizer
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("Audio Normalizer - Нормализация звука в видео")
        self.root.geometry("800x700")
        self.root.minsize(600, 500)
        
        # Переменные
        self.input_files = []
        self.output_directory = tk.StringVar()
        self.method = tk.StringVar(value="ebu_r128")
        self.target_loudness = tk.DoubleVar(value=-23.0)
        self.use_gpu = tk.BooleanVar(value=True)
        self.noise_reduction = tk.BooleanVar(value=False)
        self.speech_enhancement = tk.BooleanVar(value=False)
        self.dynamic_compression = tk.BooleanVar(value=False)
        self.preserve_video = tk.BooleanVar(value=True)
        self.output_format = tk.StringVar(value="mp4")
        self.audio_codec = tk.StringVar(value="aac")
        self.audio_bitrate = tk.StringVar(value="192k")
        self.sample_rate = tk.IntVar(value=48000)
        self.verbose = tk.BooleanVar(value=True)
        
        # Очередь для обновления GUI из потоков
        self.log_queue = queue.Queue()
        self.progress_queue = queue.Queue()
        
        # Флаг обработки
        self.processing = False
        self.processing_thread = None
        
        # Создание интерфейса
        self.create_widgets()
        self.setup_styles()
        
        # Запуск обновления GUI
        self.update_gui()
        
        # Проверка зависимостей при запуске
        self.root.after(100, self.check_dependencies)
    
    def create_widgets(self):
        """Создание виджетов интерфейса"""
        # Главный контейнер
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Настройка сетки
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="Audio Normalizer", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=row, column=0, columnspan=3, pady=(0, 20))
        row += 1
        
        # Секция выбора файлов
        files_frame = ttk.LabelFrame(main_frame, text="Файлы для обработки", padding="10")
        files_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        files_frame.columnconfigure(0, weight=1)
        row += 1
        
        # Список файлов
        self.files_listbox = tk.Listbox(files_frame, height=6, selectmode=tk.EXTENDED)
        files_scrollbar = ttk.Scrollbar(files_frame, orient=tk.VERTICAL, command=self.files_listbox.yview)
        self.files_listbox.configure(yscrollcommand=files_scrollbar.set)
        
        self.files_listbox.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        files_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Кнопки управления файлами
        files_buttons_frame = ttk.Frame(files_frame)
        files_buttons_frame.grid(row=1, column=0, columnspan=2, pady=(10, 0))
        
        ttk.Button(files_buttons_frame, text="Добавить файлы", 
                  command=self.add_files).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(files_buttons_frame, text="Добавить папку", 
                  command=self.add_folder).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(files_buttons_frame, text="Удалить выбранные", 
                  command=self.remove_selected_files).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(files_buttons_frame, text="Очистить все", 
                  command=self.clear_files).pack(side=tk.LEFT)
        
        # Секция настроек
        settings_notebook = ttk.Notebook(main_frame)
        settings_notebook.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        main_frame.rowconfigure(row, weight=1)
        row += 1
        
        # Вкладка основных настроек
        basic_frame = ttk.Frame(settings_notebook, padding="10")
        settings_notebook.add(basic_frame, text="Основные настройки")
        
        self.create_basic_settings(basic_frame)
        
        # Вкладка дополнительных настроек
        advanced_frame = ttk.Frame(settings_notebook, padding="10")
        settings_notebook.add(advanced_frame, text="Дополнительно")
        
        self.create_advanced_settings(advanced_frame)
        
        # Вкладка технических настроек
        tech_frame = ttk.Frame(settings_notebook, padding="10")
        settings_notebook.add(tech_frame, text="Технические")
        
        self.create_tech_settings(tech_frame)
        
        # Секция вывода
        output_frame = ttk.LabelFrame(main_frame, text="Выходная директория", padding="10")
        output_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        output_frame.columnconfigure(0, weight=1)
        row += 1
        
        ttk.Entry(output_frame, textvariable=self.output_directory).grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 5))
        ttk.Button(output_frame, text="Обзор", command=self.select_output_directory).grid(row=0, column=1)
        
        # Прогресс бар с подписью
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(0, 10))
        progress_frame.columnconfigure(0, weight=1)
        
        self.progress_label = ttk.Label(progress_frame, text="Готов к обработке")
        self.progress_label.grid(row=0, column=0, sticky=tk.W)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        
        self.progress_percent_label = ttk.Label(progress_frame, text="0%")
        self.progress_percent_label.grid(row=1, column=1, sticky=tk.E, padx=(5, 0))
        row += 1
        
        # Кнопки управления
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=row, column=0, columnspan=3, pady=(0, 10))
        row += 1
        
        self.start_button = ttk.Button(buttons_frame, text="Начать обработку", 
                                      command=self.start_processing, style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(buttons_frame, text="Остановить", 
                                     command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(buttons_frame, text="Информация о системе", 
                  command=self.show_system_info).pack(side=tk.LEFT, padx=(0, 10))
        
        ttk.Button(buttons_frame, text="О программе", 
                  command=self.show_about).pack(side=tk.LEFT)
        
        # Лог
        log_frame = ttk.LabelFrame(main_frame, text="Лог обработки", padding="10")
        log_frame.grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, state=tk.DISABLED)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    
    def create_basic_settings(self, parent):
        """Создание основных настроек"""
        row = 0
        
        # Метод нормализации
        ttk.Label(parent, text="Метод нормализации:").grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        method_combo = ttk.Combobox(parent, textvariable=self.method, 
                                   values=["ebu_r128", "rms", "peak"], state="readonly")
        method_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=(0, 5), padx=(10, 0))
        row += 1
        
        # Целевая громкость
        ttk.Label(parent, text="Целевая громкость (LUFS/dB):").grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        loudness_frame = ttk.Frame(parent)
        loudness_frame.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=(0, 5), padx=(10, 0))
        ttk.Spinbox(loudness_frame, from_=-50, to=0, increment=0.5, textvariable=self.target_loudness, 
                   width=10).pack(side=tk.LEFT)
        ttk.Button(loudness_frame, text="Авто", command=self.auto_configure_settings, 
                  width=6).pack(side=tk.LEFT, padx=(5, 0))
        row += 1
        
        # Чекбоксы
        ttk.Checkbutton(parent, text="Использовать GPU (если доступно)", 
                       variable=self.use_gpu).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        row += 1
        
        ttk.Checkbutton(parent, text="Подавление шумов", 
                       variable=self.noise_reduction).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        row += 1
        
        ttk.Checkbutton(parent, text="Улучшение речи", 
                       variable=self.speech_enhancement).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        row += 1
        
        ttk.Checkbutton(parent, text="Сжатие динамического диапазона", 
                       variable=self.dynamic_compression).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        row += 1
        
        ttk.Checkbutton(parent, text="Сохранять видео поток без изменений", 
                       variable=self.preserve_video).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        parent.columnconfigure(1, weight=1)
    
    def create_advanced_settings(self, parent):
        """Создание дополнительных настроек"""
        row = 0
        
        # Формат выходного файла
        ttk.Label(parent, text="Формат выходного файла:").grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        format_combo = ttk.Combobox(parent, textvariable=self.output_format, 
                                   values=["mp4", "avi", "mkv", "mov", "webm"], state="readonly")
        format_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=(0, 5), padx=(10, 0))
        row += 1
        
        # Аудио кодек
        ttk.Label(parent, text="Аудио кодек:").grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        codec_combo = ttk.Combobox(parent, textvariable=self.audio_codec, 
                                  values=["aac", "mp3", "flac", "opus", "vorbis"], state="readonly")
        codec_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=(0, 5), padx=(10, 0))
        row += 1
        
        # Битрейт аудио
        ttk.Label(parent, text="Битрейт аудио:").grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        bitrate_combo = ttk.Combobox(parent, textvariable=self.audio_bitrate, 
                                    values=["128k", "192k", "256k", "320k", "512k"], state="readonly")
        bitrate_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=(0, 5), padx=(10, 0))
        row += 1
        
        # Частота дискретизации
        ttk.Label(parent, text="Частота дискретизации (Hz):").grid(row=row, column=0, sticky=tk.W, pady=(0, 5))
        sample_combo = ttk.Combobox(parent, textvariable=self.sample_rate, 
                                   values=[22050, 44100, 48000, 96000], state="readonly")
        sample_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=(0, 5), padx=(10, 0))
        
        parent.columnconfigure(1, weight=1)
    
    def create_tech_settings(self, parent):
        """Создание технических настроек"""
        row = 0
        
        ttk.Checkbutton(parent, text="Подробный вывод в лог", 
                       variable=self.verbose).grid(row=row, column=0, columnspan=2, sticky=tk.W, pady=(0, 10))
        row += 1
        
        # Информация о поддерживаемых форматах
        info_text = tk.Text(parent, height=10, wrap=tk.WORD, state=tk.DISABLED)
        info_scrollbar = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=info_text.yview)
        info_text.configure(yscrollcommand=info_scrollbar.set)
        
        info_text.grid(row=row, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        info_scrollbar.grid(row=row, column=1, sticky=(tk.N, tk.S))
        
        # Заполнение информации
        info_content = """
Поддерживаемые форматы:

Видео файлы:
• MP4 (.mp4) - рекомендуется
• AVI (.avi)
• MKV (.mkv)
• MOV (.mov)
• WMV (.wmv)
• FLV (.flv)
• WebM (.webm)

Аудио файлы:
• MP3 (.mp3)
• WAV (.wav)
• FLAC (.flac)
• AAC (.aac)
• OGG (.ogg)
• M4A (.m4a)

Методы нормализации:
• EBU R128 - стандарт вещания (рекомендуется)
• RMS - среднеквадратичная нормализация
• Peak - пиковая нормализация

Дополнительные возможности:
• Подавление шумов с помощью нейронных сетей
• GPU ускорение для быстрой обработки
• Пакетная обработка множества файлов
• Сохранение качества видео
        """
        
        info_text.config(state=tk.NORMAL)
        info_text.insert(tk.END, info_content)
        info_text.config(state=tk.DISABLED)
        
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(row, weight=1)
    
    def setup_styles(self):
        """Настройка стилей"""
        style = ttk.Style()
        
        # Стиль для кнопки "Начать обработку"
        style.configure("Accent.TButton", font=('Arial', 10, 'bold'))
    
    def add_files(self):
        """Добавление файлов"""
        filetypes = [
            ('Все поддерживаемые', '*.mp4;*.avi;*.mkv;*.mov;*.wmv;*.flv;*.webm;*.mp3;*.wav;*.flac;*.aac;*.ogg;*.m4a'),
            ('Видео файлы', '*.mp4;*.avi;*.mkv;*.mov;*.wmv;*.flv;*.webm'),
            ('Аудио файлы', '*.mp3;*.wav;*.flac;*.aac;*.ogg;*.m4a'),
            ('Все файлы', '*.*')
        ]
        
        files = filedialog.askopenfilenames(
            title="Выберите файлы для обработки",
            filetypes=filetypes
        )
        
        for file in files:
            if file not in self.input_files:
                self.input_files.append(file)
                self.files_listbox.insert(tk.END, os.path.basename(file))
    
    def add_folder(self):
        """Добавление папки"""
        folder = filedialog.askdirectory(title="Выберите папку с файлами")
        
        if folder:
            supported_extensions = AudioNormalizer.SUPPORTED_VIDEO_FORMATS | AudioNormalizer.SUPPORTED_AUDIO_FORMATS
            
            for root, dirs, files in os.walk(folder):
                for file in files:
                    if Path(file).suffix.lower() in supported_extensions:
                        full_path = os.path.join(root, file)
                        if full_path not in self.input_files:
                            self.input_files.append(full_path)
                            self.files_listbox.insert(tk.END, os.path.relpath(full_path, folder))
    
    def remove_selected_files(self):
        """Удаление выбранных файлов"""
        selected_indices = self.files_listbox.curselection()
        
        for index in reversed(selected_indices):
            self.files_listbox.delete(index)
            del self.input_files[index]
    
    def clear_files(self):
        """Очистка всех файлов"""
        self.files_listbox.delete(0, tk.END)
        self.input_files.clear()
    
    def select_output_directory(self):
        """Выбор выходной директории"""
        directory = filedialog.askdirectory(title="Выберите папку для сохранения")
        if directory:
            self.output_directory.set(directory)
    
    def start_processing(self):
        """Начало обработки"""
        if not self.input_files:
            messagebox.showwarning("Предупреждение", "Выберите файлы для обработки")
            return
        
        if not self.output_directory.get():
            response = messagebox.askyesno(
                "Выходная директория", 
                "Выходная директория не указана. Сохранить файлы рядом с исходными?"
            )
            if not response:
                return
        
        # Блокировка интерфейса
        self.processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        # Очистка лога
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        
        # Сброс прогресса
        self.progress_var.set(0)
        
        # Запуск обработки в отдельном потоке
        self.processing_thread = threading.Thread(target=self.process_files, daemon=True)
        self.processing_thread.start()
    
    def stop_processing(self):
        """Остановка обработки"""
        self.processing = False
        self.log("Остановка обработки...")
    
    def process_files(self):
        """Обработка файлов в отдельном потоке"""
        try:
            # Создание нормализатора
            normalizer = AudioNormalizer(
                method=self.method.get(),
                target_loudness=self.target_loudness.get(),
                use_gpu=self.use_gpu.get(),
                noise_reduction=self.noise_reduction.get(),
                speech_enhancement=self.speech_enhancement.get(),
                dynamic_range_compression=self.dynamic_compression.get(),
                preserve_video=self.preserve_video.get(),
                output_format=self.output_format.get(),
                audio_codec=self.audio_codec.get(),
                audio_bitrate=self.audio_bitrate.get(),
                sample_rate=self.sample_rate.get(),
                verbose=self.verbose.get()
            )
            
            self.log(f"Начинаем обработку {len(self.input_files)} файлов...")
            
            output_dir = self.output_directory.get() if self.output_directory.get() else None
            
            processed_count = 0
            total_files = len(self.input_files)
            
            for i, input_file in enumerate(self.input_files):
                if not self.processing:
                    break
                
                try:
                    filename = os.path.basename(input_file)
                    self.log(f"Обрабатываем ({i+1}/{total_files}): {filename}")
                    
                    # Обновление прогресса с информацией о текущем файле
                    progress = (i / total_files) * 100
                    self.progress_queue.put({
                        'progress': progress,
                        'label': f'Обработка файла {i+1}/{total_files}: {filename}'
                    })
                    
                    if output_dir:
                        output_path = Path(output_dir) / normalizer._generate_output_filename(Path(input_file))
                    else:
                        output_path = None
                    
                    result = normalizer.normalize_file(input_file, output_path)
                    
                    self.log(f"Готово: {os.path.basename(result)}")
                    processed_count += 1
                    
                except Exception as e:
                    self.log(f"Ошибка при обработке {os.path.basename(input_file)}: {e}")
                
                # Обновление прогресса после завершения файла
                progress = ((i + 1) / total_files) * 100
                self.progress_queue.put({
                    'progress': progress,
                    'label': f'Завершено файлов: {processed_count}/{total_files}'
                })
            
            if self.processing:
                self.log(f"\nОбработка завершена! Обработано файлов: {processed_count}/{total_files}")
            else:
                self.log(f"\nОбработка остановлена. Обработано файлов: {processed_count}/{total_files}")
            
        except Exception as e:
            self.log(f"Критическая ошибка: {e}")
        
        finally:
            # Разблокировка интерфейса
            self.root.after(0, self.processing_finished)
    
    def auto_configure_settings(self):
        """Автонастройка параметров по выбранному файлу"""
        if not self.input_files:
            messagebox.showwarning("Предупреждение", "Сначала выберите файл для анализа")
            return
        
        # Берем первый файл для анализа
        sample_file = self.input_files[0]
        
        try:
            self.progress_label.config(text="Анализ файла...")
            self.progress_var.set(0)
            
            # Запуск анализа в отдельном потоке
            analysis_thread = threading.Thread(target=self._analyze_file, args=(sample_file,), daemon=True)
            analysis_thread.start()
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось проанализировать файл: {e}")
            self.progress_label.config(text="Готов к обработке")
            self.progress_var.set(0)
    
    def _analyze_file(self, file_path):
        """Анализ файла для автонастройки"""
        try:
            from audio_normalizer import AudioNormalizer
            
            # Создаем временный нормализатор для анализа
            analyzer = AudioNormalizer(verbose=False)
            
            self.progress_queue.put({'progress': 25, 'label': 'Анализ аудио характеристик...'})
            
            # Получаем информацию о файле
            file_info = analyzer._get_file_info(file_path)
            
            self.progress_queue.put({'progress': 50, 'label': 'Определение оптимальных параметров...'})
            
            # Анализируем аудио для определения оптимальных настроек
            optimal_settings = analyzer._analyze_audio_for_settings(file_path)
            
            self.progress_queue.put({'progress': 75, 'label': 'Применение настроек...'})
            
            # Применяем настройки в главном потоке
            self.root.after(0, lambda: self._apply_auto_settings(optimal_settings, file_info))
            
            self.progress_queue.put({'progress': 100, 'label': 'Автонастройка завершена'})
            
            # Возвращаем к исходному состоянию через 2 секунды
            self.root.after(2000, lambda: self._reset_progress_to_ready())
            
        except Exception as e:
            self.log_queue.put(f"Ошибка анализа: {e}\n")
            self.root.after(0, lambda: self._reset_progress_to_ready())
    
    def _apply_auto_settings(self, settings, file_info):
        """Применение автоматических настроек"""
        try:
            # Применяем рекомендуемые настройки
            if 'target_loudness' in settings:
                self.target_loudness.set(settings['target_loudness'])
            
            if 'method' in settings:
                self.method.set(settings['method'])
            
            if 'noise_reduction' in settings:
                self.noise_reduction.set(settings['noise_reduction'])
            
            if 'speech_enhancement' in settings:
                self.speech_enhancement.set(settings['speech_enhancement'])
            
            if 'sample_rate' in settings:
                self.sample_rate.set(settings['sample_rate'])
            
            if 'audio_bitrate' in settings:
                self.audio_bitrate.set(settings['audio_bitrate'])
            
            # Показываем информацию о примененных настройках
            info_msg = "Автонастройка завершена!\n\n"
            info_msg += f"Файл: {file_info.get('format', 'Неизвестно')}\n"
            info_msg += f"Длительность: {file_info.get('duration', 'Неизвестно')}\n"
            info_msg += f"Аудио кодек: {file_info.get('audio_codec', 'Неизвестно')}\n"
            info_msg += f"Частота: {file_info.get('sample_rate', 'Неизвестно')} Hz\n\n"
            info_msg += "Применены настройки:\n"
            info_msg += f"• Метод: {settings.get('method', 'ebu_r128')}\n"
            info_msg += f"• Целевая громкость: {settings.get('target_loudness', -23)} LUFS\n"
            info_msg += f"• Подавление шумов: {'Да' if settings.get('noise_reduction', False) else 'Нет'}\n"
            
            messagebox.showinfo("Автонастройка", info_msg)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось применить настройки: {e}")
    
    def _reset_progress_to_ready(self):
        """Сброс прогресса к готовности"""
        self.progress_label.config(text="Готов к обработке")
        self.progress_var.set(0)
        self.progress_percent_label.config(text="0%")
    
    def processing_finished(self):
        """Завершение обработки"""
        self.processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_label.config(text="Обработка завершена")
        self.progress_var.set(100 if self.progress_var.get() > 0 else 0)
        self.progress_percent_label.config(text="100%" if self.progress_var.get() > 0 else "0%")
    
    def log(self, message):
        """Добавление сообщения в лог"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}\n"
        self.log_queue.put(formatted_message)
    
    def update_gui(self):
        """Обновление GUI из очередей"""
        # Обновление лога
        try:
            while True:
                message = self.log_queue.get_nowait()
                self.log_text.config(state=tk.NORMAL)
                self.log_text.insert(tk.END, message)
                self.log_text.see(tk.END)
                self.log_text.config(state=tk.DISABLED)
        except queue.Empty:
            pass
        
        # Обновление прогресса
        try:
            while True:
                progress_data = self.progress_queue.get_nowait()
                if isinstance(progress_data, dict):
                    progress = progress_data.get('progress', 0)
                    label = progress_data.get('label', 'Обработка...')
                    self.progress_var.set(progress)
                    self.progress_label.config(text=label)
                    self.progress_percent_label.config(text=f"{progress:.1f}%")
                else:
                    # Обратная совместимость
                    self.progress_var.set(progress_data)
                    self.progress_percent_label.config(text=f"{progress_data:.1f}%")
        except queue.Empty:
            pass
        
        # Планирование следующего обновления
        self.root.after(100, self.update_gui)
    
    def check_dependencies(self):
        """Проверка зависимостей"""
        try:
            # Создание тестового нормализатора для проверки
            AudioNormalizer(verbose=False)
            self.log("Все зависимости найдены. Готов к работе!")
        except Exception as e:
            self.log(f"Ошибка зависимостей: {e}")
            messagebox.showerror(
                "Ошибка зависимостей",
                f"Не удалось инициализировать Audio Normalizer:\n\n{e}\n\n"
                "Убедитесь, что установлены FFmpeg и ffmpeg-normalize."
            )
    
    def show_system_info(self):
        """Показ информации о системе"""
        try:
            normalizer = AudioNormalizer(verbose=False)
            info = normalizer.get_system_info()
            
            info_text = "Информация о системе:\n\n"
            for key, value in info.items():
                if 'memory' in key.lower():
                    # Конвертация байтов в ГБ
                    value_gb = value / (1024**3)
                    info_text += f"{key}: {value_gb:.1f} ГБ\n"
                else:
                    info_text += f"{key}: {value}\n"
            
            messagebox.showinfo("Информация о системе", info_text)
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось получить информацию о системе: {e}")
    
    def show_about(self):
        """Показ информации о программе"""
        about_text = """
Audio Normalizer v1.0

Программа для нормализации звука в видео файлах.

Возможности:
• EBU R128, RMS и Peak нормализация
• Подавление шумов с помощью ИИ
• GPU ускорение
• Пакетная обработка
• Поддержка множества форматов

Разработано с использованием:
• Python
• FFmpeg
• PyTorch
• Tkinter

© 2024
        """
        
        messagebox.showinfo("О программе", about_text)


def main():
    """Главная функция"""
    root = tk.Tk()
    
    # Настройка темы
    try:
        root.tk.call('source', 'azure.tcl')
        root.tk.call('set_theme', 'light')
    except:
        pass  # Если тема не найдена, используем стандартную
    
    app = AudioNormalizerGUI(root)
    
    # Обработка закрытия окна
    def on_closing():
        if app.processing:
            if messagebox.askokcancel("Выход", "Обработка файлов в процессе. Завершить?"):
                app.processing = False
                root.destroy()
        else:
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Запуск приложения
    root.mainloop()


if __name__ == '__main__':
    main()