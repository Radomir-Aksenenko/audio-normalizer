import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os
from PIL import Image
import threading
import re
import queue
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import locale
import json

class AudioNormalizerApp:
    def __init__(self):
        self.window = ctk.CTk()
        self.window.title("Audio Normalizer")
        self.window.geometry("800x700")
        
        # Настройка темы
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Очередь для обновления UI
        self.queue = queue.Queue()
        
        # Флаг для отмены операции
        self.cancel_operation = False
        
        # Семафор для синхронизации прогресса
        self.progress_lock = threading.Lock()
        
        # Максимальное количество потоков
        self.max_workers = os.cpu_count() or 4
        
        # Загрузка языков
        self.load_languages()
        
        # Определение языка системы
        self.current_language = self.detect_system_language()
        
        # Создаем элементы интерфейса
        self.setup_ui()
        
        # Инициализируем тексты интерфейса
        self.update_ui_texts()
        
        # Запускаем проверку очереди
        self.check_queue()
        
    def load_languages(self):
        self.languages = {
            'en': {
                'title': 'Audio Normalizer',
                'mode_label': 'Mode:',
                'file_mode': 'Single File',
                'folder_mode': 'Folder',
                'select_file': 'Select File',
                'select_folder': 'Select Folder',
                'select_save': 'Select Save Folder',
                'normalize': 'Normalize',
                'cancel': 'Cancel',
                'total_progress': 'Total Progress:',
                'current_progress': 'Current File Progress:',
                'no_file': 'No file/folder selected',
                'no_save': 'No save folder selected',
                'analyzing': 'Analyzing audio...',
                'normalizing': 'Normalizing...',
                'success': 'Success!',
                'error': 'Error',
                'canceled': 'Operation canceled',
                'no_audio': 'No audio stream found',
                'processing': 'Processing file:',
                'complete': 'Processing complete!',
                'files_processed': 'Files processed:'
            },
            'ru': {
                'title': 'Нормализатор аудио',
                'mode_label': 'Режим:',
                'file_mode': 'Один файл',
                'folder_mode': 'Папка',
                'select_file': 'Выбрать файл',
                'select_folder': 'Выбрать папку',
                'select_save': 'Выбрать папку для сохранения',
                'normalize': 'Нормализовать',
                'cancel': 'Отменить',
                'total_progress': 'Общий прогресс:',
                'current_progress': 'Прогресс текущего файла:',
                'no_file': 'Файл/папка не выбраны',
                'no_save': 'Папка сохранения не выбрана',
                'analyzing': 'Анализ аудио...',
                'normalizing': 'Нормализация...',
                'success': 'Успешно!',
                'error': 'Ошибка',
                'canceled': 'Операция отменена',
                'no_audio': 'Аудио поток не найден',
                'processing': 'Обработка файла:',
                'complete': 'Обработка завершена!',
                'files_processed': 'Обработано файлов:'
            },
            'es': {
                'title': 'Normalizador de Audio',
                'mode_label': 'Modo:',
                'file_mode': 'Archivo Único',
                'folder_mode': 'Carpeta',
                'select_file': 'Seleccionar Archivo',
                'select_folder': 'Seleccionar Carpeta',
                'select_save': 'Seleccionar Carpeta de Guardado',
                'normalize': 'Normalizar',
                'cancel': 'Cancelar',
                'total_progress': 'Progreso Total:',
                'current_progress': 'Progreso del Archivo Actual:',
                'no_file': 'No se ha seleccionado archivo/carpeta',
                'no_save': 'No se ha seleccionado carpeta de guardado',
                'analyzing': 'Analizando audio...',
                'normalizing': 'Normalizando...',
                'success': '¡Éxito!',
                'error': 'Error',
                'canceled': 'Operación cancelada',
                'no_audio': 'No se encontró flujo de audio',
                'processing': 'Procesando archivo:',
                'complete': '¡Procesamiento completado!',
                'files_processed': 'Archivos procesados:'
            },
            'fr': {
                'title': 'Normalisateur Audio',
                'mode_label': 'Mode:',
                'file_mode': 'Fichier Unique',
                'folder_mode': 'Dossier',
                'select_file': 'Sélectionner un Fichier',
                'select_folder': 'Sélectionner un Dossier',
                'select_save': 'Sélectionner un Dossier de Sauvegarde',
                'normalize': 'Normaliser',
                'cancel': 'Annuler',
                'total_progress': 'Progression Totale:',
                'current_progress': 'Progression du Fichier Actuel:',
                'no_file': 'Aucun fichier/dossier sélectionné',
                'no_save': 'Aucun dossier de sauvegarde sélectionné',
                'analyzing': 'Analyse audio...',
                'normalizing': 'Normalisation...',
                'success': 'Succès!',
                'error': 'Erreur',
                'canceled': 'Opération annulée',
                'no_audio': 'Aucun flux audio trouvé',
                'processing': 'Traitement du fichier:',
                'complete': 'Traitement terminé!',
                'files_processed': 'Fichiers traités:'
            },
            'de': {
                'title': 'Audio-Normalisierer',
                'mode_label': 'Modus:',
                'file_mode': 'Einzelne Datei',
                'folder_mode': 'Ordner',
                'select_file': 'Datei Auswählen',
                'select_folder': 'Ordner Auswählen',
                'select_save': 'Speicherordner Auswählen',
                'normalize': 'Normalisieren',
                'cancel': 'Abbrechen',
                'total_progress': 'Gesamtfortschritt:',
                'current_progress': 'Aktueller Dateifortschritt:',
                'no_file': 'Keine Datei/Ordner ausgewählt',
                'no_save': 'Kein Speicherordner ausgewählt',
                'analyzing': 'Audio wird analysiert...',
                'normalizing': 'Normalisierung...',
                'success': 'Erfolg!',
                'error': 'Fehler',
                'canceled': 'Vorgang abgebrochen',
                'no_audio': 'Kein Audiostream gefunden',
                'processing': 'Datei wird verarbeitet:',
                'complete': 'Verarbeitung abgeschlossen!',
                'files_processed': 'Verarbeitete Dateien:'
            },
            'it': {
                'title': 'Normalizzatore Audio',
                'mode_label': 'Modalità:',
                'file_mode': 'Singolo File',
                'folder_mode': 'Cartella',
                'select_file': 'Seleziona File',
                'select_folder': 'Seleziona Cartella',
                'select_save': 'Seleziona Cartella di Salvataggio',
                'normalize': 'Normalizza',
                'cancel': 'Annulla',
                'total_progress': 'Progresso Totale:',
                'current_progress': 'Progresso File Corrente:',
                'no_file': 'Nessun file/cartella selezionato',
                'no_save': 'Nessuna cartella di salvataggio selezionata',
                'analyzing': 'Analisi audio...',
                'normalizing': 'Normalizzazione...',
                'success': 'Successo!',
                'error': 'Errore',
                'canceled': 'Operazione annullata',
                'no_audio': 'Nessun flusso audio trovato',
                'processing': 'Elaborazione file:',
                'complete': 'Elaborazione completata!',
                'files_processed': 'File elaborati:'
            },
            'pt': {
                'title': 'Normalizador de Áudio',
                'mode_label': 'Modo:',
                'file_mode': 'Arquivo Único',
                'folder_mode': 'Pasta',
                'select_file': 'Selecionar Arquivo',
                'select_folder': 'Selecionar Pasta',
                'select_save': 'Selecionar Pasta de Salvamento',
                'normalize': 'Normalizar',
                'cancel': 'Cancelar',
                'total_progress': 'Progresso Total:',
                'current_progress': 'Progresso do Arquivo Atual:',
                'no_file': 'Nenhum arquivo/pasta selecionado',
                'no_save': 'Nenhuma pasta de salvamento selecionada',
                'analyzing': 'Analisando áudio...',
                'normalizing': 'Normalizando...',
                'success': 'Sucesso!',
                'error': 'Erro',
                'canceled': 'Operação cancelada',
                'no_audio': 'Nenhum fluxo de áudio encontrado',
                'processing': 'Processando arquivo:',
                'complete': 'Processamento concluído!',
                'files_processed': 'Arquivos processados:'
            },
            'ja': {
                'title': 'オーディオ正規化',
                'mode_label': 'モード:',
                'file_mode': '単一ファイル',
                'folder_mode': 'フォルダ',
                'select_file': 'ファイルを選択',
                'select_folder': 'フォルダを選択',
                'select_save': '保存フォルダを選択',
                'normalize': '正規化',
                'cancel': 'キャンセル',
                'total_progress': '全体の進捗:',
                'current_progress': '現在のファイルの進捗:',
                'no_file': 'ファイル/フォルダが選択されていません',
                'no_save': '保存フォルダが選択されていません',
                'analyzing': 'オーディオを分析中...',
                'normalizing': '正規化中...',
                'success': '成功!',
                'error': 'エラー',
                'canceled': '操作がキャンセルされました',
                'no_audio': 'オーディオストリームが見つかりません',
                'processing': 'ファイルを処理中:',
                'complete': '処理が完了しました!',
                'files_processed': '処理されたファイル:'
            },
            'zh': {
                'title': '音频标准化',
                'mode_label': '模式:',
                'file_mode': '单个文件',
                'folder_mode': '文件夹',
                'select_file': '选择文件',
                'select_folder': '选择文件夹',
                'select_save': '选择保存文件夹',
                'normalize': '标准化',
                'cancel': '取消',
                'total_progress': '总进度:',
                'current_progress': '当前文件进度:',
                'no_file': '未选择文件/文件夹',
                'no_save': '未选择保存文件夹',
                'analyzing': '正在分析音频...',
                'normalizing': '正在标准化...',
                'success': '成功!',
                'error': '错误',
                'canceled': '操作已取消',
                'no_audio': '未找到音频流',
                'processing': '正在处理文件:',
                'complete': '处理完成!',
                'files_processed': '已处理文件:'
            },
            'ko': {
                'title': '오디오 정규화',
                'mode_label': '모드:',
                'file_mode': '단일 파일',
                'folder_mode': '폴더',
                'select_file': '파일 선택',
                'select_folder': '폴더 선택',
                'select_save': '저장 폴더 선택',
                'normalize': '정규화',
                'cancel': '취소',
                'total_progress': '전체 진행률:',
                'current_progress': '현재 파일 진행률:',
                'no_file': '파일/폴더가 선택되지 않았습니다',
                'no_save': '저장 폴더가 선택되지 않았습니다',
                'analyzing': '오디오 분석 중...',
                'normalizing': '정규화 중...',
                'success': '성공!',
                'error': '오류',
                'canceled': '작업이 취소되었습니다',
                'no_audio': '오디오 스트림을 찾을 수 없습니다',
                'processing': '파일 처리 중:',
                'complete': '처리가 완료되었습니다!',
                'files_processed': '처리된 파일:'
            }
        }
        
    def detect_system_language(self):
        try:
            # Получаем язык системы
            system_lang = locale.getdefaultlocale()[0][:2]
            # Проверяем, есть ли перевод для этого языка
            if system_lang in self.languages:
                return system_lang
        except:
            pass
        # Если не удалось определить или нет перевода, используем английский
        return 'en'
        
    def change_language(self, lang_code):
        if lang_code in self.languages:
            self.current_language = lang_code
            self.language_var.set(lang_code)
            self.update_ui_texts()
            
    def update_ui_texts(self):
        lang = self.languages[self.current_language]
        self.window.title(lang['title'])
        self.mode_label.configure(text=lang['mode_label'])
        self.file_radio.configure(text=lang['file_mode'])
        self.folder_radio.configure(text=lang['folder_mode'])
        self.select_button.configure(text=lang['select_file'])
        self.save_button.configure(text=lang['select_save'])
        self.normalize_button.configure(text=lang['normalize'])
        self.cancel_button.configure(text=lang['cancel'])
        self.total_progress_label.configure(text=lang['total_progress'])
        self.current_progress_label.configure(text=lang['current_progress'])
        
        # Обновляем текст для выбранного файла/папки
        if hasattr(self, 'file_path'):
            self.file_label.configure(
                text=f"{lang['select_file']}: {os.path.basename(self.file_path)}"
            )
        elif hasattr(self, 'folder_path'):
            self.file_label.configure(
                text=f"{lang['select_folder']}: {os.path.basename(self.folder_path)}"
            )
        else:
            self.file_label.configure(text=lang['no_file'])
            
        # Обновляем текст для папки сохранения
        if hasattr(self, 'save_path'):
            self.save_label.configure(
                text=f"{lang['select_save']}: {os.path.basename(self.save_path)}"
            )
        else:
            self.save_label.configure(text=lang['no_save'])
            
        # Обновляем текст статуса
        if hasattr(self, 'status_label'):
            self.status_label.configure(text="")
            
        # Обновляем текст счетчика
        if hasattr(self, 'counter_label'):
            self.counter_label.configure(text="")
            
    def setup_ui(self):
        # Основной фрейм с прокруткой
        main_frame = ctk.CTkScrollableFrame(self.window)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)
        
        # Заголовок с иконкой
        header_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        header_frame.pack(fill="x", pady=(0, 20))
        
        title_label = ctk.CTkLabel(
            header_frame, 
            text="Audio Normalizer",
            font=("Arial", 24, "bold")
        )
        title_label.pack(side="left", padx=10)
        
        # Фрейм для выбора режима
        mode_frame = ctk.CTkFrame(main_frame, height=50)
        mode_frame.pack(fill="x", pady=10)
        
        self.mode_label = ctk.CTkLabel(mode_frame, text="Mode:", font=("Arial", 14))
        self.mode_label.pack(side="left", padx=10)
        
        self.mode_var = tk.StringVar(value="file")
        
        self.file_radio = ctk.CTkRadioButton(
            mode_frame,
            text="Single File",
            variable=self.mode_var,
            value="file",
            font=("Arial", 12)
        )
        self.file_radio.pack(side="left", padx=20)
        
        self.folder_radio = ctk.CTkRadioButton(
            mode_frame,
            text="Folder",
            variable=self.mode_var,
            value="folder",
            font=("Arial", 12)
        )
        self.folder_radio.pack(side="left", padx=20)
        
        # Фрейм для выбора файлов
        file_frame = ctk.CTkFrame(main_frame)
        file_frame.pack(fill="x", pady=10)
        
        # Кнопки выбора
        self.select_button = ctk.CTkButton(
            file_frame,
            text="Select File",
            command=self.select_file,
            width=200,
            height=40,
            font=("Arial", 12)
        )
        self.select_button.pack(side="left", padx=10, pady=10)
        
        # Индикатор выбранного файла/папки
        self.file_label = ctk.CTkLabel(
            file_frame,
            text="No file/folder selected",
            wraplength=400,
            font=("Arial", 12)
        )
        self.file_label.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        # Фрейм для выбора папки сохранения
        save_frame = ctk.CTkFrame(main_frame)
        save_frame.pack(fill="x", pady=10)
        
        # Кнопка выбора папки сохранения
        self.save_button = ctk.CTkButton(
            save_frame,
            text="Select Save Folder",
            command=self.select_save_folder,
            width=200,
            height=40,
            font=("Arial", 12)
        )
        self.save_button.pack(side="left", padx=10, pady=10)
        
        # Индикатор папки сохранения
        self.save_label = ctk.CTkLabel(
            save_frame,
            text="No save folder selected",
            wraplength=400,
            font=("Arial", 12)
        )
        self.save_label.pack(side="left", padx=10, pady=10, fill="x", expand=True)
        
        # Фрейм для кнопок управления
        control_frame = ctk.CTkFrame(main_frame)
        control_frame.pack(fill="x", pady=20)
        
        # Кнопка нормализации
        self.normalize_button = ctk.CTkButton(
            control_frame,
            text="Normalize",
            command=self.normalize_audio,
            width=200,
            height=40,
            font=("Arial", 12, "bold")
        )
        self.normalize_button.pack(side="left", padx=10, pady=10)
        
        # Кнопка отмены
        self.cancel_button = ctk.CTkButton(
            control_frame,
            text="Cancel",
            command=lambda: self.cancel_operation(),
            state="disabled",
            width=200,
            height=40,
            font=("Arial", 12),
            fg_color="#FF4444",
            hover_color="#CC0000"
        )
        self.cancel_button.pack(side="left", padx=10, pady=10)
        
        # Фрейм для прогресс-баров
        progress_frame = ctk.CTkFrame(main_frame)
        progress_frame.pack(fill="x", pady=10)
        
        # Общий прогресс
        total_progress_header = ctk.CTkFrame(progress_frame, fg_color="transparent")
        total_progress_header.pack(fill="x", pady=5)
        
        self.total_progress_label = ctk.CTkLabel(
            total_progress_header,
            text="Total Progress:",
            font=("Arial", 12, "bold")
        )
        self.total_progress_label.pack(side="left")
        
        self.total_progress_value = ctk.CTkLabel(
            total_progress_header,
            text="0%",
            font=("Arial", 12)
        )
        self.total_progress_value.pack(side="right")
        
        self.total_progress_bar = ctk.CTkProgressBar(progress_frame)
        self.total_progress_bar.pack(fill="x", pady=5)
        self.total_progress_bar.set(0)
        
        # Прогресс текущего файла
        current_progress_header = ctk.CTkFrame(progress_frame, fg_color="transparent")
        current_progress_header.pack(fill="x", pady=(20, 5))
        
        self.current_progress_label = ctk.CTkLabel(
            current_progress_header,
            text="Current File Progress:",
            font=("Arial", 12, "bold")
        )
        self.current_progress_label.pack(side="left")
        
        self.current_progress_value = ctk.CTkLabel(
            current_progress_header,
            text="0%",
            font=("Arial", 12)
        )
        self.current_progress_value.pack(side="right")
        
        self.current_progress_bar = ctk.CTkProgressBar(progress_frame)
        self.current_progress_bar.pack(fill="x", pady=5)
        self.current_progress_bar.set(0)
        
        # Статус
        status_frame = ctk.CTkFrame(main_frame, height=50)
        status_frame.pack(fill="x", pady=10)
        
        self.status_label = ctk.CTkLabel(
            status_frame,
            text="",
            font=("Arial", 12),
            wraplength=700
        )
        self.status_label.pack(pady=10)
        
        # Счетчик обработанных файлов
        counter_frame = ctk.CTkFrame(main_frame, height=30)
        counter_frame.pack(fill="x", pady=5)
        
        self.counter_label = ctk.CTkLabel(
            counter_frame,
            text="",
            font=("Arial", 12)
        )
        self.counter_label.pack(pady=5)
        
        # Добавляем выпадающее меню для выбора языка
        language_frame = ctk.CTkFrame(main_frame)
        language_frame.pack(fill="x", pady=10)
        
        language_label = ctk.CTkLabel(
            language_frame,
            text="Language:",
            font=("Arial", 12)
        )
        language_label.pack(side="left", padx=10)
        
        self.language_var = tk.StringVar(value=self.current_language)
        language_menu = ctk.CTkOptionMenu(
            language_frame,
            values=list(self.languages.keys()),
            command=self.change_language,
            variable=self.language_var,
            width=100
        )
        language_menu.pack(side="left", padx=10)
        
    def check_queue(self):
        try:
            while True:
                func, args, kwargs = self.queue.get_nowait()
                func(*args, **kwargs)
        except queue.Empty:
            pass
        finally:
            self.window.after(100, self.check_queue)
            
    def update_ui(self, func, *args, **kwargs):
        self.queue.put((func, args, kwargs))
        
    def update_progress(self, progress_bar, progress_value, progress):
        self.update_ui(progress_bar.set, progress)
        self.update_ui(progress_value.configure, text=f"{int(progress * 100)}%")
        
    def cancel_operation(self):
        self.cancel_operation = True
        self.update_ui(self.status_label.configure, text="Отмена операции...")
        self.update_ui(self.cancel_button.configure, state="disabled")
        self.update_ui(self.normalize_button.configure, state="normal")
        
    def select_file(self):
        if self.mode_var.get() == "file":
            file_path = filedialog.askopenfilename(
                filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")]
            )
            if file_path:
                self.file_path = file_path
                self.update_ui(self.file_label.configure, 
                    text=f"{self.languages[self.current_language]['select_file']}: {os.path.basename(file_path)}")
        else:
            folder_path = filedialog.askdirectory()
            if folder_path:
                self.folder_path = folder_path
                self.update_ui(self.file_label.configure, 
                    text=f"{self.languages[self.current_language]['select_folder']}: {os.path.basename(folder_path)}")
                
    def select_save_folder(self):
        save_path = filedialog.askdirectory()
        if save_path:
            self.save_path = save_path
            self.update_ui(self.save_label.configure, 
                text=f"{self.languages[self.current_language]['select_save']}: {os.path.basename(save_path)}")
            
    def normalize_audio(self):
        if not hasattr(self, 'save_path'):
            messagebox.showerror(
                self.languages[self.current_language]['error'],
                self.languages[self.current_language]['no_save']
            )
            return
            
        self.cancel_operation = False
        self.normalize_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        
        if self.mode_var.get() == "file":
            if not hasattr(self, 'file_path'):
                messagebox.showerror(
                    self.languages[self.current_language]['error'],
                    self.languages[self.current_language]['no_file']
                )
                return
            thread = threading.Thread(target=self._normalize_single_file, args=(self.file_path,))
        else:
            if not hasattr(self, 'folder_path'):
                messagebox.showerror(
                    self.languages[self.current_language]['error'],
                    self.languages[self.current_language]['no_file']
                )
                return
            thread = threading.Thread(target=self._normalize_folder)
            
        thread.start()
            
    def _normalize_folder(self):
        try:
            video_extensions = ('.mp4', '.avi', '.mkv', '.mov')
            video_files = [f for f in os.listdir(self.folder_path) 
                          if f.lower().endswith(video_extensions)]
            
            if not video_files:
                self.update_ui(messagebox.showerror, "Ошибка", "В выбранной папке нет видео файлов")
                return
                
            self.total_files = len(video_files)
            self.processed_files = 0
            
            self.update_progress(self.total_progress_bar, self.total_progress_value, 0)
            self.update_progress(self.current_progress_bar, self.current_progress_value, 0)
            self.update_ui(self.status_label.configure, text=f"Найдено файлов: {self.total_files}")
            
            # Создаем пул потоков
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Создаем список задач
                futures = []
                for i, filename in enumerate(video_files):
                    if self.cancel_operation:
                        self.update_ui(self.status_label.configure, text="Операция отменена")
                        break
                        
                    file_path = os.path.join(self.folder_path, filename)
                    output_path = os.path.join(self.save_path, f"{os.path.splitext(filename)[0]}_normalized.mp4")
                    
                    # Запускаем задачу в пуле потоков
                    future = executor.submit(self._normalize_single_file, file_path, output_path, True)
                    futures.append((future, filename))
                
                # Обрабатываем результаты по мере их завершения
                for future, filename in futures:
                    if self.cancel_operation:
                        break
                        
                    try:
                        future.result()  # Ждем завершения задачи
                        with self.progress_lock:
                            self.processed_files += 1
                            progress = self.processed_files / self.total_files
                            self.update_progress(self.total_progress_bar, self.total_progress_value, progress)
                            self.update_ui(self.counter_label.configure,
                                text=f"Обработан файл: {filename} ({self.processed_files}/{self.total_files})"
                            )
                    except Exception as e:
                        self.update_ui(self.status_label.configure, text=f"Ошибка при обработке {filename}: {str(e)}")
                
            if not self.cancel_operation:
                self.update_ui(self.status_label.configure,
                    text=f"Обработка завершена! Обработано файлов: {self.total_files}"
                )
                self.update_ui(messagebox.showinfo,
                    "Успех", f"Обработка завершена! Обработано файлов: {self.total_files}"
                )
            else:
                self.update_ui(self.status_label.configure, text="Операция отменена")
                self.update_ui(messagebox.showinfo, "Информация", "Операция отменена пользователем")
            
        except Exception as e:
            self.update_ui(self.status_label.configure, text=f"Ошибка: {str(e)}")
            self.update_ui(messagebox.showerror, "Ошибка", str(e))
            
        finally:
            self.update_ui(self.normalize_button.configure, state="normal")
            self.update_ui(self.cancel_button.configure, state="disabled")
            self.update_progress(self.total_progress_bar, self.total_progress_value, 0)
            self.update_progress(self.current_progress_bar, self.current_progress_value, 0)
            self.update_ui(self.counter_label.configure, text="")
            
    def _normalize_single_file(self, file_path, output_path=None, is_folder=False):
        try:
            if self.cancel_operation:
                return
                
            if output_path is None:
                base_name = os.path.splitext(file_path)[0]
                output_path = os.path.join(self.save_path, f"{os.path.basename(base_name)}_normalized.mp4")
            
            # Проверяем наличие аудио потока
            check_audio_cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a',
                '-show_entries', 'stream=codec_type',
                '-of', 'json',
                file_path
            ]
            
            check_process = subprocess.Popen(
                check_audio_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            
            check_output = check_process.communicate()[0]
            if '"codec_type": "audio"' not in check_output:
                raise Exception("В файле не обнаружен аудио поток")
            
            # Шаг 1: Анализ аудио для определения уровня нормализации
            if not is_folder:
                self.update_ui(self.status_label.configure, text="Анализ аудио...")
            
            # Используем более надежный метод анализа
            analyze_cmd = [
                'ffmpeg',
                '-i', file_path,
                '-af', 'ebur128=peak=true:dualmono=true:target=-23:lra=7:print_format=json',
                '-f', 'null',
                '-'
            ]
            
            process = subprocess.Popen(
                analyze_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            output = process.communicate()[1]
            
            if self.cancel_operation:
                if process.poll() is None:
                    process.terminate()
                return
            
            # Извлекаем параметры из вывода с улучшенной обработкой ошибок
            try:
                i_match = re.search(r'"integrated_loudness"\s*:\s*"([^"]+)"', output)
                tp_match = re.search(r'"true_peak"\s*:\s*"([^"]+)"', output)
                lra_match = re.search(r'"loudness_range"\s*:\s*"([^"]+)"', output)
                
                if not all([i_match, tp_match, lra_match]):
                    # Если не удалось получить параметры, используем значения по умолчанию
                    i = -23.0
                    tp = -1.5
                    lra = 7.0
                    self.update_ui(self.status_label.configure, 
                        text="Не удалось получить точные параметры аудио. Используются значения по умолчанию.")
                else:
                    i = float(i_match.group(1))
                    tp = float(tp_match.group(1))
                    lra = float(lra_match.group(1))
            except Exception as e:
                # В случае ошибки используем значения по умолчанию
                i = -23.0
                tp = -1.5
                lra = 7.0
                self.update_ui(self.status_label.configure, 
                    text=f"Ошибка анализа аудио: {str(e)}. Используются значения по умолчанию.")
            
            # Шаг 2: Нормализация с автоматически определенными параметрами
            if not is_folder:
                self.update_ui(self.status_label.configure, text="Нормализация...")
            
            # Комплексная нормализация с несколькими фильтрами
            normalize_cmd = [
                'ffmpeg',
                '-i', file_path,
                '-af', f'loudnorm=I=-23:TP=-1.5:LRA=7:measured_I={i}:measured_TP={tp}:measured_LRA={lra},'
                       f'dynaudnorm=f=75:g=25:r=0.9:p=0.5:m=100:s=12,'
                       f'compand=attacks=0.1:decays=0.1:points=-80/-80|-20/-20|0/0',
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '320k',
                '-ar', '48000',
                '-ac', '2',
                output_path
            ]
            
            process = subprocess.Popen(
                normalize_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            # Получаем длительность видео
            duration_cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'format=duration',
                '-of', 'default=noprint_wrappers=1:nokey=1',
                file_path
            ]
            
            duration_process = subprocess.Popen(
                duration_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0
            )
            
            duration = float(duration_process.communicate()[0])
            
            # Обновляем прогресс
            while True:
                if self.cancel_operation:
                    process.terminate()
                    return
                    
                output = process.stderr.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    if 'time=' in output:
                        time_str = output.split('time=')[1].split()[0]
                        h, m, s = map(float, time_str.split(':'))
                        current_time = h * 3600 + m * 60 + s
                        progress = current_time / duration
                        self.update_progress(self.current_progress_bar, self.current_progress_value, progress)
            
            if process.returncode == 0:
                if not is_folder:
                    self.update_ui(self.status_label.configure, text="Нормализация завершена успешно!")
                    self.update_ui(messagebox.showinfo, "Успех", f"Файл сохранен как: {output_path}")
            else:
                if not is_folder:
                    self.update_ui(self.status_label.configure, text="Ошибка при нормализации")
                    self.update_ui(messagebox.showerror, "Ошибка", "Произошла ошибка при нормализации")
                
        except Exception as e:
            if not is_folder:
                self.update_ui(self.status_label.configure, text=f"Ошибка: {str(e)}")
                self.update_ui(messagebox.showerror, "Ошибка", str(e))
            
        finally:
            if not is_folder:
                self.update_ui(self.normalize_button.configure, state="normal")
                self.update_ui(self.cancel_button.configure, state="disabled")
                self.update_progress(self.current_progress_bar, self.current_progress_value, 0)
            
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = AudioNormalizerApp()
    app.run() 