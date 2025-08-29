#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Audio Normalizer для видео файлов
Основной модуль для нормализации звука с поддержкой различных алгоритмов
"""

import os
import sys
import logging
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import List, Optional, Union, Dict, Any
import json
import time
import numpy as np

try:
    import torch
    import torchaudio
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch не установлен. GPU ускорение недоступно.")

try:
    import numpy as np
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    logging.warning("Библиотеки для обработки аудио не установлены.")

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False
    logging.warning("noisereduce не установлен. Подавление шумов ограничено.")

from tqdm import tqdm
import psutil


class AudioNormalizer:
    """
    Класс для нормализации аудио в видео файлах
    
    Поддерживает различные методы нормализации:
    - EBU R128 (рекомендуется)
    - RMS нормализация
    - Peak нормализация
    
    Дополнительные возможности:
    - Подавление шумов
    - GPU ускорение
    - Пакетная обработка
    """
    
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm'}
    SUPPORTED_AUDIO_FORMATS = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a'}
    
    def __init__(self, 
                 method: str = 'ebu_r128',
                 target_loudness: float = -23.0,
                 use_gpu: bool = True,
                 noise_reduction: bool = True,
                 speech_enhancement: bool = False,
                 dynamic_range_compression: bool = False,
                 preserve_video: bool = True,
                 output_format: str = 'mp4',
                 audio_codec: str = 'aac',
                 audio_bitrate: str = '192k',
                 sample_rate: int = 48000,
                 temp_dir: Optional[str] = None,
                 verbose: bool = True):
        """
        Инициализация нормализатора
        
        Args:
            method: Метод нормализации ('ebu_r128', 'rms', 'peak')
            target_loudness: Целевая громкость в LUFS/dB
            use_gpu: Использовать GPU если доступно
            noise_reduction: Включить подавление шумов
            speech_enhancement: Улучшение речи
            dynamic_range_compression: Сжатие динамического диапазона
            preserve_video: Сохранять видео поток без изменений
            output_format: Формат выходного файла
            audio_codec: Аудио кодек
            audio_bitrate: Битрейт аудио
            sample_rate: Частота дискретизации
            temp_dir: Директория для временных файлов
            verbose: Подробный вывод
        """
        self.method = method
        self.target_loudness = target_loudness
        self.use_gpu = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self.noise_reduction = noise_reduction
        self.speech_enhancement = speech_enhancement
        self.dynamic_range_compression = dynamic_range_compression
        self.preserve_video = preserve_video
        self.output_format = output_format
        self.audio_codec = audio_codec
        self.audio_bitrate = audio_bitrate
        self.sample_rate = sample_rate
        self.temp_dir = temp_dir or tempfile.gettempdir()
        self.verbose = verbose
        
        # Настройка логирования
        self._setup_logging()
        
        # Проверка зависимостей
        self._check_dependencies()
        
        # Инициализация GPU если доступно
        if self.use_gpu:
            self.device = torch.device('cuda')
            self.logger.info(f"Используется GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Используется CPU")
    
    def _setup_logging(self):
        """Настройка системы логирования"""
        self.logger = logging.getLogger('AudioNormalizer')
        self.logger.setLevel(logging.DEBUG if self.verbose else logging.INFO)
        
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
    
    def _check_dependencies(self):
        """Проверка необходимых зависимостей"""
        # Проверка FFmpeg
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, check=True)
            self.logger.info("FFmpeg найден")
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("FFmpeg не найден. Установите FFmpeg и добавьте в PATH")
        
        # Проверка ffmpeg-normalize
        ffmpeg_normalize_found = False
        
        # Сначала пробуем прямую команду
        try:
            result = subprocess.run(['ffmpeg-normalize', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                ffmpeg_normalize_found = True
        except:
            pass
        
        # Если не получилось, пробуем через python -m
        if not ffmpeg_normalize_found:
            try:
                result = subprocess.run([sys.executable, '-m', 'ffmpeg_normalize', '--version'], 
                                      capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    ffmpeg_normalize_found = True
            except:
                pass
        
        if ffmpeg_normalize_found:
            self.logger.info("ffmpeg-normalize найден")
        else:
            raise RuntimeError("ffmpeg-normalize не найден. Установите: pip install ffmpeg-normalize")
    
    def _get_ffmpeg_normalize_cmd(self):
        """Получить правильную команду для ffmpeg-normalize"""
        # Сначала пробуем прямую команду
        try:
            result = subprocess.run(['ffmpeg-normalize', '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return ['ffmpeg-normalize']
        except:
            pass
        
        # Если не получилось, используем python -m
        return [sys.executable, '-m', 'ffmpeg_normalize']
    
    def _get_file_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Получение информации о файле"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', str(file_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            data = json.loads(result.stdout)
            
            # Извлекаем информацию о файле
            format_info = data.get('format', {})
            streams = data.get('streams', [])
            
            # Находим аудио поток
            audio_stream = None
            for stream in streams:
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            file_info = {
                'format': format_info.get('format_name', 'unknown'),
                'duration': float(format_info.get('duration', 0)),
                'size': int(format_info.get('size', 0)),
                'bitrate': int(format_info.get('bit_rate', 0))
            }
            
            if audio_stream:
                file_info.update({
                    'audio_codec': audio_stream.get('codec_name', 'unknown'),
                    'sample_rate': int(audio_stream.get('sample_rate', 0)),
                    'channels': int(audio_stream.get('channels', 0)),
                    'audio_bitrate': int(audio_stream.get('bit_rate', 0))
                })
            
            return file_info
            
        except Exception as e:
            self.logger.error(f"Ошибка получения информации о файле: {e}")
            return {}
    
    def _analyze_audio_for_settings(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Анализ аудио для определения оптимальных настроек"""
        try:
            file_info = self._get_file_info(file_path)
            settings = {}
            
            # Определяем оптимальный метод нормализации
            duration = file_info.get('duration', 0)
            if duration > 300:  # Более 5 минут - используем EBU R128
                settings['method'] = 'ebu_r128'
                settings['target_loudness'] = -23.0
            elif duration > 60:  # 1-5 минут - RMS
                settings['method'] = 'rms'
                settings['target_loudness'] = -20.0
            else:  # Короткие файлы - Peak
                settings['method'] = 'peak'
                settings['target_loudness'] = -3.0
            
            # Анализируем качество аудио
            sample_rate = file_info.get('sample_rate', 0)
            audio_bitrate = file_info.get('audio_bitrate', 0)
            
            # Рекомендуем подавление шумов для низкокачественного аудио
            if audio_bitrate < 128000 or sample_rate < 44100:
                settings['noise_reduction'] = True
                settings['speech_enhancement'] = True
            else:
                settings['noise_reduction'] = False
                settings['speech_enhancement'] = False
            
            # Оптимальная частота дискретизации
            if sample_rate >= 48000:
                settings['sample_rate'] = 48000
            elif sample_rate >= 44100:
                settings['sample_rate'] = 44100
            else:
                settings['sample_rate'] = 48000  # Апсемплинг
            
            # Оптимальный битрейт
            if audio_bitrate >= 256000:
                settings['audio_bitrate'] = '256k'
            elif audio_bitrate >= 192000:
                settings['audio_bitrate'] = '192k'
            else:
                settings['audio_bitrate'] = '192k'  # Минимум для хорошего качества
            
            # Дополнительный анализ с помощью librosa (если доступно)
            if AUDIO_LIBS_AVAILABLE:
                try:
                    # Извлекаем аудио для анализа
                    temp_audio = self._extract_audio_for_analysis(file_path)
                    if temp_audio:
                        audio_analysis = self._analyze_audio_characteristics(temp_audio)
                        settings.update(audio_analysis)
                        os.unlink(temp_audio)  # Удаляем временный файл
                except Exception as e:
                    self.logger.warning(f"Не удалось выполнить детальный анализ аудио: {e}")
            
            return settings
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа аудио: {e}")
            return {
                'method': 'ebu_r128',
                'target_loudness': -23.0,
                'noise_reduction': False,
                'speech_enhancement': False,
                'sample_rate': 48000,
                'audio_bitrate': '192k'
            }
    
    def _extract_audio_for_analysis(self, file_path: Union[str, Path]) -> Optional[str]:
        """Извлечение короткого фрагмента аудио для анализа"""
        try:
            temp_audio = os.path.join(self.temp_dir, f"analysis_{int(time.time())}.wav")
            
            # Извлекаем первые 30 секунд для анализа
            cmd = [
                'ffmpeg', '-i', str(file_path),
                '-t', '30',  # Первые 30 секунд
                '-vn',  # Без видео
                '-acodec', 'pcm_s16le',
                '-ar', '22050',  # Низкая частота для быстрого анализа
                '-ac', '1',  # Моно
                '-y', temp_audio
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0 and os.path.exists(temp_audio):
                return temp_audio
            
        except Exception as e:
            self.logger.warning(f"Не удалось извлечь аудио для анализа: {e}")
        
        return None
    
    def _analyze_audio_characteristics(self, audio_path: str) -> Dict[str, Any]:
        """Анализ характеристик аудио с помощью librosa"""
        try:
            # Загружаем аудио
            y, sr = librosa.load(audio_path, sr=None)
            
            # Анализ уровня шума
            rms_energy = librosa.feature.rms(y=y)[0]
            noise_level = np.percentile(rms_energy, 10)  # 10-й процентиль как уровень шума
            
            # Анализ спектральных характеристик
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            
            # Определяем тип контента (речь/музыка)
            mean_centroid = np.mean(spectral_centroids)
            mean_rolloff = np.mean(spectral_rolloff)
            
            analysis = {}
            
            # Рекомендации на основе анализа
            if noise_level > 0.01:  # Высокий уровень шума
                analysis['noise_reduction'] = True
            
            # Если спектральные характеристики указывают на речь
            if mean_centroid < 2000 and mean_rolloff < 4000:
                analysis['speech_enhancement'] = True
                analysis['target_loudness'] = -16.0  # Громче для речи
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Ошибка анализа характеристик аудио: {e}")
            return {}
    
    def normalize_file(self, 
                      input_path: Union[str, Path], 
                      output_path: Optional[Union[str, Path]] = None,
                      **kwargs) -> str:
        """
        Нормализация одного файла
        
        Args:
            input_path: Путь к входному файлу
            output_path: Путь к выходному файлу (опционально)
            **kwargs: Дополнительные параметры
            
        Returns:
            Путь к обработанному файлу
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"Файл не найден: {input_path}")
        
        if not self._is_supported_format(input_path):
            raise ValueError(f"Неподдерживаемый формат файла: {input_path.suffix}")
        
        # Определение выходного пути
        if output_path is None:
            output_path = self._generate_output_path(input_path)
        else:
            output_path = Path(output_path)
        
        self.logger.info(f"Начинаем обработку: {input_path}")
        start_time = time.time()
        
        try:
            # Основная обработка
            if self.method == 'ebu_r128':
                self._normalize_ebu_r128(input_path, output_path, **kwargs)
            elif self.method == 'rms':
                self._normalize_rms(input_path, output_path, **kwargs)
            elif self.method == 'peak':
                self._normalize_peak(input_path, output_path, **kwargs)
            else:
                raise ValueError(f"Неподдерживаемый метод: {self.method}")
            
            # Дополнительная обработка
            if self.noise_reduction and NOISEREDUCE_AVAILABLE:
                self._apply_noise_reduction(output_path)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Обработка завершена за {processing_time:.2f} сек: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Ошибка при обработке {input_path}: {e}")
            raise
    
    def normalize_batch(self, 
                       input_paths: List[Union[str, Path]], 
                       output_dir: Optional[Union[str, Path]] = None,
                       max_workers: Optional[int] = None) -> List[str]:
        """
        Пакетная обработка файлов
        
        Args:
            input_paths: Список путей к входным файлам
            output_dir: Директория для выходных файлов
            max_workers: Максимальное количество потоков
            
        Returns:
            Список путей к обработанным файлам
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        results = []
        failed_files = []
        
        self.logger.info(f"Начинаем пакетную обработку {len(input_paths)} файлов")
        
        with tqdm(total=len(input_paths), desc="Обработка файлов") as pbar:
            for input_path in input_paths:
                try:
                    input_path = Path(input_path)
                    
                    if output_dir:
                        output_path = output_dir / self._generate_output_filename(input_path)
                    else:
                        output_path = None
                    
                    result = self.normalize_file(input_path, output_path)
                    results.append(result)
                    
                except Exception as e:
                    self.logger.error(f"Ошибка при обработке {input_path}: {e}")
                    failed_files.append(str(input_path))
                
                pbar.update(1)
        
        if failed_files:
            self.logger.warning(f"Не удалось обработать {len(failed_files)} файлов: {failed_files}")
        
        self.logger.info(f"Пакетная обработка завершена. Обработано: {len(results)} файлов")
        return results
    
    def _normalize_ebu_r128(self, input_path: Path, output_path: Path, **kwargs):
        """EBU R128 нормализация через ffmpeg-normalize"""
        cmd = self._get_ffmpeg_normalize_cmd() + [
            str(input_path),
            '-o', str(output_path),
            '-c:a', self.audio_codec,
            '-b:a', self.audio_bitrate,
            '--loudness-range-target', '7.0',
            '--true-peak', '-2.0',
            '--target-level', str(self.target_loudness)
        ]
        
        if self.preserve_video:
            cmd.extend(['-c:v', 'copy'])
        
        if not self.verbose:
            cmd.append('--quiet')
        
        self.logger.debug(f"Команда: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if self.verbose and result.stdout:
                self.logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Ошибка ffmpeg-normalize: {e.stderr}")
            raise
    
    def _normalize_rms(self, input_path: Path, output_path: Path, **kwargs):
        """RMS нормализация"""
        cmd = self._get_ffmpeg_normalize_cmd() + [
            str(input_path),
            '-o', str(output_path),
            '-c:a', self.audio_codec,
            '-b:a', self.audio_bitrate,
            '--normalization-type', 'rms',
            '--target-level', str(self.target_loudness)
        ]
        
        if self.preserve_video:
            cmd.extend(['-c:v', 'copy'])
        
        if not self.verbose:
            cmd.append('--quiet')
        
        self.logger.debug(f"Команда: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if self.verbose and result.stdout:
                self.logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Ошибка ffmpeg-normalize: {e.stderr}")
            raise
    
    def _normalize_peak(self, input_path: Path, output_path: Path, **kwargs):
        """Peak нормализация"""
        cmd = self._get_ffmpeg_normalize_cmd() + [
            str(input_path),
            '-o', str(output_path),
            '-c:a', self.audio_codec,
            '-b:a', self.audio_bitrate,
            '--normalization-type', 'peak',
            '--target-level', str(self.target_loudness)
        ]
        
        if self.preserve_video:
            cmd.extend(['-c:v', 'copy'])
        
        if not self.verbose:
            cmd.append('--quiet')
        
        self.logger.debug(f"Команда: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            if self.verbose and result.stdout:
                self.logger.info(result.stdout)
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Ошибка ffmpeg-normalize: {e.stderr}")
            raise
    
    def _apply_noise_reduction(self, file_path: Path):
        """Применение подавления шумов"""
        if not AUDIO_LIBS_AVAILABLE or not NOISEREDUCE_AVAILABLE:
            self.logger.warning("Библиотеки для подавления шумов недоступны")
            return
        
        self.logger.info("Применяем подавление шумов...")
        
        try:
            # Временный файл для аудио
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_audio:
                temp_audio_path = temp_audio.name
            
            # Извлечение аудио
            cmd = [
                'ffmpeg', '-i', str(file_path),
                '-vn', '-acodec', 'pcm_s16le',
                '-ar', str(self.sample_rate),
                '-ac', '1',  # моно для лучшей обработки
                '-y', temp_audio_path
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Загрузка аудио
            data, rate = librosa.load(temp_audio_path, sr=self.sample_rate)
            
            # Подавление шумов
            reduced_noise = nr.reduce_noise(y=data, sr=rate, stationary=False)
            
            # Сохранение обработанного аудио
            sf.write(temp_audio_path, reduced_noise, rate)
            
            # Замена аудио в исходном файле
            temp_output = str(file_path).replace('.', '_temp.')
            cmd = [
                'ffmpeg', '-i', str(file_path), '-i', temp_audio_path,
                '-c:v', 'copy', '-c:a', self.audio_codec,
                '-b:a', self.audio_bitrate,
                '-map', '0:v:0', '-map', '1:a:0',
                '-y', temp_output
            ]
            
            subprocess.run(cmd, capture_output=True, check=True)
            
            # Замена исходного файла
            os.replace(temp_output, str(file_path))
            
            # Очистка временных файлов
            os.unlink(temp_audio_path)
            
            self.logger.info("Подавление шумов завершено")
            
        except Exception as e:
            self.logger.error(f"Ошибка при подавлении шумов: {e}")
            # Очистка временных файлов в случае ошибки
            for temp_file in [temp_audio_path, temp_output]:
                try:
                    if 'temp_file' in locals() and os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass
    
    def _is_supported_format(self, file_path: Path) -> bool:
        """Проверка поддерживаемого формата"""
        suffix = file_path.suffix.lower()
        return suffix in self.SUPPORTED_VIDEO_FORMATS or suffix in self.SUPPORTED_AUDIO_FORMATS
    
    def _generate_output_path(self, input_path: Path) -> Path:
        """Генерация пути для выходного файла"""
        stem = input_path.stem
        parent = input_path.parent
        return parent / f"{stem}_normalized.{self.output_format}"
    
    def _generate_output_filename(self, input_path: Path) -> str:
        """Генерация имени выходного файла"""
        stem = input_path.stem
        return f"{stem}_normalized.{self.output_format}"
    
    def get_audio_info(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """Получение информации об аудио файле"""
        file_path = Path(file_path)
        
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', str(file_path)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            info = json.loads(result.stdout)
            
            # Поиск аудио потока
            audio_stream = None
            for stream in info.get('streams', []):
                if stream.get('codec_type') == 'audio':
                    audio_stream = stream
                    break
            
            if audio_stream:
                return {
                    'duration': float(info['format'].get('duration', 0)),
                    'bitrate': int(info['format'].get('bit_rate', 0)),
                    'sample_rate': int(audio_stream.get('sample_rate', 0)),
                    'channels': int(audio_stream.get('channels', 0)),
                    'codec': audio_stream.get('codec_name', 'unknown'),
                    'format': info['format'].get('format_name', 'unknown')
                }
            else:
                raise ValueError("Аудио поток не найден")
                
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Ошибка получения информации о файле: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Ошибка парсинга JSON: {e}")
            raise
    
    def estimate_processing_time(self, file_paths: List[Union[str, Path]]) -> float:
        """Оценка времени обработки"""
        total_duration = 0
        
        for file_path in file_paths:
            try:
                info = self.get_audio_info(file_path)
                total_duration += info['duration']
            except Exception as e:
                self.logger.warning(f"Не удалось получить информацию о {file_path}: {e}")
        
        # Примерная оценка: 0.5x реального времени на CPU, 2x на GPU
        multiplier = 0.2 if self.use_gpu else 0.5
        return total_duration * multiplier
    
    def get_system_info(self) -> Dict[str, Any]:
        """Получение информации о системе"""
        info = {
            'cpu_count': psutil.cpu_count(),
            'memory_total': psutil.virtual_memory().total,
            'memory_available': psutil.virtual_memory().available,
            'gpu_available': self.use_gpu,
        }
        
        if self.use_gpu and TORCH_AVAILABLE:
            info.update({
                'gpu_name': torch.cuda.get_device_name(),
                'gpu_memory_total': torch.cuda.get_device_properties(0).total_memory,
                'gpu_memory_allocated': torch.cuda.memory_allocated(),
            })
        
        return info


if __name__ == '__main__':
    # Пример использования
    normalizer = AudioNormalizer(
        method='ebu_r128',
        target_loudness=-23.0,
        use_gpu=True,
        noise_reduction=True,
        verbose=True
    )
    
    # Информация о системе
    print("Информация о системе:")
    for key, value in normalizer.get_system_info().items():
        print(f"  {key}: {value}")
    
    print("\nАудио нормализатор готов к работе!")
    print("Используйте normalize_file() или normalize_batch() для обработки файлов.")