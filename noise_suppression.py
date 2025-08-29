#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль подавления шумов для Audio Normalizer
Интеграция различных методов подавления шумов включая RNNoise
"""

import os
import sys
import logging
import subprocess
import tempfile
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Union
import time

try:
    import torch
    import torchaudio
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False

try:
    import noisereduce as nr
    NOISEREDUCE_AVAILABLE = True
except ImportError:
    NOISEREDUCE_AVAILABLE = False

try:
    import scipy.signal
    import scipy.fft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class SpectralGateNoiseReduction:
    """
    Спектральное подавление шумов на основе статистического анализа
    """
    
    def __init__(self, 
                 stationary: bool = True,
                 prop_decrease: float = 1.0,
                 n_grad_freq: int = 2,
                 n_grad_time: int = 4,
                 n_fft: int = 2048,
                 win_length: Optional[int] = None,
                 hop_length: Optional[int] = None):
        """
        Инициализация спектрального подавителя шумов
        
        Args:
            stationary: Стационарный ли шум
            prop_decrease: Коэффициент уменьшения шума (0-1)
            n_grad_freq: Градиент по частоте
            n_grad_time: Градиент по времени
            n_fft: Размер FFT
            win_length: Длина окна
            hop_length: Шаг окна
        """
        self.stationary = stationary
        self.prop_decrease = prop_decrease
        self.n_grad_freq = n_grad_freq
        self.n_grad_time = n_grad_time
        self.n_fft = n_fft
        self.win_length = win_length or n_fft
        self.hop_length = hop_length or n_fft // 4
        
        self.logger = logging.getLogger('SpectralGate')
    
    def reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Применение спектрального подавления шумов
        
        Args:
            audio: Аудио сигнал
            sr: Частота дискретизации
            
        Returns:
            Обработанный аудио сигнал
        """
        if not NOISEREDUCE_AVAILABLE:
            self.logger.warning("noisereduce недоступен, используем базовый фильтр")
            return self._basic_noise_filter(audio, sr)
        
        try:
            # Используем noisereduce для спектрального подавления
            reduced_noise = nr.reduce_noise(
                y=audio,
                sr=sr,
                stationary=self.stationary,
                prop_decrease=self.prop_decrease,
                n_grad_freq=self.n_grad_freq,
                n_grad_time=self.n_grad_time,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length
            )
            
            return reduced_noise
            
        except Exception as e:
            self.logger.error(f"Ошибка спектрального подавления: {e}")
            return self._basic_noise_filter(audio, sr)
    
    def _basic_noise_filter(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Базовый фильтр шумов без внешних библиотек
        """
        if not SCIPY_AVAILABLE:
            return audio
        
        # Простой высокочастотный фильтр для удаления низкочастотных шумов
        nyquist = sr / 2
        low_cutoff = 80 / nyquist  # Убираем частоты ниже 80 Гц
        high_cutoff = 8000 / nyquist  # Убираем частоты выше 8 кГц
        
        try:
            # Полосовой фильтр Баттерворта
            b, a = scipy.signal.butter(4, [low_cutoff, high_cutoff], btype='band')
            filtered_audio = scipy.signal.filtfilt(b, a, audio)
            return filtered_audio
        except:
            return audio


class RNNoiseWrapper:
    """
    Обертка для RNNoise (если доступен)
    """
    
    def __init__(self):
        self.logger = logging.getLogger('RNNoise')
        self.rnnoise_available = self._check_rnnoise()
    
    def _check_rnnoise(self) -> bool:
        """
        Проверка доступности RNNoise
        """
        try:
            # Попытка найти rnnoise_demo или другие исполняемые файлы
            result = subprocess.run(['rnnoise_demo'], 
                                  capture_output=True, text=True, timeout=5)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Проверка Python биндингов
        try:
            import rnnoise
            return True
        except ImportError:
            pass
        
        return False
    
    def reduce_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Применение RNNoise для подавления шумов
        
        Args:
            audio: Аудио сигнал
            sr: Частота дискретизации
            
        Returns:
            Обработанный аудио сигнал
        """
        if not self.rnnoise_available:
            self.logger.warning("RNNoise недоступен")
            return audio
        
        try:
            # Попытка использовать Python биндинги
            import rnnoise
            
            # RNNoise работает с 48kHz моно
            target_sr = 48000
            
            # Ресемплинг если нужно
            if sr != target_sr and AUDIO_LIBS_AVAILABLE:
                audio_resampled = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            else:
                audio_resampled = audio
            
            # Конвертация в int16
            audio_int16 = (audio_resampled * 32767).astype(np.int16)
            
            # Применение RNNoise
            denoiser = rnnoise.RNNoise()
            denoised_int16 = denoiser.process(audio_int16)
            
            # Конвертация обратно в float
            denoised_float = denoised_int16.astype(np.float32) / 32767.0
            
            # Ресемплинг обратно если нужно
            if sr != target_sr and AUDIO_LIBS_AVAILABLE:
                denoised_final = librosa.resample(denoised_float, orig_sr=target_sr, target_sr=sr)
            else:
                denoised_final = denoised_float
            
            return denoised_final
            
        except Exception as e:
            self.logger.error(f"Ошибка RNNoise: {e}")
            return self._rnnoise_cli(audio, sr)
    
    def _rnnoise_cli(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Использование RNNoise через CLI
        """
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_input:
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_output:
                    temp_input_path = temp_input.name
                    temp_output_path = temp_output.name
            
            # Сохранение входного аудио
            if AUDIO_LIBS_AVAILABLE:
                sf.write(temp_input_path, audio, sr)
            else:
                return audio
            
            # Запуск rnnoise_demo
            cmd = ['rnnoise_demo', temp_input_path, temp_output_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Загрузка обработанного аудио
                denoised_audio, _ = sf.read(temp_output_path)
                
                # Очистка временных файлов
                os.unlink(temp_input_path)
                os.unlink(temp_output_path)
                
                return denoised_audio
            else:
                self.logger.error(f"RNNoise CLI ошибка: {result.stderr}")
                return audio
                
        except Exception as e:
            self.logger.error(f"Ошибка RNNoise CLI: {e}")
            return audio
        finally:
            # Очистка временных файлов в случае ошибки
            for temp_file in [temp_input_path, temp_output_path]:
                try:
                    if 'temp_file' in locals() and os.path.exists(temp_file):
                        os.unlink(temp_file)
                except:
                    pass


class DeepNoiseReduction(nn.Module):
    """
    Простая нейронная сеть для подавления шумов
    Основана на автоэнкодере для спектрограмм
    """
    
    def __init__(self, n_fft: int = 2048, hidden_dim: int = 512):
        super().__init__()
        
        self.n_fft = n_fft
        self.freq_bins = n_fft // 2 + 1
        
        # Энкодер
        self.encoder = nn.Sequential(
            nn.Linear(self.freq_bins, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Декодер
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim // 4, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, self.freq_bins),
            nn.Sigmoid()  # Маска от 0 до 1
        )
    
    def forward(self, x):
        """
        Прямой проход
        
        Args:
            x: Спектрограмма [batch, freq_bins, time]
            
        Returns:
            Маска для очистки спектрограммы
        """
        batch_size, freq_bins, time_steps = x.shape
        
        # Перестановка для обработки по временным кадрам
        x = x.permute(0, 2, 1)  # [batch, time, freq]
        x = x.reshape(-1, freq_bins)  # [batch*time, freq]
        
        # Энкодер-декодер
        encoded = self.encoder(x)
        mask = self.decoder(encoded)
        
        # Восстановление формы
        mask = mask.reshape(batch_size, time_steps, freq_bins)
        mask = mask.permute(0, 2, 1)  # [batch, freq, time]
        
        return mask


class NoiseSuppressionPipeline:
    """
    Основной класс для подавления шумов
    Объединяет различные методы
    """
    
    def __init__(self, 
                 method: str = 'spectral',
                 use_gpu: bool = True,
                 model_path: Optional[str] = None,
                 verbose: bool = True):
        """
        Инициализация пайплайна подавления шумов
        
        Args:
            method: Метод подавления ('spectral', 'rnnoise', 'deep', 'combined')
            use_gpu: Использовать GPU если доступно
            model_path: Путь к предобученной модели
            verbose: Подробный вывод
        """
        self.method = method
        self.use_gpu = use_gpu and TORCH_AVAILABLE and torch.cuda.is_available()
        self.model_path = model_path
        self.verbose = verbose
        
        self.logger = logging.getLogger('NoiseSuppressionPipeline')
        
        # Инициализация компонентов
        self.spectral_gate = SpectralGateNoiseReduction()
        self.rnnoise = RNNoiseWrapper()
        
        # Инициализация нейронной сети если нужно
        self.deep_model = None
        if method in ['deep', 'combined'] and TORCH_AVAILABLE:
            self._init_deep_model()
        
        if self.use_gpu:
            self.device = torch.device('cuda')
            self.logger.info(f"Используется GPU: {torch.cuda.get_device_name()}")
        else:
            self.device = torch.device('cpu')
            self.logger.info("Используется CPU")
    
    def _init_deep_model(self):
        """
        Инициализация глубокой модели
        """
        try:
            self.deep_model = DeepNoiseReduction()
            
            if self.model_path and os.path.exists(self.model_path):
                # Загрузка предобученной модели
                checkpoint = torch.load(self.model_path, map_location=self.device)
                self.deep_model.load_state_dict(checkpoint['model_state_dict'])
                self.logger.info(f"Загружена модель: {self.model_path}")
            else:
                self.logger.warning("Предобученная модель не найдена, используется случайная инициализация")
            
            self.deep_model.to(self.device)
            self.deep_model.eval()
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации глубокой модели: {e}")
            self.deep_model = None
    
    def suppress_noise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Основной метод подавления шумов
        
        Args:
            audio: Аудио сигнал
            sr: Частота дискретизации
            
        Returns:
            Обработанный аудио сигнал
        """
        if self.verbose:
            self.logger.info(f"Применяем подавление шумов методом: {self.method}")
        
        start_time = time.time()
        
        try:
            if self.method == 'spectral':
                result = self.spectral_gate.reduce_noise(audio, sr)
            
            elif self.method == 'rnnoise':
                result = self.rnnoise.reduce_noise(audio, sr)
            
            elif self.method == 'deep':
                result = self._deep_noise_reduction(audio, sr)
            
            elif self.method == 'combined':
                # Комбинированный подход
                result = self._combined_noise_reduction(audio, sr)
            
            else:
                self.logger.warning(f"Неизвестный метод: {self.method}, используем спектральный")
                result = self.spectral_gate.reduce_noise(audio, sr)
            
            processing_time = time.time() - start_time
            
            if self.verbose:
                self.logger.info(f"Подавление шумов завершено за {processing_time:.2f} сек")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка подавления шумов: {e}")
            return audio
    
    def _deep_noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Глубокое подавление шумов с помощью нейронной сети
        """
        if self.deep_model is None or not TORCH_AVAILABLE:
            self.logger.warning("Глубокая модель недоступна, используем спектральный метод")
            return self.spectral_gate.reduce_noise(audio, sr)
        
        try:
            # Вычисление STFT
            n_fft = self.deep_model.n_fft
            hop_length = n_fft // 4
            
            if AUDIO_LIBS_AVAILABLE:
                stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
                magnitude = np.abs(stft)
                phase = np.angle(stft)
            else:
                return self.spectral_gate.reduce_noise(audio, sr)
            
            # Конвертация в тензор
            magnitude_tensor = torch.FloatTensor(magnitude).unsqueeze(0).to(self.device)
            
            # Применение модели
            with torch.no_grad():
                mask = self.deep_model(magnitude_tensor)
                mask = mask.squeeze(0).cpu().numpy()
            
            # Применение маски
            cleaned_magnitude = magnitude * mask
            
            # Восстановление аудио
            cleaned_stft = cleaned_magnitude * np.exp(1j * phase)
            
            if AUDIO_LIBS_AVAILABLE:
                cleaned_audio = librosa.istft(cleaned_stft, hop_length=hop_length)
            else:
                cleaned_audio = audio
            
            return cleaned_audio
            
        except Exception as e:
            self.logger.error(f"Ошибка глубокого подавления шумов: {e}")
            return self.spectral_gate.reduce_noise(audio, sr)
    
    def _combined_noise_reduction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Комбинированное подавление шумов
        """
        # Сначала спектральное подавление
        audio_spectral = self.spectral_gate.reduce_noise(audio, sr)
        
        # Затем RNNoise если доступен
        if self.rnnoise.rnnoise_available:
            audio_rnnoise = self.rnnoise.reduce_noise(audio_spectral, sr)
        else:
            audio_rnnoise = audio_spectral
        
        # Наконец глубокое подавление если доступно
        if self.deep_model is not None:
            audio_final = self._deep_noise_reduction(audio_rnnoise, sr)
        else:
            audio_final = audio_rnnoise
        
        return audio_final
    
    def get_available_methods(self) -> list:
        """
        Получение списка доступных методов
        """
        methods = ['spectral']
        
        if self.rnnoise.rnnoise_available:
            methods.append('rnnoise')
        
        if TORCH_AVAILABLE and self.deep_model is not None:
            methods.append('deep')
        
        if len(methods) > 1:
            methods.append('combined')
        
        return methods
    
    def benchmark_methods(self, audio: np.ndarray, sr: int) -> dict:
        """
        Бенчмарк различных методов
        
        Args:
            audio: Тестовый аудио сигнал
            sr: Частота дискретизации
            
        Returns:
            Словарь с результатами бенчмарка
        """
        results = {}
        available_methods = self.get_available_methods()
        
        for method in available_methods:
            if method == 'combined':
                continue  # Пропускаем комбинированный для бенчмарка
            
            original_method = self.method
            self.method = method
            
            start_time = time.time()
            processed_audio = self.suppress_noise(audio.copy(), sr)
            processing_time = time.time() - start_time
            
            # Простые метрики качества
            snr_improvement = self._calculate_snr_improvement(audio, processed_audio)
            
            results[method] = {
                'processing_time': processing_time,
                'snr_improvement': snr_improvement,
                'available': True
            }
            
            self.method = original_method
        
        return results
    
    def _calculate_snr_improvement(self, original: np.ndarray, processed: np.ndarray) -> float:
        """
        Простая оценка улучшения SNR
        """
        try:
            # Оценка шума как разность между оригиналом и обработанным
            noise_estimate = original - processed
            
            # SNR как отношение сигнала к шуму
            signal_power = np.mean(processed ** 2)
            noise_power = np.mean(noise_estimate ** 2)
            
            if noise_power > 0:
                snr_db = 10 * np.log10(signal_power / noise_power)
                return snr_db
            else:
                return float('inf')
                
        except:
            return 0.0


if __name__ == '__main__':
    # Пример использования
    pipeline = NoiseSuppressionPipeline(method='spectral', verbose=True)
    
    print("Доступные методы подавления шумов:")
    for method in pipeline.get_available_methods():
        print(f"  - {method}")
    
    print("\nПайплайн подавления шумов готов к работе!")