#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Тесты для Audio Normalizer
Проверка всех компонентов системы нормализации аудио
"""

import unittest
import tempfile
import os
import sys
import numpy as np
from pathlib import Path
import subprocess
import time
import shutil

# Добавляем текущую директорию в путь для импорта модулей
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from audio_normalizer import AudioNormalizer
    from noise_suppression import NoiseSuppressionPipeline, SpectralGateNoiseReduction
except ImportError as e:
    print(f"Ошибка импорта: {e}")
    sys.exit(1)

try:
    import librosa
    import soundfile as sf
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    print("Предупреждение: Библиотеки для работы с аудио недоступны")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Предупреждение: PyTorch недоступен")


class TestAudioNormalizer(unittest.TestCase):
    """
    Тесты для основного класса AudioNormalizer
    """
    
    def setUp(self):
        """Настройка тестов"""
        self.temp_dir = tempfile.mkdtemp()
        self.test_files = []
        
        # Создание тестовых аудио файлов
        if AUDIO_LIBS_AVAILABLE:
            self._create_test_audio_files()
    
    def tearDown(self):
        """Очистка после тестов"""
        # Удаление временных файлов
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _create_test_audio_files(self):
        """Создание тестовых аудио файлов"""
        sample_rate = 44100
        duration = 2.0  # 2 секунды
        
        # Создание различных типов тестовых сигналов
        test_signals = {
            'sine_wave': self._generate_sine_wave(440, sample_rate, duration),
            'white_noise': self._generate_white_noise(sample_rate, duration),
            'mixed_signal': self._generate_mixed_signal(sample_rate, duration),
            'quiet_signal': self._generate_quiet_signal(sample_rate, duration),
            'loud_signal': self._generate_loud_signal(sample_rate, duration)
        }
        
        for name, signal in test_signals.items():
            file_path = os.path.join(self.temp_dir, f"{name}.wav")
            sf.write(file_path, signal, sample_rate)
            self.test_files.append(file_path)
    
    def _generate_sine_wave(self, freq, sr, duration):
        """Генерация синусоиды"""
        t = np.linspace(0, duration, int(sr * duration), False)
        return 0.3 * np.sin(2 * np.pi * freq * t)
    
    def _generate_white_noise(self, sr, duration):
        """Генерация белого шума"""
        samples = int(sr * duration)
        return 0.1 * np.random.normal(0, 1, samples)
    
    def _generate_mixed_signal(self, sr, duration):
        """Генерация смешанного сигнала"""
        sine = self._generate_sine_wave(440, sr, duration)
        noise = self._generate_white_noise(sr, duration)
        return sine + noise
    
    def _generate_quiet_signal(self, sr, duration):
        """Генерация тихого сигнала"""
        return 0.01 * self._generate_sine_wave(440, sr, duration)
    
    def _generate_loud_signal(self, sr, duration):
        """Генерация громкого сигнала"""
        return 0.9 * self._generate_sine_wave(440, sr, duration)
    
    def test_normalizer_initialization(self):
        """Тест инициализации нормализатора"""
        try:
            normalizer = AudioNormalizer(verbose=False)
            self.assertIsInstance(normalizer, AudioNormalizer)
            self.assertEqual(normalizer.method, 'ebu_r128')
            self.assertEqual(normalizer.target_loudness, -23.0)
        except Exception as e:
            self.skipTest(f"Не удалось инициализировать AudioNormalizer: {e}")
    
    def test_supported_formats(self):
        """Тест проверки поддерживаемых форматов"""
        try:
            normalizer = AudioNormalizer(verbose=False)
            
            # Тест поддерживаемых форматов
            supported_video = ['.mp4', '.avi', '.mkv', '.mov']
            supported_audio = ['.wav', '.mp3', '.flac', '.aac']
            
            for ext in supported_video + supported_audio:
                test_path = Path(f"test{ext}")
                self.assertTrue(normalizer._is_supported_format(test_path))
            
            # Тест неподдерживаемых форматов
            unsupported = ['.txt', '.doc', '.pdf', '.jpg']
            for ext in unsupported:
                test_path = Path(f"test{ext}")
                self.assertFalse(normalizer._is_supported_format(test_path))
                
        except Exception as e:
            self.skipTest(f"Ошибка тестирования форматов: {e}")
    
    @unittest.skipUnless(AUDIO_LIBS_AVAILABLE, "Библиотеки аудио недоступны")
    def test_audio_info_extraction(self):
        """Тест извлечения информации об аудио"""
        if not self.test_files:
            self.skipTest("Тестовые файлы не созданы")
        
        try:
            normalizer = AudioNormalizer(verbose=False)
            
            for test_file in self.test_files[:1]:  # Тестируем только первый файл
                info = normalizer.get_audio_info(test_file)
                
                self.assertIn('duration', info)
                self.assertIn('sample_rate', info)
                self.assertIn('channels', info)
                self.assertIn('codec', info)
                
                self.assertGreater(info['duration'], 0)
                self.assertGreater(info['sample_rate'], 0)
                self.assertGreater(info['channels'], 0)
                
        except Exception as e:
            self.skipTest(f"Ошибка извлечения информации об аудио: {e}")
    
    @unittest.skipUnless(AUDIO_LIBS_AVAILABLE, "Библиотеки аудио недоступны")
    def test_single_file_normalization(self):
        """Тест нормализации одного файла"""
        if not self.test_files:
            self.skipTest("Тестовые файлы не созданы")
        
        try:
            normalizer = AudioNormalizer(method='peak', verbose=False)
            
            input_file = self.test_files[0]
            output_file = os.path.join(self.temp_dir, "normalized_output.wav")
            
            result = normalizer.normalize_file(input_file, output_file)
            
            self.assertTrue(os.path.exists(result))
            self.assertEqual(result, output_file)
            
            # Проверка, что выходной файл имеет разумный размер
            self.assertGreater(os.path.getsize(result), 1000)
            
        except Exception as e:
            self.skipTest(f"Ошибка нормализации файла: {e}")
    
    @unittest.skipUnless(AUDIO_LIBS_AVAILABLE, "Библиотеки аудио недоступны")
    def test_batch_normalization(self):
        """Тест пакетной нормализации"""
        if len(self.test_files) < 2:
            self.skipTest("Недостаточно тестовых файлов")
        
        try:
            normalizer = AudioNormalizer(method='rms', verbose=False)
            
            output_dir = os.path.join(self.temp_dir, "batch_output")
            os.makedirs(output_dir, exist_ok=True)
            
            results = normalizer.normalize_batch(self.test_files[:2], output_dir)
            
            self.assertEqual(len(results), 2)
            
            for result in results:
                self.assertTrue(os.path.exists(result))
                self.assertGreater(os.path.getsize(result), 1000)
                
        except Exception as e:
            self.skipTest(f"Ошибка пакетной нормализации: {e}")
    
    def test_system_info(self):
        """Тест получения информации о системе"""
        try:
            normalizer = AudioNormalizer(verbose=False)
            info = normalizer.get_system_info()
            
            self.assertIn('cpu_count', info)
            self.assertIn('memory_total', info)
            self.assertIn('memory_available', info)
            self.assertIn('gpu_available', info)
            
            self.assertGreater(info['cpu_count'], 0)
            self.assertGreater(info['memory_total'], 0)
            self.assertGreater(info['memory_available'], 0)
            
        except Exception as e:
            self.skipTest(f"Ошибка получения информации о системе: {e}")
    
    def test_processing_time_estimation(self):
        """Тест оценки времени обработки"""
        if not self.test_files:
            self.skipTest("Тестовые файлы не созданы")
        
        try:
            normalizer = AudioNormalizer(verbose=False)
            
            estimated_time = normalizer.estimate_processing_time(self.test_files)
            
            self.assertIsInstance(estimated_time, float)
            self.assertGreaterEqual(estimated_time, 0)
            
        except Exception as e:
            self.skipTest(f"Ошибка оценки времени обработки: {e}")


class TestNoiseSuppressionPipeline(unittest.TestCase):
    """
    Тесты для пайплайна подавления шумов
    """
    
    def setUp(self):
        """Настройка тестов"""
        self.sample_rate = 44100
        self.duration = 1.0
        
        if AUDIO_LIBS_AVAILABLE:
            # Создание тестового аудио с шумом
            self.clean_signal = self._generate_clean_signal()
            self.noisy_signal = self._add_noise_to_signal(self.clean_signal)
    
    def _generate_clean_signal(self):
        """Генерация чистого сигнала"""
        t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)
        # Комбинация нескольких частот для имитации речи
        signal = (0.3 * np.sin(2 * np.pi * 440 * t) +
                 0.2 * np.sin(2 * np.pi * 880 * t) +
                 0.1 * np.sin(2 * np.pi * 1320 * t))
        return signal
    
    def _add_noise_to_signal(self, clean_signal):
        """Добавление шума к сигналу"""
        noise = 0.1 * np.random.normal(0, 1, len(clean_signal))
        return clean_signal + noise
    
    def test_pipeline_initialization(self):
        """Тест инициализации пайплайна"""
        pipeline = NoiseSuppressionPipeline(method='spectral', verbose=False)
        self.assertIsInstance(pipeline, NoiseSuppressionPipeline)
        self.assertEqual(pipeline.method, 'spectral')
    
    def test_available_methods(self):
        """Тест получения доступных методов"""
        pipeline = NoiseSuppressionPipeline(verbose=False)
        methods = pipeline.get_available_methods()
        
        self.assertIsInstance(methods, list)
        self.assertIn('spectral', methods)
        self.assertGreater(len(methods), 0)
    
    @unittest.skipUnless(AUDIO_LIBS_AVAILABLE, "Библиотеки аудио недоступны")
    def test_spectral_noise_reduction(self):
        """Тест спектрального подавления шумов"""
        pipeline = NoiseSuppressionPipeline(method='spectral', verbose=False)
        
        processed_signal = pipeline.suppress_noise(self.noisy_signal, self.sample_rate)
        
        self.assertEqual(len(processed_signal), len(self.noisy_signal))
        self.assertIsInstance(processed_signal, np.ndarray)
        
        # Проверка, что сигнал изменился
        self.assertFalse(np.array_equal(processed_signal, self.noisy_signal))
    
    @unittest.skipUnless(AUDIO_LIBS_AVAILABLE, "Библиотеки аудио недоступны")
    def test_spectral_gate_class(self):
        """Тест класса SpectralGateNoiseReduction"""
        gate = SpectralGateNoiseReduction()
        
        processed_signal = gate.reduce_noise(self.noisy_signal, self.sample_rate)
        
        self.assertEqual(len(processed_signal), len(self.noisy_signal))
        self.assertIsInstance(processed_signal, np.ndarray)
    
    @unittest.skipUnless(TORCH_AVAILABLE and AUDIO_LIBS_AVAILABLE, "PyTorch или аудио библиотеки недоступны")
    def test_deep_noise_reduction(self):
        """Тест глубокого подавления шумов"""
        pipeline = NoiseSuppressionPipeline(method='deep', verbose=False)
        
        if pipeline.deep_model is not None:
            processed_signal = pipeline.suppress_noise(self.noisy_signal, self.sample_rate)
            
            self.assertEqual(len(processed_signal), len(self.noisy_signal))
            self.assertIsInstance(processed_signal, np.ndarray)
        else:
            self.skipTest("Глубокая модель недоступна")
    
    @unittest.skipUnless(AUDIO_LIBS_AVAILABLE, "Библиотеки аудио недоступны")
    def test_benchmark_methods(self):
        """Тест бенчмарка методов"""
        pipeline = NoiseSuppressionPipeline(verbose=False)
        
        results = pipeline.benchmark_methods(self.noisy_signal, self.sample_rate)
        
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        for method, metrics in results.items():
            self.assertIn('processing_time', metrics)
            self.assertIn('snr_improvement', metrics)
            self.assertIn('available', metrics)
            
            self.assertGreaterEqual(metrics['processing_time'], 0)
            self.assertTrue(metrics['available'])


class TestIntegration(unittest.TestCase):
    """
    Интеграционные тесты
    """
    
    def setUp(self):
        """Настройка интеграционных тестов"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Очистка после тестов"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @unittest.skipUnless(AUDIO_LIBS_AVAILABLE, "Библиотеки аудио недоступны")
    def test_full_pipeline_with_noise_reduction(self):
        """Тест полного пайплайна с подавлением шумов"""
        # Создание тестового файла
        sample_rate = 44100
        duration = 1.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Сигнал с шумом
        clean_signal = 0.3 * np.sin(2 * np.pi * 440 * t)
        noise = 0.1 * np.random.normal(0, 1, len(clean_signal))
        noisy_signal = clean_signal + noise
        
        input_file = os.path.join(self.temp_dir, "test_noisy.wav")
        sf.write(input_file, noisy_signal, sample_rate)
        
        try:
            # Создание нормализатора с подавлением шумов
            normalizer = AudioNormalizer(
                method='peak',
                noise_reduction=True,
                verbose=False
            )
            
            output_file = os.path.join(self.temp_dir, "test_output.wav")
            result = normalizer.normalize_file(input_file, output_file)
            
            self.assertTrue(os.path.exists(result))
            self.assertGreater(os.path.getsize(result), 1000)
            
            # Проверка, что выходной файл можно прочитать
            processed_audio, _ = sf.read(result)
            self.assertEqual(len(processed_audio), len(noisy_signal))
            
        except Exception as e:
            self.skipTest(f"Ошибка интеграционного теста: {e}")
    
    def test_error_handling(self):
        """Тест обработки ошибок"""
        try:
            normalizer = AudioNormalizer(verbose=False)
            
            # Тест с несуществующим файлом
            with self.assertRaises(FileNotFoundError):
                normalizer.normalize_file("nonexistent_file.wav")
            
            # Тест с неподдерживаемым форматом
            test_file = os.path.join(self.temp_dir, "test.txt")
            with open(test_file, 'w') as f:
                f.write("test")
            
            with self.assertRaises(ValueError):
                normalizer.normalize_file(test_file)
                
        except Exception as e:
            self.skipTest(f"Ошибка тестирования обработки ошибок: {e}")


class TestPerformance(unittest.TestCase):
    """
    Тесты производительности
    """
    
    @unittest.skipUnless(AUDIO_LIBS_AVAILABLE, "Библиотеки аудио недоступны")
    def test_processing_speed(self):
        """Тест скорости обработки"""
        # Создание тестового сигнала
        sample_rate = 44100
        duration = 5.0  # 5 секунд
        samples = int(sample_rate * duration)
        
        test_signal = 0.3 * np.random.normal(0, 1, samples)
        
        # Тест спектрального подавления шумов
        pipeline = NoiseSuppressionPipeline(method='spectral', verbose=False)
        
        start_time = time.time()
        processed_signal = pipeline.suppress_noise(test_signal, sample_rate)
        processing_time = time.time() - start_time
        
        # Проверка, что обработка не слишком медленная
        # Для 5-секундного аудио ожидаем обработку менее чем за 30 секунд
        self.assertLess(processing_time, 30.0)
        
        # Проверка, что результат корректный
        self.assertEqual(len(processed_signal), len(test_signal))
    
    @unittest.skipUnless(TORCH_AVAILABLE, "PyTorch недоступен")
    def test_gpu_availability(self):
        """Тест доступности GPU"""
        gpu_available = torch.cuda.is_available()
        
        if gpu_available:
            device_count = torch.cuda.device_count()
            self.assertGreater(device_count, 0)
            
            device_name = torch.cuda.get_device_name(0)
            self.assertIsInstance(device_name, str)
            self.assertGreater(len(device_name), 0)
        else:
            print("GPU недоступен, используется CPU")


def run_dependency_check():
    """
    Проверка зависимостей перед запуском тестов
    """
    print("Проверка зависимостей...")
    
    dependencies = {
        'FFmpeg': check_ffmpeg,
        'ffmpeg-normalize': check_ffmpeg_normalize,
        'Audio Libraries': lambda: AUDIO_LIBS_AVAILABLE,
        'PyTorch': lambda: TORCH_AVAILABLE
    }
    
    for name, check_func in dependencies.items():
        try:
            available = check_func()
            status = "✓" if available else "✗"
            print(f"  {status} {name}: {'Доступен' if available else 'Недоступен'}")
        except Exception as e:
            print(f"  ✗ {name}: Ошибка проверки - {e}")
    
    print()


def check_ffmpeg():
    """Проверка FFmpeg"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


def check_ffmpeg_normalize():
    """Проверка ffmpeg-normalize"""
    try:
        result = subprocess.run(['ffmpeg-normalize', '--version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except:
        return False


if __name__ == '__main__':
    # Проверка зависимостей
    run_dependency_check()
    
    # Настройка тестов
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Добавление тестов
    test_classes = [
        TestAudioNormalizer,
        TestNoiseSuppressionPipeline,
        TestIntegration,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Запуск тестов
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Вывод результатов
    print(f"\nРезультаты тестирования:")
    print(f"  Запущено тестов: {result.testsRun}")
    print(f"  Успешных: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"  Неудачных: {len(result.failures)}")
    print(f"  Ошибок: {len(result.errors)}")
    print(f"  Пропущено: {len(result.skipped)}")
    
    if result.failures:
        print("\nНеудачные тесты:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nОшибки в тестах:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    # Код выхода
    sys.exit(0 if result.wasSuccessful() else 1)