"""
C02-05: OutOfCoreProcessor - Обработка больших данных (>2GB) с использованием Dask
"""

import os
import tempfile
from typing import List, Optional, Union, Dict, Any, Callable
import numpy as np


class OutOfCoreProcessor:
    """
    Процессор для работы с данными, не помещающимися в память (>2GB).
    
    Использует Dask для out-of-core вычислений или разбивает данные на чанки.
    """
    
    def __init__(
        self,
        chunk_size: int = 10000,
        use_dask: bool = False,
        temp_dir: Optional[str] = None
    ):
        """
        Инициализация OutOfCoreProcessor
        
        Args:
            chunk_size: Размер чанка в строках
            use_dask: Использовать Dask (если установлен)
            temp_dir: Директория для временных файлов
        """
        self.chunk_size = chunk_size
        self.use_dask = use_dask and self._has_dask()
        self.temp_dir = temp_dir or tempfile.gettempdir()
        
        self._dask_client = None
    
    def _has_dask(self) -> bool:
        """Проверка наличия Dask"""
        try:
            import dask
            return True
        except ImportError:
            return False
    
    def _get_dask(self):
        """Импорт Dask при необходимости"""
        if not self._has_dask():
            raise ImportError("Dask не установлен. Установите: pip install dask[dataframe]")
        
        import dask.dataframe as dd
        return dd
    
    def process_in_chunks(
        self,
        data,
        processor_func: Callable,
        **kwargs
    ):
        """
        Обработка данных по чанкам
        
        Args:
            data: Входные данные (список, массив или DataFrame)
            processor_func: Функция для обработки одного чанка
            **kwargs: Дополнительные аргументы для processor_func
            
        Returns:
            List: Результаты обработки всех чанков
        """
        results = []
        n_chunks = (len(data) + self.chunk_size - 1) // self.chunk_size
        
        for i in range(n_chunks):
            start = i * self.chunk_size
            end = min((i + 1) * self.chunk_size, len(data))
            chunk = data[start:end]
            
            result = processor_func(chunk, **kwargs)
            results.append(result)
        
        return results
    
    def create_lag_features_chunked(
        self,
        values: List[float],
        lags: List[int],
        chunk_size: Optional[int] = None
    ) -> List[List[float]]:
        """
        Создание лаговых признаков с обработкой по чанкам
        
        Args:
            values: Временной ряд
            lags: Список лагов
            chunk_size: Размер чанка (опционально)
            
        Returns:
            List[List[float]]: Матрица признаков
        """
        chunk_sz = chunk_size or self.chunk_size
        max_lag = max(lags)
        n_samples = len(values) - max_lag
        
        if n_samples <= chunk_sz:
            # Данные маленькие, обрабатываем целиком
            X = []
            for i in range(max_lag, len(values)):
                features = [values[i - lag] for lag in lags]
                X.append(features)
            return X
        
        # Обработка по чанкам
        all_features = []
        for start in range(0, n_samples, chunk_sz):
            end = min(start + chunk_sz, n_samples)
            chunk_features = []
            
            for i in range(start, end):
                features = [values[max_lag + i - lag] for lag in lags]
                chunk_features.append(features)
            
            all_features.extend(chunk_features)
        
        return all_features
    
    def using_dask(self) -> bool:
        """Возвращает True, если используется Dask"""
        return self.use_dask
    
    def get_metadata(self) -> Dict[str, Any]:
        """Получение метаданных процессора"""
        return {
            "creator": "OutOfCoreProcessor",
            "chunk_size": self.chunk_size,
            "use_dask": self.use_dask,
            "temp_dir": self.temp_dir,
            "dask_available": self._has_dask()
        }
