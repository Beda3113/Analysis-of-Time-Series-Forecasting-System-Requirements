"""
C02-06: FeatureCache - Кэширование матрицы признаков
"""

import hashlib
import json
import pickle
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime, timedelta
import os


class FeatureCache:
    """
    Кэширование созданных признаков для ускорения повторных вычислений.
    
    Использует комбинацию in-memory и файлового кэширования.
    """
    
    def __init__(
        self,
        max_size_mb: int = 100,
        ttl_seconds: int = 3600,
        cache_dir: Optional[str] = None
    ):
        """
        Инициализация FeatureCache
        
        Args:
            max_size_mb: Максимальный размер кэша в MB
            ttl_seconds: Время жизни кэша в секундах
            cache_dir: Директория для файлового кэша
        """
        self.max_size_mb = max_size_mb
        self.ttl_seconds = ttl_seconds
        self.cache_dir = cache_dir or os.path.join(tempfile.gettempdir(), 'feature_cache')
        
        self._memory_cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_size_bytes = 0
        
        # Создание директории для кэша
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def _generate_key(
        self, 
        series_id: str, 
        feature_type: str, 
        params: Dict[str, Any]
    ) -> str:
        """
        Генерация уникального ключа для кэша
        
        Args:
            series_id: ID временного ряда
            feature_type: Тип признаков (lags, rolling, time)
            params: Параметры создания признаков
            
        Returns:
            str: Хэш-ключ
        """
        content = json.dumps({
            "series_id": series_id,
            "feature_type": feature_type,
            "params": params
        }, sort_keys=True)
        
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(
        self, 
        series_id: str, 
        feature_type: str, 
        params: Dict[str, Any]
    ) -> Optional[Any]:
        """
        Получение признаков из кэша
        
        Returns:
            Optional[Any]: Признаки или None, если не найдены
        """
        key = self._generate_key(series_id, feature_type, params)
        
        # Проверка in-memory кэша
        if key in self._memory_cache:
            data, cached_at = self._memory_cache[key]
            if datetime.now() - cached_at < timedelta(seconds=self.ttl_seconds):
                return data
            else:
                # Удаляем просроченный кэш
                del self._memory_cache[key]
        
        # Проверка файлового кэша
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                
                # Проверка времени жизни
                mtime = datetime.fromtimestamp(os.path.getmtime(cache_path))
                if datetime.now() - mtime < timedelta(seconds=self.ttl_seconds):
                    return data
                else:
                    os.remove(cache_path)
            except Exception:
                pass
        
        return None
    
    def set(
        self, 
        series_id: str, 
        feature_type: str, 
        params: Dict[str, Any], 
        features: Any
    ) -> None:
        """
        Сохранение признаков в кэш
        """
        key = self._generate_key(series_id, feature_type, params)
        
        # In-memory кэш
        self._memory_cache[key] = (features, datetime.now())
        
        # Файловый кэш
        cache_path = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(features, f)
        except Exception as e:
            # Если ошибка записи, просто пропускаем
            pass
    
    def invalidate(self, series_id: str) -> None:
        """
        Инвалидация кэша для конкретного ряда
        """
        # Очистка in-memory кэша
        keys_to_delete = []
        for key in self._memory_cache:
            if key.startswith(series_id) or series_id in key:
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            del self._memory_cache[key]
        
        # Очистка файлового кэша
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                cache_key = filename.replace('.pkl', '')
                if series_id in cache_key:
                    os.remove(os.path.join(self.cache_dir, filename))
    
    def clear(self) -> None:
        """Полная очистка кэша"""
        self._memory_cache.clear()
        
        for filename in os.listdir(self.cache_dir):
            if filename.endswith('.pkl'):
                os.remove(os.path.join(self.cache_dir, filename))
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша"""
        return {
            "memory_cache_size": len(self._memory_cache),
            "memory_cache_bytes": self._cache_size_bytes,
            "file_cache_dir": self.cache_dir,
            "ttl_seconds": self.ttl_seconds,
            "max_size_mb": self.max_size_mb
        }


# Импорт tempfile для функции set
import tempfile
