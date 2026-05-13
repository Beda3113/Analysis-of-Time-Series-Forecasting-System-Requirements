"""
C03-06: CachedExplainer - Кэширование объяснений в Redis
"""

import hashlib
import json
from typing import Optional, Dict, Any, Callable
from datetime import datetime, timedelta
import warnings


class CachedExplainer:
    """
    Кэширование объяснений для ускорения повторных запросов.
    """
    
    def __init__(
        self,
        redis_client=None,
        ttl_seconds: int = 3600,
        use_in_memory: bool = True
    ):
        """
        Инициализация кэшированного объяснителя
        
        Args:
            redis_client: Redis клиент (если None, используется in-memory кэш)
            ttl_seconds: Время жизни кэша в секундах
            use_in_memory: Использовать in-memory кэш (если Redis недоступен)
        """
        self.redis_client = redis_client
        self.ttl_seconds = ttl_seconds
        self.use_in_memory = use_in_memory
        self._memory_cache: Dict[str, tuple] = {}
        self._has_redis = redis_client is not None
    
    def _generate_key(
        self,
        model_id: str,
        series_id: str,
        params: Dict[str, Any]
    ) -> str:
        """
        Генерация уникального ключа для кэша
        
        Args:
            model_id: ID модели
            series_id: ID ряда
            params: Дополнительные параметры
            
        Returns:
            str: Хэш-ключ
        """
        content = json.dumps({
            "model_id": model_id,
            "series_id": series_id,
            "params": params
        }, sort_keys=True)
        
        return f"explanation:{hashlib.md5(content.encode()).hexdigest()}"
    
    def get(
        self,
        model_id: str,
        series_id: str,
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Получение кэшированного объяснения
        
        Returns:
            Optional[Dict]: Объяснение или None
        """
        key = self._generate_key(model_id, series_id, params or {})
        
        # Проверка Redis
        if self._has_redis:
            try:
                cached = self.redis_client.get(key)
                if cached:
                    return json.loads(cached)
            except Exception:
                pass
        
        # Проверка in-memory кэша
        if self.use_in_memory and key in self._memory_cache:
            data, cached_at = self._memory_cache[key]
            if datetime.now() - cached_at < timedelta(seconds=self.ttl_seconds):
                return data
        
        return None
    
    def set(
        self,
        explanation: Dict[str, Any],
        model_id: str,
        series_id: str,
        params: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Сохранение объяснения в кэш
        """
        key = self._generate_key(model_id, series_id, params or {})
        
        # Сохранение в Redis
        if self._has_redis:
            try:
                self.redis_client.setex(
                    key,
                    self.ttl_seconds,
                    json.dumps(explanation, default=str)
                )
            except Exception:
                pass
        
        # Сохранение в in-memory кэш
        if self.use_in_memory:
            self._memory_cache[key] = (explanation, datetime.now())
    
    def invalidate(
        self,
        model_id: Optional[str] = None,
        series_id: Optional[str] = None
    ) -> None:
        """
        Инвалидация кэша
        """
        if model_id is None and series_id is None:
            # Полная очистка
            self._memory_cache.clear()
            # В Redis очистка не делается для простоты
        else:
            # Частичная очистка (только in-memory)
            keys_to_delete = []
            for key in self._memory_cache:
                if (model_id and model_id in key) or (series_id and series_id in key):
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                del self._memory_cache[key]
    
    def clear(self) -> None:
        """Полная очистка кэша"""
        self._memory_cache.clear()
        # В Redis очистка не делается для простоты
    
    def get_stats(self) -> Dict[str, Any]:
        """Получение статистики кэша"""
        return {
            "has_redis": self._has_redis,
            "use_in_memory": self.use_in_memory,
            "ttl_seconds": self.ttl_seconds,
            "memory_cache_size": len(self._memory_cache)
        }
