"""
S03-02, S03-03: Кэш прогнозов и объяснений
"""

import json
import hashlib
from typing import Optional, Any, Dict, List
from datetime import timedelta
import logging

from src.storage.redis.client import get_redis_client

logger = logging.getLogger(__name__)


class ForecastCache:
    """S03-02: Кэш прогнозов forecast:{id}:{model} → JSON"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.ttl = ttl_seconds
        self.prefix = "forecast"
        self.client = get_redis_client()
    
    def _get_key(self, series_id: str, model_id: str, horizon: int) -> str:
        """Формирование ключа для кэша"""
        return f"{self.prefix}:{series_id}:{model_id}:{horizon}"
    
    def get(self, series_id: str, model_id: str, horizon: int) -> Optional[Dict]:
        """Получение прогноза из кэша"""
        try:
            redis = self.client.get_client()
            key = self._get_key(series_id, model_id, horizon)
            data = redis.get(key)
            if data:
                logger.debug(f"Cache hit: {key}")
                return json.loads(data)
            logger.debug(f"Cache miss: {key}")
            return None
        except Exception as e:
            logger.error(f"Failed to get forecast from cache: {str(e)}")
            return None
    
    def set(self, series_id: str, model_id: str, horizon: int, data: Dict) -> bool:
        """Сохранение прогноза в кэш"""
        try:
            redis = self.client.get_client()
            key = self._get_key(series_id, model_id, horizon)
            redis.setex(key, self.ttl, json.dumps(data, default=str))
            logger.debug(f"Cached forecast: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache forecast: {str(e)}")
            return False
    
    def invalidate(self, series_id: str, model_id: str = None) -> int:
        """Инвалидация кэша прогнозов"""
        try:
            redis = self.client.get_client()
            pattern = f"{self.prefix}:{series_id}:*" if model_id is None else f"{self.prefix}:{series_id}:{model_id}:*"
            keys = redis.keys(pattern)
            if keys:
                deleted = redis.delete(*keys)
                logger.info(f"Invalidated {deleted} forecast cache entries")
                return deleted
            return 0
        except Exception as e:
            logger.error(f"Failed to invalidate cache: {str(e)}")
            return 0


class ExplanationCache:
    """S03-03: Кэш объяснений qwen:{hash} → текст"""
    
    def __init__(self, ttl_seconds: int = 86400):  # 24 часа
        self.ttl = ttl_seconds
        self.prefix = "explanation"
        self.client = get_redis_client()
    
    def _generate_hash(self, model_id: str, series_id: str, lags: List[int]) -> str:
        """Генерация хэша для ключа"""
        content = f"{model_id}:{series_id}:{sorted(lags)}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_key(self, model_id: str, series_id: str, lags: List[int]) -> str:
        """Формирование ключа для кэша"""
        hash_key = self._generate_hash(model_id, series_id, lags)
        return f"{self.prefix}:qwen:{hash_key}"
    
    def get(self, model_id: str, series_id: str, lags: List[int]) -> Optional[str]:
        """Получение объяснения из кэша"""
        try:
            redis = self.client.get_client()
            key = self._get_key(model_id, series_id, lags)
            explanation = redis.get(key)
            if explanation:
                logger.debug(f"Cache hit: {key}")
                return explanation
            return None
        except Exception as e:
            logger.error(f"Failed to get explanation from cache: {str(e)}")
            return None
    
    def set(self, model_id: str, series_id: str, lags: List[int], explanation: str) -> bool:
        """Сохранение объяснения в кэш"""
        try:
            redis = self.client.get_client()
            key = self._get_key(model_id, series_id, lags)
            redis.setex(key, self.ttl, explanation)
            logger.debug(f"Cached explanation: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to cache explanation: {str(e)}")
            return False


# Глобальные экземпляры
forecast_cache = ForecastCache()
explanation_cache = ExplanationCache()


def get_forecast_cache() -> ForecastCache:
    return forecast_cache


def get_explanation_cache() -> ExplanationCache:
    return explanation_cache
