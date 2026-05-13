"""
S03-01: Клиент инициализация Redis (redis-py подключение)
"""

import redis
import json
from typing import Optional, Any, Dict, List
from datetime import timedelta
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """Клиент для работы с Redis"""
    
    def __init__(self):
        self.host = settings.redis_host
        self.port = settings.redis_port
        self.db = settings.redis_db
        self.password = settings.redis_password or None
        
        self.client: Optional[redis.Redis] = None
        self._initialized = False
    
    def initialize(self) -> 'RedisClient':
        """Инициализация подключения к Redis"""
        try:
            self.client = redis.Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                password=self.password,
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True
            )
            
            # Проверка подключения
            self.client.ping()
            
            self._initialized = True
            logger.info(f"Redis client initialized: {self.host}:{self.port}/{self.db}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis client: {str(e)}")
            raise
        
        return self
    
    def get_client(self) -> redis.Redis:
        """Получение Redis клиента"""
        if not self._initialized:
            self.initialize()
        return self.client
    
    def is_healthy(self) -> bool:
        """Проверка здоровья соединения"""
        try:
            if self.client:
                return self.client.ping()
            return False
        except Exception:
            return False
    
    def close(self) -> None:
        """Закрытие соединения"""
        if self.client:
            self.client.close()
            self._initialized = False
            logger.info("Redis connection closed")


# Глобальный экземпляр клиента
redis_client = RedisClient()


def get_redis_client() -> RedisClient:
    """Получение экземпляра Redis клиента"""
    return redis_client
