"""
S03-04, S03-05: Rate limiting счётчики и сессии пользователей
"""

import time
from typing import Optional, Dict, Any
from datetime import timedelta
import logging

from src.storage.redis.client import get_redis_client

logger = logging.getLogger(__name__)


class RateLimiter:
    """S03-04: Rate limiting счётчики ratelimit:{user}:{endpoint}"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
        self.window_seconds = 60
        self.prefix = "ratelimit"
        self.client = get_redis_client()
    
    def _get_key(self, user_id: str, endpoint: str) -> str:
        """Формирование ключа для rate limiter"""
        return f"{self.prefix}:{user_id}:{endpoint}"
    
    def check_and_increment(self, user_id: str, endpoint: str) -> tuple:
        """
        Проверка лимита и увеличение счётчика
        
        Returns:
            (allowed: bool, remaining: int, retry_after: int)
        """
        try:
            redis = self.client.get_client()
            key = self._get_key(user_id, endpoint)
            
            # Текущее время
            now = time.time()
            window_start = now - self.window_seconds
            
            # Удаляем старые записи
            redis.zremrangebyscore(key, 0, window_start)
            
            # Количество запросов в окне
            current_count = redis.zcard(key)
            
            if current_count >= self.requests_per_minute:
                # Находим самую старую запись для расчёта retry_after
                oldest = redis.zrange(key, 0, 0, withscores=True)
                if oldest:
                    retry_after = int(oldest[0][1] + self.window_seconds - now) + 1
                else:
                    retry_after = self.window_seconds
                return False, 0, retry_after
            
            # Добавляем текущий запрос
            redis.zadd(key, {str(now): now})
            redis.expire(key, self.window_seconds)
            
            remaining = self.requests_per_minute - current_count - 1
            return True, remaining, 0
            
        except Exception as e:
            logger.error(f"Rate limiter error: {str(e)}")
            return True, self.requests_per_minute, 0
    
    def reset(self, user_id: str, endpoint: str) -> bool:
        """Сброс счётчика для пользователя"""
        try:
            redis = self.client.get_client()
            key = self._get_key(user_id, endpoint)
            redis.delete(key)
            return True
        except Exception as e:
            logger.error(f"Failed to reset rate limiter: {str(e)}")
            return False


class UserSession:
    """S03-05: Сессии пользователей user:{id}:session"""
    
    def __init__(self, ttl_seconds: int = 3600):  # 1 час
        self.ttl = ttl_seconds
        self.prefix = "user"
        self.client = get_redis_client()
    
    def _get_key(self, user_id: str) -> str:
        return f"{self.prefix}:{user_id}:session"
    
    def set_session(self, user_id: str, session_data: Dict[str, Any]) -> bool:
        """Сохранение сессии пользователя"""
        try:
            redis = self.client.get_client()
            key = self._get_key(user_id)
            
            import json
            redis.setex(key, self.ttl, json.dumps(session_data, default=str))
            logger.debug(f"Session saved for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to save session: {str(e)}")
            return False
    
    def get_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Получение сессии пользователя"""
        try:
            redis = self.client.get_client()
            key = self._get_key(user_id)
            data = redis.get(key)
            if data:
                import json
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Failed to get session: {str(e)}")
            return None
    
    def delete_session(self, user_id: str) -> bool:
        """Удаление сессии (logout)"""
        try:
            redis = self.client.get_client()
            key = self._get_key(user_id)
            redis.delete(key)
            logger.debug(f"Session deleted for user {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete session: {str(e)}")
            return False
    
    def update_last_active(self, user_id: str) -> bool:
        """Обновление времени последней активности"""
        session = self.get_session(user_id)
        if session:
            session['last_active'] = time.time()
            return self.set_session(user_id, session)
        return False


# Глобальные экземпляры
rate_limiter = RateLimiter()
user_session = UserSession()


def get_rate_limiter() -> RateLimiter:
    return rate_limiter


def get_user_session() -> UserSession:
    return user_session
