"""
Rate limiting middleware с использованием Redis
"""

from typing import Callable
from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from src.config import settings
from src.storage.redis.rate_limiter import get_rate_limiter
from src.utils.logger import get_logger

logger = get_logger("rate_limit")


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware для ограничения частоты запросов через Redis"""
    
    def __init__(self, app):
        super().__init__(app)
        self.rate_limiter = get_rate_limiter()
        # Устанавливаем лимит из настроек
        self.rate_limiter.requests_per_minute = settings.rate_limit_requests
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        if not settings.rate_limit_enabled:
            return await call_next(request)
        
        # Пропускаем health check
        if request.url.path in ["/health", "/ready", "/live", "/metrics"]:
            return await call_next(request)
        
        # Определяем ключ для rate limiting (IP + endpoint)
        client_ip = request.client.host if request.client else "unknown"
        endpoint = request.url.path
        
        # Проверка лимита
        allowed, remaining, retry_after = self.rate_limiter.check_and_increment(
            user_id=client_ip,
            endpoint=endpoint
        )
        
        if not allowed:
            logger.warning(f"Rate limit exceeded for {client_ip}:{endpoint}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": {
                        "code": "RATE_LIMIT_EXCEEDED",
                        "message": f"Too many requests. Please try again in {retry_after} seconds."
                    }
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        response = await call_next(request)
        
        # Добавляем заголовки с информацией о лимитах
        response.headers["X-RateLimit-Limit"] = str(self.rate_limiter.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(retry_after)
        
        return response
