"""
B02-01..B02-06: Error Handling Module
"""

from typing import Any, Dict, Optional
from fastapi import status


# ========== B02-01: Базовое исключение ==========

class AppException(Exception):
    """Базовое исключение приложения"""
    
    def __init__(
        self,
        status_code: int = 500,
        detail: str = "Internal server error",
        error_code: str = "INTERNAL_ERROR",
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None
    ):
        self.status_code = status_code
        self.detail = detail
        self.error_code = error_code
        self.data = data
        self.headers = headers or {}
        super().__init__(detail)
    
    def to_dict(self) -> Dict[str, Any]:
        response = {
            "success": False,
            "error": {
                "code": self.error_code,
                "message": self.detail,
            }
        }
        if self.data:
            response["error"]["data"] = self.data
        return response


# ========== B02-02: 404 Not Found ==========

class NotFoundError(AppException):
    """404 - Ресурс не найден"""
    
    def __init__(self, resource: str, resource_id: str = None):
        detail = f"{resource} не найден"
        if resource_id:
            detail = f"{resource} с id '{resource_id}' не найден"
        
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail,
            error_code="RESOURCE_NOT_FOUND",
            data={"resource": resource, "resource_id": resource_id}
        )


# ========== B02-03: 422 Validation Error ==========

class ValidationError(AppException):
    """422 - Ошибка валидации данных"""
    
    def __init__(self, detail: str, field: Optional[str] = None):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail,
            error_code="VALIDATION_ERROR",
            data={"field": field} if field else None
        )


# ========== B02-04: 401 Unauthorized ==========

class UnauthorizedError(AppException):
    """401 - Не авторизован"""
    
    def __init__(self, detail: str = "Необходима авторизация"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            error_code="UNAUTHORIZED",
            headers={"WWW-Authenticate": "Bearer"}
        )


class InvalidTokenError(UnauthorizedError):
    """Недействительный токен"""
    def __init__(self):
        super().__init__("Недействительный или истёкший токен")


class TokenExpiredError(UnauthorizedError):
    """Токен истёк"""
    def __init__(self):
        super().__init__("Токен истёк. Обновите токен")


# ========== B02-05: 403 Forbidden ==========

class ForbiddenError(AppException):
    """403 - Доступ запрещён"""
    
    def __init__(self, detail: str = "Доступ запрещён"):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
            error_code="FORBIDDEN"
        )


# ========== B02-05: 409 Conflict ==========

class ConflictError(AppException):
    """409 - Конфликт (ресурс уже существует)"""
    
    def __init__(self, detail: str, resource: str = None):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
            error_code="CONFLICT",
            data={"resource": resource} if resource else None
        )


# ========== B02-05: 429 Rate Limit ==========

class RateLimitError(AppException):
    """429 - Превышен лимит запросов"""
    
    def __init__(self, retry_after: int = 60):
        super().__init__(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Превышен лимит запросов. Повторите через {retry_after} секунд",
            error_code="RATE_LIMIT_EXCEEDED",
            headers={"Retry-After": str(retry_after)}
        )


# ========== B02-05: 500 Internal Error ==========

class InternalError(AppException):
    """500 - Внутренняя ошибка сервера"""
    
    def __init__(self, detail: str = "Внутренняя ошибка сервера"):
        super().__init__(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail,
            error_code="INTERNAL_ERROR"
        )
