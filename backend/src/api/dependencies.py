"""
Зависимости для API эндпоинтов
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from src.utils.jwt import decode_token, get_token_type
from src.utils.exceptions import UnauthorizedError, InvalidTokenError, TokenExpiredError
from src.storage.postgres.connection import get_session
from src.storage.postgres.crud import UserCRUD
from src.models import User  # Оставляем для типа, но уберём позже

# Security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_session)
) -> User:
    """
    Получение текущего аутентифицированного пользователя
    """
    if not credentials:
        raise UnauthorizedError("Отсутствует токен авторизации")
    
    token = credentials.credentials
    
    # Декодируем токен
    payload = decode_token(token)
    if not payload:
        raise InvalidTokenError()
    
    # Проверяем тип токена
    token_type = payload.get("type")
    if token_type != "access":
        raise InvalidTokenError()
    
    # Проверяем срок действия
    # (JWT библиотека уже проверила exp)
    
    # Получаем пользователя
    user_id = payload.get("sub")
    if not user_id:
        raise InvalidTokenError()
    
    # ИЗМЕНЕНО: используем PostgreSQL вместо in-memory
    user = await UserCRUD.get_by_id(db, user_id)
    if not user:
        raise InvalidTokenError()
    
    if not user.is_active:
        raise UnauthorizedError("Пользователь деактивирован")
    
    return user


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: AsyncSession = Depends(get_session)
) -> Optional[User]:
    """
    Получение текущего пользователя (опционально)
    """
    if not credentials:
        return None
    
    try:
        return await get_current_user(credentials, db)
    except UnauthorizedError:
        return None