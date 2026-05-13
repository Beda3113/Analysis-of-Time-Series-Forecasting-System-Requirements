"""
Pydantic схемы для аутентификации (Pydantic v2)
"""

from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional
from datetime import datetime


class UserRegister(BaseModel):
    """B03-01: Регистрация пользователя"""
    email: EmailStr
    password: str = Field(..., min_length=6, max_length=100)
    name: str = Field(..., min_length=1, max_length=100)
    
    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError('Пароль должен содержать минимум 6 символов')
        return v


class UserLogin(BaseModel):
    """B03-02: Логин пользователя"""
    email: EmailStr
    password: str


class UserResponse(BaseModel):
    """Ответ с данными пользователя"""
    id: str
    email: str
    name: str
    is_active: bool
    created_at: datetime
    
    model_config = {"from_attributes": True}


class TokenResponse(BaseModel):
    """B03-02: Ответ с токенами"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


class RefreshTokenRequest(BaseModel):
    """B03-03: Запрос на обновление токена"""
    refresh_token: str


class ChangePasswordRequest(BaseModel):
    """B03-06: Смена пароля"""
    old_password: str
    new_password: str = Field(..., min_length=6)
    
    @field_validator('new_password')
    @classmethod
    def validate_new_password(cls, v: str) -> str:
        if len(v) < 6:
            raise ValueError('Новый пароль должен содержать минимум 6 символов')
        return v


class ResetPasswordRequest(BaseModel):
    """B03-07: Восстановление пароля"""
    email: EmailStr


class ResetPasswordConfirm(BaseModel):
    """B03-07: Подтверждение сброса пароля"""
    token: str
    new_password: str = Field(..., min_length=6)
