"""
S01-04: CRUD операции для всех таблиц
"""

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

from src.storage.postgres.models import (
    User, TimeSeries, TrainedModel, Forecast, RefreshToken
)


class UserCRUD:
    @staticmethod
    async def create(db: AsyncSession, email: str, name: str, hashed_password: str) -> User:
        user = User(
            id=str(uuid.uuid4()),
            email=email,
            name=name,
            hashed_password=hashed_password
        )
        db.add(user)
        await db.flush()
        return user
    
    @staticmethod
    async def get_by_email(db: AsyncSession, email: str) -> Optional[User]:
        result = await db.execute(select(User).where(User.email == email))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_by_id(db: AsyncSession, user_id: str) -> Optional[User]:
        result = await db.execute(select(User).where(User.id == user_id))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def update(db: AsyncSession, user_id: str, **kwargs) -> Optional[User]:
        await db.execute(update(User).where(User.id == user_id).values(**kwargs))
        await db.flush()
        return await UserCRUD.get_by_id(db, user_id)


class RefreshTokenCRUD:
    @staticmethod
    async def create(db: AsyncSession, token: str, user_id: str) -> RefreshToken:
        refresh_token = RefreshToken(
            token=token,
            user_id=user_id,
            expires_at=datetime.utcnow()
        )
        db.add(refresh_token)
        await db.flush()
        return refresh_token
    
    @staticmethod
    async def get_user_id(db: AsyncSession, token: str) -> Optional[str]:
        result = await db.execute(
            select(RefreshToken.user_id).where(
                RefreshToken.token == token,
                RefreshToken.expires_at > datetime.utcnow()
            )
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def delete(db: AsyncSession, token: str) -> None:
        await db.execute(delete(RefreshToken).where(RefreshToken.token == token))
        await db.flush()
    
    @staticmethod
    async def delete_expired(db: AsyncSession) -> int:
        result = await db.execute(
            delete(RefreshToken).where(RefreshToken.expires_at <= datetime.utcnow())
        )
        await db.flush()
        return result.rowcount
