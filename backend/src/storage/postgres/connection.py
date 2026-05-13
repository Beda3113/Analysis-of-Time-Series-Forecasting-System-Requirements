"""
S01-01: Connection pool - asyncpg для асинхронной работы
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import sessionmaker
from typing import AsyncGenerator, Optional
import os

from src.config import settings
from src.storage.postgres.models import Base

# Глобальные переменные
engine = None
async_session_maker = None


async def init_db():
    """Инициализация подключения к БД"""
    global engine, async_session_maker
    
    database_url = settings.async_database_url
    
    engine = create_async_engine(
        database_url,
        echo=settings.debug,
        pool_size=10,
        max_overflow=20,
        pool_pre_ping=True,
    )
    
    async_session_maker = async_sessionmaker(
        engine, 
        class_=AsyncSession, 
        expire_on_commit=False
    )
    
    # Создание таблиц
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    return engine


async def close_db():
    """Закрытие соединения с БД"""
    global engine
    if engine:
        await engine.dispose()
        engine = None


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """Получение сессии для Dependency Injection"""
    if async_session_maker is None:
        await init_db()
    
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_sync_engine():
    """Получение синхронного движка для миграций"""
    from sqlalchemy import create_engine
    return create_engine(settings.database_url, echo=settings.debug)
