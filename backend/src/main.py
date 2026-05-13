"""
B01-01: FastAPI инициализация
B01-07: Health check endpoints
"""

from contextlib import asynccontextmanager
from typing import AsyncGenerator
import time
import uuid
import logging

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from src.config import settings
from src.api.router import register_routers
from src.storage.postgres.connection import init_db, close_db

# Настройка простого логгера
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Управление жизненным циклом"""
    logger.info(f"🚀 Запуск {settings.app_name} v{settings.app_version}")
    logger.info(f"🌐 CORS разрешён для: {settings.cors_origins}")
    
    # Инициализация базы данных при запуске
    try:
        await init_db()
        logger.info(" База данных инициализирована")
    except Exception as e:
        logger.error(f" Ошибка инициализации БД: {str(e)}")
    
    yield  # Приложение работает
    
    # Закрытие соединения с БД при остановке
    try:
        await close_db()
        logger.info(" Соединение с БД закрыто")
    except Exception as e:
        logger.error(f" Ошибка закрытия БД: {str(e)}")
    
    logger.info(" Остановка сервера...")


# Создание приложения
app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    start_time = time.time()
    
    response = await call_next(request)
    
    duration = time.time() - start_time
    response.headers["X-Request-ID"] = request_id
    logger.info(f"{request.method} {request.url.path} → {response.status_code} ({duration:.3f}s)")
    
    return response


# Health checks
@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "version": settings.app_version}


@app.get("/ready", tags=["System"])
async def readiness_check():
    return {"status": "ready"}


@app.get("/live", tags=["System"])
async def liveness_check():
    return {"status": "alive"}


@app.get("/metrics", tags=["System"])
async def metrics():
    """Prometheus метрики"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/", tags=["System"])
async def root():
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "status": "running"
    }


# Регистрация API роутеров
register_routers(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host=settings.host, port=settings.port, reload=settings.reload)