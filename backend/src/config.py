"""
Конфигурация приложения
"""

from pydantic_settings import BaseSettings
from typing import List
import json


class Settings(BaseSettings):
    app_name: str = "Time Series Forecasting API"
    app_version: str = "1.0.0"
    debug: bool = True
    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    cors_origins_raw: str = '["http://localhost:3000", "http://localhost:5173", "http://localhost:8000"]'
    
    @property
    def cors_origins(self) -> List[str]:
        try:
            return json.loads(self.cors_origins_raw)
        except:
            return ["http://localhost:3000", "http://localhost:5173", "http://localhost:8000"]
    
    log_level: str = "INFO"
    log_format: str = "text"
    
    db_host: str = "postgres"
    db_port: int = 5432
    db_user: str = "postgres"
    db_pass: str = "postgres"
    db_name: str = "timeseries"
    
    @property
    def database_url(self) -> str:
        return f"postgresql://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    @property
    def async_database_url(self) -> str:
        return f"postgresql+asyncpg://{self.db_user}:{self.db_pass}@{self.db_host}:{self.db_port}/{self.db_name}"
    
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_db: int = 0
    
    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "timeseries-models"
    minio_secure: bool = False
    
    jwt_secret_key: str = "test-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expiration_minutes: int = 30
    
    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()


def get_settings() -> Settings:
    return settings
