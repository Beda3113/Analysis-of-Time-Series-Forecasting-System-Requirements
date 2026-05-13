"""
S02-01: Клиент инициализация MinIO (boto3 setup, bucket creation)
"""

import os
import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from typing import Optional, Dict, Any
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class MinIOClient:
    """Клиент для работы с MinIO/S3 хранилищем"""
    
    def __init__(self):
        self.endpoint = settings.minio_endpoint
        self.access_key = settings.minio_access_key
        self.secret_key = settings.minio_secret_key
        self.bucket = settings.minio_bucket
        self.secure = settings.minio_secure
        
        self.client = None
        self._initialized = False
    
    def initialize(self) -> 'MinIOClient':
        """Инициализация клиента и создание bucket"""
        try:
            self.client = boto3.client(
                's3',
                endpoint_url=f"{'https' if self.secure else 'http'}://{self.endpoint}",
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                config=Config(
                    signature_version='s3v4',
                    connect_timeout=5,
                    read_timeout=30,
                    retries={'max_attempts': 3}
                )
            )
            
            # Проверка/создание bucket
            self._ensure_bucket()
            
            self._initialized = True
            logger.info(f"MinIO client initialized: endpoint={self.endpoint}, bucket={self.bucket}")
            
        except Exception as e:
            logger.error(f"Failed to initialize MinIO client: {str(e)}")
            raise
        
        return self
    
    def _ensure_bucket(self) -> None:
        """Проверка существования bucket и создание при необходимости"""
        try:
            self.client.head_bucket(Bucket=self.bucket)
            logger.info(f"Bucket '{self.bucket}' already exists")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == '404':
                # Bucket не существует, создаём
                try:
                    self.client.create_bucket(Bucket=self.bucket)
                    logger.info(f"Bucket '{self.bucket}' created successfully")
                except Exception as create_error:
                    logger.error(f"Failed to create bucket: {str(create_error)}")
                    raise
            else:
                logger.error(f"Error checking bucket: {str(e)}")
                raise
    
    def get_client(self):
        """Получение boto3 клиента"""
        if not self._initialized:
            self.initialize()
        return self.client
    
    def is_healthy(self) -> bool:
        """Проверка здоровья соединения"""
        try:
            self.client.head_bucket(Bucket=self.bucket)
            return True
        except Exception:
            return False


# Глобальный экземпляр клиента
minio_client = MinIOClient()


def get_minio_client() -> MinIOClient:
    """Получение экземпляра клиента MinIO"""
    return minio_client
