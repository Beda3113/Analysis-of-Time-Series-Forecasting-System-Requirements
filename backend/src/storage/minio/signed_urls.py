"""
S02-06: Подписанные URL для безопасного скачивания
"""

from datetime import datetime, timedelta
from typing import Optional
import logging

from src.storage.minio.client import get_minio_client

logger = logging.getLogger(__name__)


class SignedURLGenerator:
    """Генератор подписанных URL для безопасного доступа к файлам"""
    
    def __init__(self, expiry_minutes: int = 15):
        self.client = get_minio_client()
        self.bucket = self.client.bucket
        self.expiry_minutes = expiry_minutes
    
    def _ensure_client(self):
        if not self.client._initialized:
            self.client.initialize()
        return self.client.get_client()
    
    def generate_download_url(
        self, 
        object_path: str, 
        expiry_minutes: Optional[int] = None
    ) -> Optional[str]:
        """
        Генерация подписанной URL для скачивания
        
        Args:
            object_path: Путь к объекту в MinIO
            expiry_minutes: Время действия URL в минутах
            
        Returns:
            str: Подписанная URL или None
        """
        s3_client = self._ensure_client()
        
        expiry = expiry_minutes or self.expiry_minutes
        
        try:
            url = s3_client.generate_presigned_url(
                ClientMethod='get_object',
                Params={
                    'Bucket': self.bucket,
                    'Key': object_path
                },
                ExpiresIn=expiry * 60
            )
            logger.info(f"Generated signed URL for {object_path}, expires in {expiry} min")
            return url
        except Exception as e:
            logger.error(f"Failed to generate signed URL: {str(e)}")
            return None
    
    def generate_upload_url(
        self, 
        object_path: str, 
        expiry_minutes: Optional[int] = None
    ) -> Optional[str]:
        """
        Генерация подписанной URL для загрузки
        
        Args:
            object_path: Путь к объекту в MinIO
            expiry_minutes: Время действия URL в минутах
            
        Returns:
            str: Подписанная URL или None
        """
        s3_client = self._ensure_client()
        
        expiry = expiry_minutes or self.expiry_minutes
        
        try:
            url = s3_client.generate_presigned_url(
                ClientMethod='put_object',
                Params={
                    'Bucket': self.bucket,
                    'Key': object_path
                },
                ExpiresIn=expiry * 60
            )
            logger.info(f"Generated signed upload URL for {object_path}")
            return url
        except Exception as e:
            logger.error(f"Failed to generate signed upload URL: {str(e)}")
            return None
    
    def generate_model_download_url(
        self, 
        model_id: str, 
        model_type: str,
        expiry_minutes: Optional[int] = None
    ) -> Optional[str]:
        """
        Генерация подписанной URL для скачивания модели
        
        Args:
            model_id: ID модели
            model_type: Тип модели
            expiry_minutes: Время действия URL в минутах
            
        Returns:
            str: Подписанная URL или None
        """
        path = f"models/{model_type}/{model_id}.pkl"
        return self.generate_download_url(path, expiry_minutes)


# Глобальный экземпляр
signed_url_generator = SignedURLGenerator()


def get_signed_url_generator() -> SignedURLGenerator:
    """Получение экземпляра генератора подписанных URL"""
    return signed_url_generator
