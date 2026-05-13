"""
S02-05: TTL управление - временные файлы (удаление через час)
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional
import logging

from src.storage.minio.client import get_minio_client

logger = logging.getLogger(__name__)


class TTLManager:
    """Управление временем жизни временных файлов"""
    
    def __init__(self, temp_prefix: str = "temp/"):
        self.client = get_minio_client()
        self.bucket = self.client.bucket
        self.temp_prefix = temp_prefix
    
    def _ensure_client(self):
        if not self.client._initialized:
            self.client.initialize()
        return self.client.get_client()
    
    def cleanup_temp_files(self, max_age_hours: int = 1) -> int:
        """
        Удаление временных файлов старше указанного возраста
        
        Args:
            max_age_hours: Максимальный возраст файла в часах
            
        Returns:
            int: Количество удалённых файлов
        """
        s3_client = self._ensure_client()
        cutoff_time = datetime.utcnow() - timedelta(hours=max_age_hours)
        deleted_count = 0
        
        try:
            # Список всех временных файлов
            response = s3_client.list_objects_v2(
                Bucket=self.bucket, 
                Prefix=self.temp_prefix
            )
            
            for obj in response.get('Contents', []):
                last_modified = obj['LastModified']
                if last_modified < cutoff_time:
                    s3_client.delete_object(Bucket=self.bucket, Key=obj['Key'])
                    deleted_count += 1
                    logger.debug(f"Deleted temp file: {obj['Key']}")
            
            logger.info(f"Cleaned up {deleted_count} temporary files")
            
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {str(e)}")
        
        return deleted_count
    
    def save_temp_file(
        self, 
        data: bytes, 
        file_name: str, 
        ttl_hours: int = 1
    ) -> str:
        """
        Сохранение временного файла с TTL
        
        Args:
            data: Данные файла
            file_name: Имя файла
            ttl_hours: Время жизни в часах
            
        Returns:
            str: Путь к сохранённому файлу
        """
        s3_client = self._ensure_client()
        
        # Формирование пути с датой для автоматической очистки
        date_prefix = datetime.utcnow().strftime("%Y%m%d")
        path = f"{self.temp_prefix}{date_prefix}/{file_name}"
        
        # Метаданные с TTL
        expiry_time = datetime.utcnow() + timedelta(hours=ttl_hours)
        metadata = {
            'ttl_hours': str(ttl_hours),
            'expires_at': expiry_time.isoformat(),
            'created_at': datetime.utcnow().isoformat()
        }
        
        try:
            s3_client.put_object(
                Bucket=self.bucket,
                Key=path,
                Body=data,
                Metadata=metadata
            )
            logger.info(f"Temp file saved: {path}, expires at {expiry_time}")
            return path
        except Exception as e:
            logger.error(f"Failed to save temp file: {str(e)}")
            raise
    
    def get_temp_file(self, path: str) -> Optional[bytes]:
        """Получение временного файла"""
        s3_client = self._ensure_client()
        
        try:
            response = s3_client.get_object(Bucket=self.bucket, Key=path)
            return response['Body'].read()
        except Exception as e:
            logger.error(f"Failed to get temp file {path}: {str(e)}")
            return None
    
    def delete_temp_file(self, path: str) -> bool:
        """Удаление временного файла"""
        s3_client = self._ensure_client()
        
        try:
            s3_client.delete_object(Bucket=self.bucket, Key=path)
            logger.info(f"Temp file deleted: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete temp file {path}: {str(e)}")
            return False


# Глобальный экземпляр TTL менеджера
ttl_manager = TTLManager()


def get_ttl_manager() -> TTLManager:
    """Получение экземпляра TTL менеджера"""
    return ttl_manager
