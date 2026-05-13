"""
S02-02, S02-03, S02-04: Сохранение, загрузка, удаление моделей в MinIO
"""

import io
import joblib
import pickle
from typing import Optional, Dict, Any
from datetime import datetime
import logging

from src.storage.minio.client import get_minio_client

logger = logging.getLogger(__name__)


class ModelStorage:
    """Хранилище моделей в MinIO"""
    
    def __init__(self):
        self.client = get_minio_client()
        self.bucket = self.client.bucket
    
    def _ensure_client(self):
        if not self.client._initialized:
            self.client.initialize()
        return self.client.get_client()
    
    def save_model(
        self, 
        model, 
        model_id: str, 
        model_type: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Сохранение модели в MinIO
        
        Args:
            model: Объект модели (pickle-совместимый)
            model_id: Уникальный идентификатор модели
            model_type: Тип модели (xgboost, lstm, etc.)
            metadata: Дополнительные метаданные
            
        Returns:
            str: Путь к сохранённой модели
        """
        s3_client = self._ensure_client()
        
        path = f"models/{model_type}/{model_id}.pkl"
        
        # Сериализация модели
        buffer = io.BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        
        # Подготовка метаданных
        s3_metadata = {
            'model_id': model_id,
            'model_type': model_type,
            'saved_at': datetime.utcnow().isoformat(),
        }
        if metadata:
            s3_metadata.update({k: str(v) for k, v in metadata.items()})
        
        # Сохранение в MinIO
        try:
            s3_client.put_object(
                Bucket=self.bucket,
                Key=path,
                Body=buffer,
                ContentType='application/octet-stream',
                Metadata=s3_metadata
            )
            logger.info(f"✅ Model saved: {path}")
            return path
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {str(e)}")
            raise
    
    def load_model(self, model_id: str, model_type: str) -> Optional[object]:
        """
        Загрузка модели из MinIO
        
        Args:
            model_id: Уникальный идентификатор модели
            model_type: Тип модели
            
        Returns:
            object: Десериализованная модель или None
        """
        s3_client = self._ensure_client()
        
        path = f"models/{model_type}/{model_id}.pkl"
        
        try:
            response = s3_client.get_object(Bucket=self.bucket, Key=path)
            buffer = io.BytesIO(response['Body'].read())
            model = joblib.load(buffer)
            logger.info(f"✅ Model loaded: {path}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model {model_id}: {str(e)}")
            return None
    
    def delete_model(self, model_id: str, model_type: str) -> bool:
        """
        Удаление модели из MinIO
        
        Args:
            model_id: Уникальный идентификатор модели
            model_type: Тип модели
            
        Returns:
            bool: True если удаление успешно
        """
        s3_client = self._ensure_client()
        
        path = f"models/{model_type}/{model_id}.pkl"
        
        try:
            s3_client.delete_object(Bucket=self.bucket, Key=path)
            logger.info(f"🗑️ Model deleted: {path}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete model {model_id}: {str(e)}")
            return False
    
    def model_exists(self, model_id: str, model_type: str) -> bool:
        """Проверка существования модели"""
        s3_client = self._ensure_client()
        
        path = f"models/{model_type}/{model_id}.pkl"
        
        try:
            s3_client.head_object(Bucket=self.bucket, Key=path)
            return True
        except Exception:
            return False


# Глобальный экземпляр хранилища
model_storage = ModelStorage()


def get_model_storage() -> ModelStorage:
    """Получение экземпляра хранилища моделей"""
    return model_storage
