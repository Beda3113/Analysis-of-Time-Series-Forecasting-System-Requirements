"""
Модели данных (временно in-memory, позже заменим на PostgreSQL)
"""

from datetime import datetime
from typing import Dict, Optional, List, Any
import uuid


class User:
    """Модель пользователя"""
    
    def __init__(self, email: str, name: str, hashed_password: str):
        self.id = str(uuid.uuid4())
        self.email = email
        self.name = name
        self.hashed_password = hashed_password
        self.is_active = True
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


class TimeSeries:
    """Модель временного ряда"""
    
    def __init__(
        self, 
        user_id: str, 
        name: str, 
        values: List[float],
        dates: Optional[List[str]] = None,
        description: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.user_id = user_id
        self.name = name
        self.description = description
        self.values = values
        self.dates = dates or [str(i) for i in range(len(values))]
        self.length = len(values)
        self.min_value = min(values) if values else 0
        self.max_value = max(values) if values else 0
        self.avg_value = sum(values) / len(values) if values else 0
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "name": self.name,
            "description": self.description,
            "length": self.length,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "avg_value": self.avg_value,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }
    
    def to_preview(self, rows: int = 20) -> dict:
        """Предпросмотр данных"""
        preview_values = self.values[:rows]
        preview_dates = self.dates[:rows] if self.dates else [str(i) for i in range(rows)]
        return {
            "headers": ["index", "date", "value"],
            "data": [
                {"index": i, "date": preview_dates[i], "value": preview_values[i]}
                for i in range(len(preview_values))
            ]
        }


# ========== In-memory хранилища ==========

users_db: Dict[str, User] = {}  # email -> User
refresh_tokens_db: Dict[str, str] = {}  # token -> user_id
series_db: Dict[str, TimeSeries] = {}  # series_id -> TimeSeries


# ========== User CRUD ==========

def add_user(user: User) -> User:
    users_db[user.email] = user
    return user


def get_user_by_email(email: str) -> Optional[User]:
    return users_db.get(email)


def get_user_by_id(user_id: str) -> Optional[User]:
    for user in users_db.values():
        if user.id == user_id:
            return user
    return None


def save_refresh_token(token: str, user_id: str) -> None:
    refresh_tokens_db[token] = user_id


def get_user_id_by_refresh_token(token: str) -> Optional[str]:
    return refresh_tokens_db.get(token)


def delete_refresh_token(token: str) -> None:
    if token in refresh_tokens_db:
        del refresh_tokens_db[token]


# ========== Series CRUD ==========

def add_series(series: TimeSeries) -> TimeSeries:
    series_db[series.id] = series
    return series


def get_series_by_id(series_id: str) -> Optional[TimeSeries]:
    return series_db.get(series_id)


def get_series_by_user(user_id: str) -> List[TimeSeries]:
    return [s for s in series_db.values() if s.user_id == user_id]


def update_series(series_id: str, **kwargs) -> Optional[TimeSeries]:
    series = series_db.get(series_id)
    if series:
        for key, value in kwargs.items():
            if hasattr(series, key):
                setattr(series, key, value)
        series.updated_at = datetime.utcnow()
    return series


def delete_series(series_id: str) -> bool:
    if series_id in series_db:
        del series_db[series_id]
        return True
    return False

# ========== Training Models ==========

class TrainedModel:
    """Модель, обученная на временном ряде"""
    
    def __init__(
        self,
        series_id: str,
        user_id: str,
        model_type: str,
        name: str,
        hyperparams: dict,
        metrics: dict,
        file_path: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())
        self.series_id = series_id
        self.user_id = user_id
        self.model_type = model_type  # xgboost, lstm, prophet, sarima
        self.name = name
        self.hyperparams = hyperparams
        self.metrics = metrics
        self.file_path = file_path
        self.is_active = False
        self.created_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "series_id": self.series_id,
            "user_id": self.user_id,
            "model_type": self.model_type,
            "name": self.name,
            "hyperparams": self.hyperparams,
            "metrics": self.metrics,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }


# Хранилище моделей
models_db: Dict[str, TrainedModel] = {}  # model_id -> TrainedModel
series_models_db: Dict[str, List[str]] = {}  # series_id -> [model_ids]

# Хранилище статусов задач (in-memory)
task_status_db: Dict[str, dict] = {}  # task_id -> {status, result, progress}


def add_model(model: TrainedModel) -> TrainedModel:
    models_db[model.id] = model
    if model.series_id not in series_models_db:
        series_models_db[model.series_id] = []
    series_models_db[model.series_id].append(model.id)
    return model


def get_model(model_id: str) -> Optional[TrainedModel]:
    return models_db.get(model_id)


def get_models_by_series(series_id: str) -> List[TrainedModel]:
    model_ids = series_models_db.get(series_id, [])
    return [models_db[mid] for mid in model_ids if mid in models_db]


def delete_model(model_id: str) -> bool:
    model = models_db.get(model_id)
    if model:
        # Удаляем из списка серии
        if model.series_id in series_models_db:
            if model_id in series_models_db[model.series_id]:
                series_models_db[model.series_id].remove(model_id)
        del models_db[model_id]
        return True
    return False


def activate_model(series_id: str, model_id: str) -> bool:
    """Деактивирует все модели серии и активирует указанную"""
    models = get_models_by_series(series_id)
    for model in models:
        model.is_active = (model.id == model_id)
    return True


def update_task_status(task_id: str, status: str, progress: int = 0, result: dict = None, error: str = None):
    task_status_db[task_id] = {
        "status": status,
        "progress": progress,
        "result": result,
        "error": error,
        "updated_at": datetime.utcnow()
    }


def get_task_status(task_id: str) -> Optional[dict]:
    return task_status_db.get(task_id)
