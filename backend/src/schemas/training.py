"""
Pydantic схемы для Training API
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class TrainingRequest(BaseModel):
    """B05-01: Запрос на обучение модели"""
    model_type: str = Field(..., description="Тип модели: xgboost, lstm, prophet, sarima")
    name: Optional[str] = Field(None, description="Название модели")
    horizon: int = Field(30, ge=1, le=365, description="Горизонт прогноза")
    hyperparams: Optional[Dict[str, Any]] = Field(None, description="Гиперпараметры модели")


class TrainingResponse(BaseModel):
    """B05-01: Ответ после запуска обучения"""
    task_id: str
    model_id: Optional[str] = None
    status: str
    message: str


class TrainingStatusResponse(BaseModel):
    """B05-02: Статус задачи обучения"""
    task_id: str
    status: str  # pending, training, completed, failed
    progress: int  # 0-100
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    updated_at: datetime


class ModelMetrics(BaseModel):
    """Метрики модели"""
    mae: float
    rmse: float
    mape: float
    train_size: Optional[int] = None
    val_size: Optional[int] = None


class ModelInfo(BaseModel):
    """B05-04: Информация о модели"""
    id: str
    series_id: str
    user_id: str
    model_type: str
    name: str
    hyperparams: Dict[str, Any]
    metrics: ModelMetrics
    is_active: bool
    created_at: datetime
    updated_at: datetime


class ModelListResponse(BaseModel):
    """B05-03: Список моделей"""
    items: List[ModelInfo]
    total: int


class ActivateModelRequest(BaseModel):
    """B05-06: Активация модели"""
    pass  # model_id из URL


class ModelCompareResponse(BaseModel):
    """B05-07: Сравнение моделей"""
    models: List[ModelInfo]
    best_by_mape: Optional[str] = None
    best_by_rmse: Optional[str] = None
