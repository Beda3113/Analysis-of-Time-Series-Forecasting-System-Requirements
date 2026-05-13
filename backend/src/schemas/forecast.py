"""
Pydantic схемы для Forecasting API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any, Dict
from datetime import datetime


class ForecastRequest(BaseModel):
    """B06-01: Запрос на прогноз"""
    model_id: Optional[str] = Field(None, description="ID модели (если не указана, используется активная)")
    horizon: int = Field(30, ge=1, le=365, description="Горизонт прогноза")
    alpha: float = Field(0.05, ge=0.01, le=0.2, description="Уровень значимости для доверительного интервала")
    include_history: bool = Field(True, description="Включить исторические данные в ответ")


class ForecastPoint(BaseModel):
    """Точка прогноза"""
    step: int
    value: float
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    date: Optional[str] = None


class ForecastResponse(BaseModel):
    """B06-01: Ответ с прогнозом"""
    series_id: str
    model_id: str
    model_type: str
    model_name: str
    horizon: int
    alpha: float
    created_at: datetime
    historical_values: Optional[List[float]] = None
    predictions: List[ForecastPoint]
    metrics: Optional[Dict[str, float]] = None


class ExportFormat(BaseModel):
    """B06-02: Формат экспорта"""
    format: str = Field("csv", description="csv или json")


class ForecastHistory(BaseModel):
    """B06-03: История прогнозов"""
    id: str
    series_id: str
    model_id: str
    created_at: datetime
    horizon: int
    predictions_summary: Dict[str, Any]


class ForecastMetricsResponse(BaseModel):
    """B06-04: Метрики качества модели"""
    model_id: str
    model_name: str
    model_type: str
    metrics: Dict[str, float]
    created_at: datetime


class BatchForecastRequest(BaseModel):
    """B06-05: Пакетный прогноз"""
    series_ids: List[str] = Field(..., description="Список ID рядов")
    model_id: Optional[str] = None
    horizon: int = Field(30, ge=1, le=365)


class BatchForecastResponse(BaseModel):
    """B06-05: Ответ на пакетный прогноз"""
    task_id: str
    total: int
    status: str
    message: str
