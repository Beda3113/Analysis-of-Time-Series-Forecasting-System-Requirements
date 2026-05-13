"""
Pydantic схемы для Preprocessing API
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class AnomalyMethod(str, Enum):
    ZSCORE = "zscore"
    IQR = "iqr"
    STL = "stl"


class AnomalyDetectionRequest(BaseModel):
    """B08-01: Запрос на детекцию аномалий"""
    method: AnomalyMethod = Field(AnomalyMethod.ZSCORE, description="Метод детекции")
    threshold: Optional[float] = Field(3.0, ge=1.0, le=5.0, description="Порог для Z-score")
    window: Optional[int] = Field(30, ge=5, le=100, description="Окно для STL")


class AnomalyPoint(BaseModel):
    """Точка-аномалия"""
    index: int
    value: float
    date: Optional[str] = None
    method: str
    score: Optional[float] = None


class AnomalyDetectionResponse(BaseModel):
    """B08-01: Ответ с аномалиями"""
    series_id: str
    method: str
    anomalies: List[AnomalyPoint]
    anomaly_count: int
    anomaly_percentage: float


class FixAnomaliesRequest(BaseModel):
    """B08-02: Запрос на обработку аномалий"""
    method: str = Field("spline", description="Метод обработки: spline, median, delete")
    anomaly_indices: Optional[List[int]] = Field(None, description="Индексы аномалий (если не указаны, используются все обнаруженные)")


class FixAnomaliesResponse(BaseModel):
    """B08-02: Ответ после обработки"""
    series_id: str
    fixed_count: int
    new_series_id: str
    message: str


class StationarityTestResponse(BaseModel):
    """B08-03: Результат теста стационарности"""
    series_id: str
    adf_statistic: float
    p_value: float
    is_stationary: bool
    critical_values: Dict[str, float]
    used_lag: int
    interpretation: str


class DifferenceRequest(BaseModel):
    """B08-04: Запрос на дифференцирование"""
    order: int = Field(1, ge=1, le=3, description="Порядок дифференцирования")
    seasonal: Optional[int] = Field(None, description="Сезонный период для сезонного дифференцирования")


class DifferenceResponse(BaseModel):
    """B08-04: Ответ после дифференцирования"""
    series_id: str
    original_series_id: str
    order: int
    seasonal: Optional[int]
    new_length: int


class DecomposeResponse(BaseModel):
    """B08-05: Результат декомпозиции"""
    series_id: str
    model: str
    trend: List[float]
    seasonal: List[float]
    residual: List[float]
    observed: List[float]
    period: int


class ScaleRequest(BaseModel):
    """B08-06: Запрос на масштабирование"""
    method: str = Field("standard", description="Метод: standard, minmax, robust")
    with_mean: bool = Field(True, description="Центрировать")
    with_std: bool = Field(True, description="Нормировать на стандартное отклонение")


class ScaleResponse(BaseModel):
    """B08-06: Ответ после масштабирования"""
    series_id: str
    original_series_id: str
    method: str
    scaled_values: List[float]
    params: Dict[str, float]
