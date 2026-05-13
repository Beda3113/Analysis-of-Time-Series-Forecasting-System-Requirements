"""
Pydantic схемы для External Models API
"""

from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ExternalModelType(str, Enum):
    XGBOOST = "xgboost"
    SKLEARN = "sklearn"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    ONNX = "onnx"
    CUSTOM = "custom"


class ExternalModelUploadRequest(BaseModel):
    """B09-01: Запрос на загрузку внешней модели"""
    name: str = Field(..., min_length=1, max_length=100)
    model_type: ExternalModelType
    framework_version: Optional[str] = None
    description: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ExternalModelUploadResponse(BaseModel):
    """B09-01: Ответ после загрузки"""
    model_id: str
    name: str
    model_type: str
    file_name: str
    file_size: int
    created_at: datetime
    message: str


class ExternalModelInfo(BaseModel):
    """B09-02: Информация о внешней модели"""
    id: str
    user_id: str
    name: str
    model_type: str
    framework_version: Optional[str]
    description: Optional[str]
    file_name: str
    file_size: int
    is_active: bool
    metadata: Dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ExternalModelListResponse(BaseModel):
    """B09-02: Список внешних моделей"""
    items: List[ExternalModelInfo]
    total: int


class ExternalModelValidateRequest(BaseModel):
    """B09-04: Запрос на валидацию модели"""
    model_type: ExternalModelType
    check_methods: List[str] = Field(default=["predict", "predict_proba"])


class ExternalModelValidateResponse(BaseModel):
    """B09-04: Ответ на валидацию"""
    is_valid: bool
    model_type: str
    checked_methods: List[str]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class ExternalForecastRequest(BaseModel):
    """B09-05: Запрос на прогноз через внешнюю модель"""
    series_id: str
    horizon: int = Field(30, ge=1, le=365)
    features: Optional[List[List[float]]] = Field(None, description="Признаки для прогноза")
    use_cache: bool = Field(True)


class ExternalForecastResponse(BaseModel):
    """B09-05: Ответ на прогноз через внешнюю модель"""
    model_id: str
    model_name: str
    series_id: str
    horizon: int
    predictions: List[float]
    confidence_intervals: Optional[Dict[str, List[float]]] = None
    execution_time_ms: float
    cached: bool
