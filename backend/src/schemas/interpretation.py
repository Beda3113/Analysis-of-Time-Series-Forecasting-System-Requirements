"""
Pydantic схемы для Interpretation API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class SHAPValuesResponse(BaseModel):
    """B07-01: SHAP значения"""
    model_id: str
    model_type: str
    base_value: float
    feature_names: List[str]
    shap_values: List[List[float]]
    global_importance: Dict[str, float]


class FeatureImportanceResponse(BaseModel):
    """B07-02: Важность признаков"""
    model_id: str
    model_type: str
    importance: List[Dict[str, Any]]
    top_features: List[str]


class LagsExplanationRequest(BaseModel):
    """B07-03: Запрос на LLM объяснение лагов"""
    model_id: str
    language: str = Field("ru", description="Язык объяснения: ru или en")


class LagsExplanationResponse(BaseModel):
    """B07-03: Ответ с LLM объяснением"""
    task_id: str
    status: str
    message: str


class LagsExplanationResult(BaseModel):
    """Результат LLM объяснения"""
    important_lags: List[int]
    explanation: str
    confidence: Optional[float] = None


class TaskStatusResponse(BaseModel):
    """B07-04: Статус LLM задачи"""
    task_id: str
    status: str  # pending, processing, completed, failed
    progress: int
    result: Optional[LagsExplanationResult] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class TextReportResponse(BaseModel):
    """B07-05: Текстовый отчёт"""
    series_id: str
    model_id: str
    generated_at: datetime
    report: str
    sections: List[str]


class LIMERequest(BaseModel):
    """B07-06: LIME объяснение"""
    sample_index: int = Field(0, ge=0, description="Индекс точки для объяснения")
    num_features: int = Field(5, ge=1, le=10, description="Количество признаков для отображения")


class LIMEResponse(BaseModel):
    """B07-06: LIME объяснение"""
    model_id: str
    sample_index: int
    prediction: float
    explanations: List[Dict[str, Any]]
    intercept: float
