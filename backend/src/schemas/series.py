"""
Pydantic схемы для Series API
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Any
from datetime import datetime


class SeriesCreate(BaseModel):
    """B04-01: Создание ряда (ручной ввод)"""
    name: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    values: List[float] = Field(..., min_length=1)
    dates: Optional[List[str]] = None


class SeriesUpdate(BaseModel):
    """B04-06: Обновление метаданных ряда"""
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None


class SeriesResponse(BaseModel):
    """Ответ с информацией о ряде"""
    id: str
    user_id: str
    name: str
    description: Optional[str] = None
    length: int
    min_value: float
    max_value: float
    avg_value: float
    created_at: datetime
    updated_at: datetime


class SeriesListResponse(BaseModel):
    """Список рядов с пагинацией"""
    items: List[SeriesResponse]
    total: int
    page: int
    page_size: int


class SeriesPreviewResponse(BaseModel):
    """B04-04: Предпросмотр данных"""
    headers: List[str]
    data: List[dict]


class SeriesUploadResponse(BaseModel):
    """B04-01: Ответ после загрузки"""
    series_id: str
    name: str
    length: int
    message: str


class SeriesPlotResponse(BaseModel):
    """B04-07: График в формате base64"""
    plot: str  # base64 encoded image
    format: str = "png"
