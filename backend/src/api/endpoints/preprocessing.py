"""
B08: Preprocessing API
- B08-01: POST /preprocess/anomalies
- B08-02: POST /preprocess/fix
- B08-03: GET /preprocess/stationarity
- B08-04: POST /preprocess/difference
- B08-05: GET /preprocess/decompose
- B08-06: POST /preprocess/scale
"""

import uuid
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.models import get_series_by_id, add_series, TimeSeries
from src.schemas.preprocessing import (
    AnomalyDetectionRequest, AnomalyDetectionResponse, AnomalyPoint,
    FixAnomaliesRequest, FixAnomaliesResponse,
    StationarityTestResponse, DifferenceRequest, DifferenceResponse,
    DecomposeResponse, ScaleRequest, ScaleResponse
)
from src.services.preprocessing_service import (
    detect_anomalies_zscore, detect_anomalies_iqr, detect_anomalies_stl,
    fix_anomalies_spline, fix_anomalies_median,
    adf_test, difference_series, decompose_series, scale_series
)
from src.api.dependencies import get_current_user
from src.utils.exceptions import NotFoundError, ValidationError
from src.utils.logger import get_logger

logger = get_logger("preprocessing")

router = APIRouter(prefix="/preprocess", tags=["Preprocessing"])


# ========== B08-01: Детекция аномалий ==========

@router.post("/anomalies", response_model=AnomalyDetectionResponse)
async def detect_anomalies(
    series_id: str = Query(..., description="ID временного ряда"),
    request: AnomalyDetectionRequest = None,
    current_user = Depends(get_current_user)
):
    """
    Детекция аномалий во временном ряде
    """
    if request is None:
        request = AnomalyDetectionRequest()
    
    series = get_series_by_id(series_id)
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    values = series.values
    dates = series.dates if hasattr(series, 'dates') and series.dates else None
    
    # Детекция в зависимости от метода
    if request.method == "zscore":
        anomaly_indices = detect_anomalies_zscore(values, request.threshold)
    elif request.method == "iqr":
        anomaly_indices = detect_anomalies_iqr(values)
    elif request.method == "stl":
        anomaly_indices = detect_anomalies_stl(values, request.window or 30)
    else:
        raise ValidationError(f"Неизвестный метод: {request.method}")
    
    # Формирование точек аномалий
    anomalies = []
    for idx in anomaly_indices:
        anomalies.append(AnomalyPoint(
            index=idx,
            value=values[idx],
            date=dates[idx] if dates and idx < len(dates) else None,
            method=request.method,
            score=None
        ))
    
    return AnomalyDetectionResponse(
        series_id=series_id,
        method=request.method,
        anomalies=anomalies,
        anomaly_count=len(anomalies),
        anomaly_percentage=round(len(anomalies) / len(values) * 100, 2) if values else 0
    )


# ========== B08-02: Обработка аномалий ==========

@router.post("/fix", response_model=FixAnomaliesResponse, status_code=status.HTTP_201_CREATED)
async def fix_anomalies(
    series_id: str = Query(..., description="ID временного ряда"),
    request: FixAnomaliesRequest = None,
    current_user = Depends(get_current_user)
):
    """
    Обработка аномалий (кубический сплайн, медиана, удаление)
    """
    if request is None:
        request = FixAnomaliesRequest()
    
    series = get_series_by_id(series_id)
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    values = series.values
    dates = series.dates
    
    # Если индексы не указаны, автоматически детектируем
    anomaly_indices = request.anomaly_indices
    if not anomaly_indices:
        anomaly_indices = detect_anomalies_zscore(values, threshold=3.0)
    
    if not anomaly_indices:
        return FixAnomaliesResponse(
            series_id=series_id,
            fixed_count=0,
            new_series_id=series_id,
            message="Аномалии не обнаружены"
        )
    
    # Обработка в зависимости от метода
    if request.method == "spline":
        fixed_values = fix_anomalies_spline(values, anomaly_indices)
    elif request.method == "median":
        fixed_values = fix_anomalies_median(values, anomaly_indices)
    elif request.method == "delete":
        fixed_values = [values[i] for i in range(len(values)) if i not in anomaly_indices]
        # Обновляем даты
        if dates:
            dates = [dates[i] for i in range(len(dates)) if i not in anomaly_indices]
    else:
        raise ValidationError(f"Неизвестный метод: {request.method}")
    
    # Создаём новый временной ряд
    new_series = TimeSeries(
        user_id=current_user.id,
        name=f"{series.name} (очищенный)",
        values=fixed_values,
        dates=dates if dates else None,
        description=f"Обработка аномалий методом {request.method}"
    )
    
    add_series(new_series)
    
    logger.info(f"Fixed {len(anomaly_indices)} anomalies in series {series_id}, new series: {new_series.id}")
    
    return FixAnomaliesResponse(
        series_id=series_id,
        fixed_count=len(anomaly_indices),
        new_series_id=new_series.id,
        message=f"Обработано {len(anomaly_indices)} аномалий. Создан новый ряд: {new_series.name}"
    )


# ========== B08-03: Тест стационарности ==========

@router.get("/stationarity", response_model=StationarityTestResponse)
async def test_stationarity(
    series_id: str = Query(..., description="ID временного ряда"),
    current_user = Depends(get_current_user)
):
    """
    Тест Дики-Фуллера на стационарность
    """
    series = get_series_by_id(series_id)
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    result = adf_test(series.values)
    
    return StationarityTestResponse(
        series_id=series_id,
        **result
    )


# ========== B08-04: Дифференцирование ==========

@router.post("/difference", response_model=DifferenceResponse)
async def difference_series_endpoint(
    series_id: str = Query(..., description="ID временного ряда"),
    request: DifferenceRequest = None,
    current_user = Depends(get_current_user)
):
    """
    Дифференцирование временного ряда для приведения к стационарности
    """
    if request is None:
        request = DifferenceRequest()
    
    series = get_series_by_id(series_id)
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    diff_values = difference_series(series.values, request.order, request.seasonal)
    
    # Создаём новый ряд
    new_series = TimeSeries(
        user_id=current_user.id,
        name=f"{series.name} (дифференцированный)",
        values=diff_values,
        dates=series.dates[request.order:] if series.dates and request.order <= len(series.dates) else None,
        description=f"Дифференцирование порядка {request.order}"
    )
    
    add_series(new_series)
    
    return DifferenceResponse(
        series_id=new_series.id,
        original_series_id=series_id,
        order=request.order,
        seasonal=request.seasonal,
        new_length=len(diff_values)
    )


# ========== B08-05: Декомпозиция ==========

@router.get("/decompose", response_model=DecomposeResponse)
async def decompose_series_endpoint(
    series_id: str = Query(..., description="ID временного ряда"),
    period: int = Query(7, ge=2, le=365, description="Период сезонности"),
    current_user = Depends(get_current_user)
):
    """
    STL-подобная декомпозиция временного ряда на тренд, сезонность и остатки
    """
    series = get_series_by_id(series_id)
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    result = decompose_series(series.values, period)
    
    return DecomposeResponse(
        series_id=series_id,
        model="additive",
        trend=result["trend"],
        seasonal=result["seasonal"],
        residual=result["residual"],
        observed=result["observed"],
        period=result["period"]
    )


# ========== B08-06: Масштабирование ==========

@router.post("/scale", response_model=ScaleResponse)
async def scale_series_endpoint(
    series_id: str = Query(..., description="ID временного ряда"),
    request: ScaleRequest = None,
    current_user = Depends(get_current_user)
):
    """
    Масштабирование временного ряда (Standard, MinMax, Robust)
    """
    if request is None:
        request = ScaleRequest()
    
    series = get_series_by_id(series_id)
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    scaled_values, params = scale_series(series.values, request.method)
    
    # Создаём новый масштабированный ряд
    new_series = TimeSeries(
        user_id=current_user.id,
        name=f"{series.name} (масштабированный)",
        values=scaled_values,
        dates=series.dates,
        description=f"Масштабирование методом {request.method}"
    )
    
    add_series(new_series)
    
    return ScaleResponse(
        series_id=new_series.id,
        original_series_id=series_id,
        method=request.method,
        scaled_values=scaled_values,
        params=params
    )
