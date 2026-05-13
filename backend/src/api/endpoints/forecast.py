"""
B06: Forecasting API
- B06-01: GET /forecast/{series_id}
- B06-02: GET /forecast/{series_id}/export
- B06-03: GET /forecast/history/{series_id}
- B06-04: GET /forecast/metrics/{model_id}
- B06-05: POST /forecast/batch
"""

import uuid
import io
import asyncio
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Response, status, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.postgres.connection import get_session
from src.storage.postgres.crud import TimeSeriesCRUD, TrainedModelCRUD, ForecastCRUD
from src.schemas.forecast import (
    ForecastRequest, ForecastResponse, ForecastPoint,
    ForecastHistory, ForecastMetricsResponse, BatchForecastRequest,
    BatchForecastResponse
)
from src.services.forecast_service import get_model_predictions
from src.api.dependencies import get_current_user
from src.utils.exceptions import NotFoundError, ValidationError
from src.utils.logger import get_logger

logger = get_logger("forecast")

router = APIRouter(prefix="/forecast", tags=["Forecasting"])


# ========== B06-01: Получение прогноза ==========

@router.get("/{series_id}", response_model=ForecastResponse)
async def get_forecast(
    series_id: str,
    model_id: Optional[str] = Query(None, description="ID модели (если не указана, используется активная)"),
    horizon: int = Query(30, ge=1, le=365, description="Горизонт прогноза"),
    alpha: float = Query(0.05, ge=0.01, le=0.2, description="Уровень значимости"),
    include_history: bool = Query(True, description="Включить исторические данные"),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """Получение прогноза для временного ряда"""
    
    # ИЗМЕНЕНО: используем CRUD вместо in-memory
    series = await TimeSeriesCRUD.get_by_id(db, series_id)
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    # Выбор модели
    model = None
    if model_id:
        model = await TrainedModelCRUD.get_by_id(db, model_id)
        if not model:
            raise NotFoundError("Модель", model_id)
        if model.series_id != series_id:
            raise HTTPException(status_code=400, detail="Модель не принадлежит этому ряду")
    else:
        models = await TrainedModelCRUD.get_by_series(db, series_id)
        active_models = [m for m in models if m.is_active]
        if active_models:
            model = active_models[0]
        elif models:
            model = models[0]
        else:
            raise ValidationError("Нет обученных моделей для этого ряда")
    
    # Получение прогноза (используем оригинальные значения из series)
    predictions, lower_bounds, upper_bounds = get_model_predictions(series, model, horizon, alpha)
    
    # Формирование точек прогноза
    forecast_points = []
    for i in range(horizon):
        point = ForecastPoint(
            step=i + 1,
            value=predictions[i],
            lower_bound=lower_bounds[i] if lower_bounds else None,
            upper_bound=upper_bounds[i] if upper_bounds else None,
            date=None
        )
        forecast_points.append(point)
    
    # ИЗМЕНЕНО: сохраняем прогноз в PostgreSQL
    forecast_record = await ForecastCRUD.create(
        db=db,
        series_id=series_id,
        model_id=model.id,
        horizon=horizon,
        predictions=predictions,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        alpha=alpha
    )
    
    return ForecastResponse(
        series_id=series_id,
        model_id=model.id,
        model_type=model.model_type,
        model_name=model.name,
        horizon=horizon,
        alpha=alpha,
        created_at=forecast_record.created_at,
        historical_values=series.values if include_history else None,
        predictions=forecast_points,
        metrics=model.metrics if hasattr(model, 'metrics') else None
    )


# ========== B06-02: Экспорт прогноза в CSV ==========

@router.get("/{series_id}/export")
async def export_forecast(
    series_id: str,
    format: str = Query("csv", pattern="^(csv|json)$", description="Формат экспорта"),
    model_id: Optional[str] = Query(None),
    horizon: int = Query(30, ge=1, le=365),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """Экспорт прогноза в CSV или JSON"""
    
    series = await TimeSeriesCRUD.get_by_id(db, series_id)
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    # Выбор модели
    model = None
    if model_id:
        model = await TrainedModelCRUD.get_by_id(db, model_id)
    else:
        models = await TrainedModelCRUD.get_by_series(db, series_id)
        active_models = [m for m in models if m.is_active]
        model = active_models[0] if active_models else (models[0] if models else None)
    
    if not model:
        raise ValidationError("Нет модели для прогноза")
    
    # Получение прогноза
    predictions, lower_bounds, upper_bounds = get_model_predictions(series, model, horizon)
    
    if format == "csv":
        output = io.StringIO()
        output.write("step,value,lower_bound,upper_bound\n")
        for i in range(horizon):
            output.write(f"{i+1},{predictions[i]},{lower_bounds[i]},{upper_bounds[i]}\n")
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=forecast_{series_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
    else:
        return {
            "series_id": series_id,
            "model_id": model.id,
            "model_name": model.name,
            "created_at": datetime.utcnow().isoformat(),
            "horizon": horizon,
            "predictions": [
                {"step": i+1, "value": predictions[i], "lower_bound": lower_bounds[i], "upper_bound": upper_bounds[i]}
                for i in range(horizon)
            ]
        }


# ========== B06-03: История прогнозов ==========

@router.get("/history/{series_id}", response_model=List[ForecastHistory])
async def get_forecast_history(
    series_id: str,
    limit: int = Query(50, ge=1, le=100),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """Получение истории прогнозов для ряда"""
    
    series = await TimeSeriesCRUD.get_by_id(db, series_id)
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    # ИЗМЕНЕНО: используем CRUD вместо in-memory списка
    history = await ForecastCRUD.get_by_series(db, series_id, limit)
    
    return [
        ForecastHistory(
            id=h.id,
            series_id=h.series_id,
            model_id=h.model_id,
            created_at=h.created_at,
            horizon=h.horizon,
            predictions_summary={
                "first": h.predictions[0] if h.predictions else None,
                "last": h.predictions[-1] if h.predictions else None,
                "min": min(h.predictions) if h.predictions else None,
                "max": max(h.predictions) if h.predictions else None
            }
        )
        for h in history
    ]


# ========== B06-04: Метрики качества модели ==========

@router.get("/metrics/{model_id}", response_model=ForecastMetricsResponse)
async def get_model_metrics(
    model_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """Получение метрик качества модели"""
    
    model = await TrainedModelCRUD.get_by_id(db, model_id)
    if not model:
        raise NotFoundError("Модель", model_id)
    
    series = await TimeSeriesCRUD.get_by_id(db, model.series_id)
    if series and series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    return ForecastMetricsResponse(
        model_id=model.id,
        model_name=model.name,
        model_type=model.model_type,
        metrics=model.metrics,
        created_at=model.created_at
    )


# ========== B06-05: Пакетный прогноз ==========

@router.post("/batch", response_model=BatchForecastResponse, status_code=status.HTTP_202_ACCEPTED)
async def batch_forecast(
    request: BatchForecastRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """Пакетный прогноз для нескольких рядов"""
    
    valid_series_ids = []
    for series_id in request.series_ids:
        series = await TimeSeriesCRUD.get_by_id(db, series_id)
        if series and series.user_id == current_user.id:
            valid_series_ids.append(series_id)
    
    if not valid_series_ids:
        raise ValidationError("Нет доступных рядов для прогноза")
    
    task_id = str(uuid.uuid4())
    
    # Передаём db в фоновую задачу
    background_tasks.add_task(
        run_batch_forecast,
        task_id=task_id,
        series_ids=valid_series_ids,
        model_id=request.model_id,
        horizon=request.horizon,
        user_id=current_user.id,
        db=db
    )
    
    return BatchForecastResponse(
        task_id=task_id,
        total=len(valid_series_ids),
        status="pending",
        message=f"Пакетный прогноз запущен для {len(valid_series_ids)} рядов"
    )


async def run_batch_forecast(
    task_id: str,
    series_ids: List[str],
    model_id: Optional[str],
    horizon: int,
    user_id: str,
    db: AsyncSession
):
    """Фоновая задача для пакетного прогноза"""
    from src.models import update_task_status, get_task_status
    from src.services.forecast_service import get_model_predictions
    
    try:
        update_task_status(task_id, "processing", 0)
        
        results = []
        total = len(series_ids)
        
        for i, series_id in enumerate(series_ids):
            # ИЗМЕНЕНО: используем CRUD
            series = await TimeSeriesCRUD.get_by_id(db, series_id)
            if not series:
                continue
            
            model = None
            if model_id:
                model = await TrainedModelCRUD.get_by_id(db, model_id)
            else:
                models = await TrainedModelCRUD.get_by_series(db, series_id)
                active_models = [m for m in models if m.is_active]
                model = active_models[0] if active_models else (models[0] if models else None)
            
            if not model:
                continue
            
            predictions, lower_bounds, upper_bounds = get_model_predictions(series, model, horizon)
            
            results.append({
                "series_id": series_id,
                "model_id": model.id,
                "predictions": predictions[:10],
                "status": "success"
            })
            
            update_task_status(task_id, "processing", int((i + 1) / total * 100))
        
        update_task_status(
            task_id, "completed", 100,
            result={"results": results, "total": len(results)}
        )
        
    except Exception as e:
        logger.error(f"Batch forecast failed: {str(e)}")
        update_task_status(task_id, "failed", 0, None, str(e))