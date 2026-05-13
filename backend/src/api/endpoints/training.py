"""
B05: Training API
- B05-01: POST /training/{series_id}
- B05-02: GET /training/status/{task_id}
- B05-03: GET /models
- B05-04: GET /models/{model_id}
- B05-05: DELETE /models/{model_id}
- B05-06: POST /models/{model_id}/activate
- B05-07: GET /models/compare
"""

import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, status
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.postgres.connection import get_session
from src.storage.postgres.crud import TimeSeriesCRUD, TrainedModelCRUD
from src.schemas.training import (
    TrainingRequest, TrainingResponse, TrainingStatusResponse,
    ModelInfo, ModelListResponse, ModelCompareResponse,
    ModelMetrics
)
from src.api.dependencies import get_current_user
from src.services.training_service import train_model
from src.utils.exceptions import NotFoundError, ValidationError
from src.utils.logger import get_logger

logger = get_logger("training")

router = APIRouter(prefix="/training", tags=["Training"])


# ========== B05-01: Запуск обучения ==========

@router.post("/{series_id}", response_model=TrainingResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_training(
    series_id: str,
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """
    Запуск асинхронного обучения модели для временного ряда
    """
    # ИЗМЕНЕНО: используем CRUD вместо in-memory
    series = await TimeSeriesCRUD.get_by_id(db, series_id)
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    # Валидация типа модели
    allowed_models = ["xgboost", "lstm", "prophet", "sarima"]
    if request.model_type not in allowed_models:
        raise ValidationError(
            f"Неподдерживаемый тип модели. Доступны: {', '.join(allowed_models)}",
            field="model_type"
        )
    
    # Создание задачи
    task_id = str(uuid.uuid4())
    model_id = str(uuid.uuid4())
    
    # Сохраняем статус задачи (пока in-memory, потом перенесём в БД)
    from src.models import update_task_status
    update_task_status(task_id, "pending", 0)
    
    # Передаём db и series в фоновую задачу
    background_tasks.add_task(
        run_training_task,
        task_id=task_id,
        model_id=model_id,
        series_id=series.id,
        series_values=series.values,
        series_name=series.name,
        user_id=current_user.id,
        model_type=request.model_type,
        name=request.name or f"{request.model_type.upper()}_{series.name[:20]}",
        horizon=request.horizon,
        hyperparams=request.hyperparams,
        db=db  # Передаём сессию
    )
    
    logger.info(f"Training started: task_id={task_id}, series_id={series_id}, model_type={request.model_type}")
    
    return TrainingResponse(
        task_id=task_id,
        model_id=model_id,
        status="pending",
        message="Обучение запущено. Используйте GET /training/status/{task_id} для отслеживания прогресса"
    )


async def run_training_task(
    task_id: str,
    model_id: str,
    series_id: str,
    series_values: list,
    series_name: str,
    user_id: str,
    model_type: str,
    name: str,
    horizon: int,
    hyperparams: Optional[dict],
    db: AsyncSession
):
    """Фоновая задача обучения модели с сохранением в БД"""
    from src.models import update_task_status, get_task_status
    
    try:
        update_task_status(task_id, "training", 10)
        
        # Создаём временный объект ряда для train_model
        class TempSeries:
            def __init__(self, id, values, name, user_id):
                self.id = id
                self.values = values
                self.name = name
                self.user_id = user_id
        
        temp_series = TempSeries(series_id, series_values, series_name, user_id)
        
        result = train_model(temp_series, model_type, task_id, model_id, hyperparams)
        
        if result.get("success"):
            # ИЗМЕНЕНО: сохраняем модель в PostgreSQL через CRUD
            model = await TrainedModelCRUD.create(
                db=db,
                series_id=series_id,
                user_id=user_id,
                model_type=model_type,
                name=name,
                hyperparams=result.get("hyperparams", {}),
                metrics=result.get("metrics", {}),
                file_path=result.get("model_path")
            )
            
            # Если это первая модель для ряда, активируем её
            models = await TrainedModelCRUD.get_by_series(db, series_id)
            if len(models) == 1:
                await TrainedModelCRUD.activate(db, series_id, model.id)
            
            update_task_status(
                task_id, "completed", 100,
                result={"model_id": model.id, "metrics": result.get("metrics")}
            )
            
            logger.info(f"Training completed: model_id={model.id}, metrics={result.get('metrics')}")
        else:
            update_task_status(
                task_id, "failed", 0, None,
                result.get("error", "Unknown error")
            )
            logger.error(f"Training failed: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"Training task error: {str(e)}")
        update_task_status(task_id, "failed", 0, None, str(e))


# ========== B05-02: Статус задачи ==========

@router.get("/status/{task_id}", response_model=TrainingStatusResponse)
async def get_training_status(
    task_id: str,
    current_user = Depends(get_current_user)
):
    """
    Получение статуса задачи обучения
    """
    from src.models import get_task_status
    
    status_data = get_task_status(task_id)
    if not status_data:
        raise NotFoundError("Задача", task_id)
    
    return TrainingStatusResponse(
        task_id=task_id,
        status=status_data.get("status", "unknown"),
        progress=status_data.get("progress", 0),
        result=status_data.get("result"),
        error=status_data.get("error"),
        updated_at=status_data.get("updated_at", datetime.utcnow())
    )


# ========== B05-03: Список моделей ряда ==========

@router.get("/models", response_model=ModelListResponse)
async def list_models(
    series_id: str = Query(..., description="ID временного ряда"),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """
    Получение списка всех моделей для временного ряда
    """
    # Проверка ряда
    series = await TimeSeriesCRUD.get_by_id(db, series_id)
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    # ИЗМЕНЕНО: используем CRUD вместо in-memory
    models = await TrainedModelCRUD.get_by_series(db, series_id)
    
    items = []
    for model in models:
        items.append(ModelInfo(
            id=model.id,
            series_id=model.series_id,
            user_id=model.user_id,
            model_type=model.model_type,
            name=model.name,
            hyperparams=model.hyperparams,
            metrics=ModelMetrics(**model.metrics),
            is_active=model.is_active,
            created_at=model.created_at,
            updated_at=model.updated_at
        ))
    
    return ModelListResponse(items=items, total=len(items))


# ========== B05-04: Информация о модели ==========

@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model_info(
    model_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """
    Получение подробной информации о модели
    """
    # ИЗМЕНЕНО: используем CRUD вместо in-memory
    model = await TrainedModelCRUD.get_by_id(db, model_id)
    if not model:
        raise NotFoundError("Модель", model_id)
    
    # Проверка доступа
    if model.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    return ModelInfo(
        id=model.id,
        series_id=model.series_id,
        user_id=model.user_id,
        model_type=model.model_type,
        name=model.name,
        hyperparams=model.hyperparams,
        metrics=ModelMetrics(**model.metrics),
        is_active=model.is_active,
        created_at=model.created_at,
        updated_at=model.updated_at
    )


# ========== B05-05: Удаление модели ==========

@router.delete("/models/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_model_endpoint(
    model_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """
    Удаление модели
    """
    # ИЗМЕНЕНО: используем CRUD вместо in-memory
    model = await TrainedModelCRUD.get_by_id(db, model_id)
    if not model:
        raise NotFoundError("Модель", model_id)
    
    if model.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    await TrainedModelCRUD.delete(db, model_id)
    logger.info(f"Model deleted: {model_id}")
    return None


# ========== B05-06: Активация модели ==========

@router.post("/models/{model_id}/activate", status_code=status.HTTP_200_OK)
async def activate_model_endpoint(
    model_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """
    Активация модели для ряда (деактивирует все остальные)
    """
    # ИЗМЕНЕНО: используем CRUD вместо in-memory
    model = await TrainedModelCRUD.get_by_id(db, model_id)
    if not model:
        raise NotFoundError("Модель", model_id)
    
    if model.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    await TrainedModelCRUD.activate(db, model.series_id, model_id)
    
    return {"message": f"Модель {model.name} активирована", "model_id": model_id}


# ========== B05-07: Сравнение моделей ==========

@router.get("/models/compare", response_model=ModelCompareResponse)
async def compare_models(
    series_id: str = Query(..., description="ID временного ряда"),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """
    Сравнение всех моделей для ряда
    """
    series = await TimeSeriesCRUD.get_by_id(db, series_id)
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    # ИЗМЕНЕНО: используем CRUD вместо in-memory
    models = await TrainedModelCRUD.get_by_series(db, series_id)
    
    items = []
    best_mape = None
    best_rmse = None
    best_mape_id = None
    best_rmse_id = None
    
    for model in models:
        items.append(ModelInfo(
            id=model.id,
            series_id=model.series_id,
            user_id=model.user_id,
            model_type=model.model_type,
            name=model.name,
            hyperparams=model.hyperparams,
            metrics=ModelMetrics(**model.metrics),
            is_active=model.is_active,
            created_at=model.created_at,
            updated_at=model.updated_at
        ))
        
        mape = model.metrics.get("mape")
        rmse = model.metrics.get("rmse")
        
        if mape is not None:
            if best_mape is None or mape < best_mape:
                best_mape = mape
                best_mape_id = model.id
        
        if rmse is not None:
            if best_rmse is None or rmse < best_rmse:
                best_rmse = rmse
                best_rmse_id = model.id
    
    return ModelCompareResponse(
        models=items,
        best_by_mape=best_mape_id,
        best_by_rmse=best_rmse_id
    )