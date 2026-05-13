"""
B09: External Models API
- B09-01: POST /models/external/upload
- B09-02: GET /models/external
- B09-03: DELETE /models/external/{id}
- B09-04: POST /models/external/validate
- B09-05: POST /forecast/external/{id}
"""

import uuid
import pickle
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Query, status

from src.schemas.external import (
    ExternalModelUploadRequest, ExternalModelUploadResponse,
    ExternalModelInfo, ExternalModelListResponse,
    ExternalModelValidateRequest, ExternalModelValidateResponse,
    ExternalForecastRequest, ExternalForecastResponse
)
from src.services.external_model_service import (
    external_model_storage, validate_model_file, generate_model_id,
    predict_with_external_model, get_model_cache_key
)
from src.api.dependencies import get_current_user
from src.utils.exceptions import NotFoundError, ValidationError
from src.utils.logger import get_logger

logger = get_logger("external_models")

router = APIRouter(prefix="/models/external", tags=["External Models"])


# ========== B09-01: Загрузка клиентской модели ==========

@router.post("/upload", response_model=ExternalModelUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_external_model(
    file: UploadFile = File(...),
    name: str = Form(...),
    model_type: str = Form(...),
    framework_version: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    current_user = Depends(get_current_user)
):
    """
    Загрузка клиентской модели (pickle-файл)
    Поддерживаются: XGBoost, scikit-learn, LightGBM, CatBoost
    """
    # Проверка формата файла
    if not file.filename.endswith(('.pkl', '.pickle', '.joblib')):
        raise ValidationError("Неподдерживаемый формат файла. Используйте .pkl, .pickle или .joblib", field="file")
    
    # Проверка размера файла (макс 100MB)
    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        raise ValidationError("Размер файла не должен превышать 100MB", field="file")
    
    # Валидация модели
    is_valid, errors, warnings, metadata = validate_model_file(content, model_type)
    
    if not is_valid:
        raise ValidationError(f"Модель не прошла валидацию: {', '.join(errors)}", field="file")
    
    # Сохранение модели
    model_id = generate_model_id()
    model_data = {
        "model": pickle.loads(content),  # В реальном проекте сохранять в MinIO
        "file_name": file.filename,
        "file_size": len(content)
    }
    
    metadata_dict = {
        "id": model_id,
        "user_id": current_user.id,
        "name": name,
        "model_type": model_type,
        "framework_version": framework_version,
        "description": description,
        "file_name": file.filename,
        "file_size": len(content),
        "is_active": True,
        "metadata": metadata,
        "warnings": warnings,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    
    external_model_storage.add_model(model_id, model_data, metadata_dict)
    
    logger.info(f"External model uploaded: {model_id} by user {current_user.id}")
    
    return ExternalModelUploadResponse(
        model_id=model_id,
        name=name,
        model_type=model_type,
        file_name=file.filename,
        file_size=len(content),
        created_at=datetime.utcnow(),
        message=f"Модель успешно загружена. Предупреждения: {len(warnings)}"
    )


# ========== B09-02: Список внешних моделей ==========

@router.get("/", response_model=ExternalModelListResponse)
async def list_external_models(
    current_user = Depends(get_current_user)
):
    """
    Получение списка загруженных внешних моделей пользователя
    """
    models = external_model_storage.list_models(current_user.id)
    
    items = []
    for model in models:
        items.append(ExternalModelInfo(
            id=model.get("id"),
            user_id=model.get("user_id"),
            name=model.get("name"),
            model_type=model.get("model_type"),
            framework_version=model.get("framework_version"),
            description=model.get("description"),
            file_name=model.get("file_name"),
            file_size=model.get("file_size"),
            is_active=model.get("is_active", True),
            metadata=model.get("metadata", {}),
            created_at=model.get("created_at"),
            updated_at=model.get("updated_at")
        ))
    
    return ExternalModelListResponse(items=items, total=len(items))


# ========== B09-03: Удаление внешней модели ==========

@router.delete("/{model_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_external_model(
    model_id: str,
    current_user = Depends(get_current_user)
):
    """
    Удаление загруженной внешней модели
    """
    metadata = external_model_storage.get_metadata(model_id)
    if not metadata:
        raise NotFoundError("Внешняя модель", model_id)
    
    if metadata.get("user_id") != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    external_model_storage.delete_model(model_id)
    logger.info(f"External model deleted: {model_id}")
    
    return None


# ========== B09-04: Валидация модели без сохранения ==========

@router.post("/validate", response_model=ExternalModelValidateResponse)
async def validate_external_model(
    file: UploadFile = File(...),
    model_type: str = Form(...),
    current_user = Depends(get_current_user)
):
    """
    Валидация модели без сохранения (проверка совместимости)
    """
    if not file.filename.endswith(('.pkl', '.pickle', '.joblib')):
        return ExternalModelValidateResponse(
            is_valid=False,
            model_type=model_type,
            checked_methods=["load", "predict"],
            errors=["Неподдерживаемый формат файла"],
            warnings=[],
            metadata={}
        )
    
    content = await file.read()
    if len(content) > 100 * 1024 * 1024:
        return ExternalModelValidateResponse(
            is_valid=False,
            model_type=model_type,
            checked_methods=["load"],
            errors=["Размер файла превышает 100MB"],
            warnings=[],
            metadata={}
        )
    
    is_valid, errors, warnings, metadata = validate_model_file(content, model_type)
    
    checked_methods = ["load"]
    if metadata.get("has_predict"):
        checked_methods.append("predict")
    if metadata.get("has_predict_proba"):
        checked_methods.append("predict_proba")
    
    return ExternalModelValidateResponse(
        is_valid=is_valid,
        model_type=model_type,
        checked_methods=checked_methods,
        errors=errors,
        warnings=warnings,
        metadata=metadata
    )


# ========== B09-05: Прогноз через внешнюю модель ==========

@router.post("/forecast/{model_id}", response_model=ExternalForecastResponse)
async def forecast_with_external_model(
    model_id: str,
    request: ExternalForecastRequest,
    current_user = Depends(get_current_user)
):
    """
    Получение прогноза через загруженную внешнюю модель
    """
    # Проверка модели
    metadata = external_model_storage.get_metadata(model_id)
    if not metadata:
        raise NotFoundError("Внешняя модель", model_id)
    
    if metadata.get("user_id") != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    # Проверка ряда
    from src.models import get_series_by_id
    series = get_series_by_id(request.series_id)
    if not series:
        raise NotFoundError("Временной ряд", request.series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    # Прогноз
    try:
        predictions, confidence_intervals, execution_time = predict_with_external_model(
            model_id=model_id,
            series_id=request.series_id,
            horizon=request.horizon,
            features=request.features
        )
        
        return ExternalForecastResponse(
            model_id=model_id,
            model_name=metadata.get("name"),
            series_id=request.series_id,
            horizon=request.horizon,
            predictions=predictions,
            confidence_intervals=confidence_intervals,
            execution_time_ms=round(execution_time, 2),
            cached=False
        )
        
    except Exception as e:
        logger.error(f"External forecast failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ошибка прогнозирования: {str(e)}")
