"""
B07: Interpretation API
- B07-01: GET /interpret/shap/{model_id}
- B07-02: GET /interpret/importance/{model_id}
- B07-03: POST /interpret/lags/{series_id}
- B07-04: GET /interpret/status/{task_id}
- B07-05: GET /interpret/report/{series_id}
- B07-06: POST /interpret/lime/{model_id}
"""

import uuid
from datetime import datetime
from typing import Optional, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, status

from src.models import (
    get_series_by_id, get_model, get_models_by_series,
    update_task_status, get_task_status
)
from src.schemas.interpretation import (
    SHAPValuesResponse, FeatureImportanceResponse,
    LagsExplanationRequest, LagsExplanationResponse, LagsExplanationResult,
    TaskStatusResponse, TextReportResponse, LIMERequest, LIMEResponse
)
from src.services.interpretation_service import (
    generate_shap_values, generate_feature_importance,
    generate_llm_explanation, get_important_lags_from_model,
    generate_text_report, generate_lime_explanation
)
from src.api.dependencies import get_current_user
from src.utils.exceptions import NotFoundError, ValidationError
from src.utils.logger import get_logger

logger = get_logger("interpretation")

router = APIRouter(prefix="/interpret", tags=["Interpretation"])


# ========== B07-01: SHAP значения ==========

@router.get("/shap/{model_id}", response_model=SHAPValuesResponse)
async def get_shap_values(
    model_id: str,
    current_user = Depends(get_current_user)
):
    """
    Получение SHAP значений для модели
    (объяснение важности признаков)
    """
    model = get_model(model_id)
    if not model:
        raise NotFoundError("Модель", model_id)
    
    # Проверка доступа
    series = get_series_by_id(model.series_id)
    if not series or series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    # Получение лагов
    lags = get_important_lags_from_model(model)
    
    # Генерация SHAP значений
    shap_data = generate_shap_values(series, model, lags)
    
    return SHAPValuesResponse(
        model_id=model.id,
        model_type=model.model_type,
        base_value=shap_data["base_value"],
        feature_names=shap_data["feature_names"],
        shap_values=shap_data["shap_values"],
        global_importance=shap_data["global_importance"]
    )


# ========== B07-02: Важность признаков ==========

@router.get("/importance/{model_id}", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    model_id: str,
    current_user = Depends(get_current_user)
):
    """
    Получение важности признаков для модели
    """
    model = get_model(model_id)
    if not model:
        raise NotFoundError("Модель", model_id)
    
    series = get_series_by_id(model.series_id)
    if not series or series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    lags = get_important_lags_from_model(model)
    importance = generate_feature_importance(model, lags)
    
    top_features = [item["feature"] for item in importance[:5]]
    
    return FeatureImportanceResponse(
        model_id=model.id,
        model_type=model.model_type,
        importance=importance,
        top_features=top_features
    )


# ========== B07-03: LLM объяснение лагов ==========

@router.post("/lags/{series_id}", response_model=LagsExplanationResponse, status_code=status.HTTP_202_ACCEPTED)
async def explain_lags(
    series_id: str,
    request: LagsExplanationRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user)
):
    """
    Запуск LLM (Qwen-7B) для объяснения важности лагов
    """
    # Проверка ряда
    series = get_series_by_id(series_id)
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    # Проверка модели
    model = get_model(request.model_id)
    if not model:
        raise NotFoundError("Модель", request.model_id)
    
    if model.series_id != series_id:
        raise HTTPException(status_code=400, detail="Модель не принадлежит этому ряду")
    
    task_id = str(uuid.uuid4())
    
    update_task_status(task_id, "pending", 0)
    
    background_tasks.add_task(
        run_llm_explanation_task,
        task_id=task_id,
        series=series,
        model=model,
        language=request.language
    )
    
    return LagsExplanationResponse(
        task_id=task_id,
        status="pending",
        message="LLM анализ запущен. Используйте GET /interpret/status/{task_id} для получения результата"
    )


async def run_llm_explanation_task(
    task_id: str,
    series,
    model,
    language: str
):
    """Фоновая задача для LLM объяснения"""
    try:
        update_task_status(task_id, "processing", 20)
        
        # Получение важных лагов
        lags = get_important_lags_from_model(model)
        
        update_task_status(task_id, "processing", 50)
        
        # Генерация объяснения (в реальном проекте здесь был бы запрос к Qwen-7B)
        explanation = generate_llm_explanation(
            important_lags=lags[:5],
            series_name=series.name,
            language=language
        )
        
        update_task_status(task_id, "processing", 80)
        
        result = LagsExplanationResult(
            important_lags=lags[:5],
            explanation=explanation,
            confidence=0.85
        )
        
        update_task_status(
            task_id, "completed", 100,
            result=result.model_dump()
        )
        
        logger.info(f"LLM explanation completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"LLM explanation failed: {str(e)}")
        update_task_status(task_id, "failed", 0, None, str(e))


# ========== B07-04: Статус LLM задачи ==========

@router.get("/status/{task_id}", response_model=TaskStatusResponse)
async def get_interpretation_status(
    task_id: str,
    current_user = Depends(get_current_user)
):
    """
    Получение статуса задачи LLM интерпретации
    """
    status_data = get_task_status(task_id)
    if not status_data:
        raise NotFoundError("Задача", task_id)
    
    result = None
    if status_data.get("result"):
        result = LagsExplanationResult(**status_data["result"])
    
    return TaskStatusResponse(
        task_id=task_id,
        status=status_data.get("status", "unknown"),
        progress=status_data.get("progress", 0),
        result=result,
        error=status_data.get("error"),
        created_at=status_data.get("created_at", datetime.utcnow()),
        updated_at=status_data.get("updated_at", datetime.utcnow())
    )


# ========== B07-05: Текстовый отчёт ==========

@router.get("/report/{series_id}", response_model=TextReportResponse)
async def get_text_report(
    series_id: str,
    model_id: Optional[str] = Query(None, description="ID модели (если не указана, используется активная)"),
    current_user = Depends(get_current_user)
):
    """
    Получение текстового отчёта о временном ряде и модели
    """
    series = get_series_by_id(series_id)
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    # Выбор модели
    model = None
    if model_id:
        model = get_model(model_id)
    else:
        models = get_models_by_series(series_id)
        active_models = [m for m in models if m.is_active]
        if active_models:
            model = active_models[0]
        elif models:
            model = models[0]
    
    if not model:
        raise ValidationError("Нет модели для генерации отчёта")
    
    report = generate_text_report(series, model)
    
    sections = ["Общая информация", "Информация о модели", "Метрики качества", "Краткий анализ"]
    
    return TextReportResponse(
        series_id=series_id,
        model_id=model.id,
        generated_at=datetime.utcnow(),
        report=report,
        sections=sections
    )


# ========== B07-06: LIME объяснение ==========

@router.post("/lime/{model_id}", response_model=LIMEResponse)
async def get_lime_explanation(
    model_id: str,
    request: LIMERequest,
    current_user = Depends(get_current_user)
):
    """
    LIME объяснение для локального прогноза
    (особенно полезно для LSTM моделей)
    """
    model = get_model(model_id)
    if not model:
        raise NotFoundError("Модель", model_id)
    
    series = get_series_by_id(model.series_id)
    if not series or series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    lime_data = generate_lime_explanation(
        series=series,
        model=model,
        sample_index=request.sample_index,
        num_features=request.num_features
    )
    
    return LIMEResponse(
        model_id=model.id,
        sample_index=request.sample_index,
        prediction=lime_data["prediction"],
        explanations=lime_data["explanations"],
        intercept=lime_data["intercept"]
    )
