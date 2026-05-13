"""
B04: Series API
- B04-01: POST /series/upload - загрузка CSV/Excel
- B04-02: GET /series - список рядов
- B04-03: GET /series/{id} - информация о ряде
- B04-04: GET /series/{id}/preview - предпросмотр
- B04-05: DELETE /series/{id} - удаление ряда
- B04-06: PATCH /series/{id} - обновление метаданных
- B04-07: GET /series/{id}/plot - график
- B04-08: POST /series/upload-multi - загрузка нескольких рядов
"""

import io
import base64
import chardet
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from src.storage.postgres.connection import get_session
from src.storage.postgres.crud import TimeSeriesCRUD, UserCRUD
from src.schemas.series import (
    SeriesCreate, SeriesUpdate, SeriesResponse, SeriesListResponse,
    SeriesPreviewResponse, SeriesUploadResponse, SeriesPlotResponse
)
from src.api.dependencies import get_current_user
from src.utils.exceptions import NotFoundError, ValidationError
from src.utils.logger import get_logger

logger = get_logger("series")

router = APIRouter(prefix="/series", tags=["Series"])


def detect_separator(first_line: str) -> str:
    """Автоматическое определение разделителя CSV"""
    common_separators = [',', ';', '\t', '|', ' ']
    for sep in common_separators:
        if sep in first_line:
            parts = first_line.split(sep)
            if len(parts) >= 2:
                return sep
    return ','


def detect_encoding(content: bytes) -> str:
    """Определение кодировки файла"""
    try:
        result = chardet.detect(content)
        if result and result['encoding']:
            return result['encoding']
    except:
        pass
    return 'utf-8'


def sanitize_column_name(col_name: str) -> str:
    """Очистка имени колонки для использования в качестве названия ряда"""
    import re
    cleaned = re.sub(r'[^\w\s]', '', col_name)
    cleaned = cleaned.replace(' ', '_')
    return cleaned[:100]


# ========== B04-08: Загрузка нескольких рядов ==========

@router.post("/upload-multi", response_model=List[SeriesUploadResponse], status_code=status.HTTP_201_CREATED)
async def upload_multiple_series(
    file: UploadFile = File(...),
    name_prefix: Optional[str] = None,
    auto_detect_date: bool = True,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """Загрузка CSV/Excel файла с автоматическим созданием нескольких временных рядов"""
    logger.info(f"📤 Multi-upload started: {file.filename} (user: {current_user.id})")
    
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise ValidationError("Неподдерживаемый формат файла. Используйте CSV или Excel", field="file")
    
    try:
        content = await file.read()
        
        if len(content) == 0:
            raise ValidationError("Файл пуст", field="file")
        
        logger.info(f"📄 File size: {len(content)} bytes")
        
        import pandas as pd
        
        if file.filename.endswith('.csv'):
            encoding = detect_encoding(content)
            logger.info(f"🔤 Detected encoding: {encoding}")
            
            try:
                decoded = content.decode(encoding)
            except UnicodeDecodeError:
                decoded = content.decode('utf-8', errors='replace')
            
            lines = [l.strip() for l in decoded.split('\n') if l.strip()]
            
            if len(lines) < 2:
                raise ValidationError("Файл должен содержать заголовок и данные", field="file")
            
            separator = detect_separator(lines[0])
            logger.info(f"📊 Detected separator: '{separator}'")
            
            try:
                df = pd.read_csv(io.BytesIO(content), sep=separator, encoding=encoding)
            except:
                df = pd.read_csv(io.BytesIO(content), encoding=encoding)
        else:
            df = pd.read_excel(io.BytesIO(content))
        
        logger.info(f"📊 DataFrame shape: {df.shape}")
        logger.info(f"📊 Columns: {list(df.columns)}")
        
        if df.empty:
            raise ValidationError("Файл не содержит данных", field="file")
        
        # Определяем колонку с датами
        date_column = None
        date_keywords = ['date', 'time', 'datetime', 'дата', 'день', 'dt', 'timestamp', 'period']
        
        for col in df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in date_keywords):
                date_column = col
                break
        
        if not date_column:
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    date_column = col
                    break
        
        logger.info(f"📅 Date column detected: {date_column}")
        
        # Определяем числовые колонки
        numeric_cols = []
        
        for col in df.columns:
            if col == date_column:
                continue
            
            if pd.api.types.is_numeric_dtype(df[col]):
                numeric_cols.append(col)
            else:
                try:
                    converted = pd.to_numeric(df[col], errors='coerce')
                    if converted.notna().sum() > 3:
                        numeric_cols.append(col)
                        df[col] = converted
                except:
                    pass
        
        if not numeric_cols:
            raise ValidationError(
                f"Не найдено числовых колонок для создания рядов. "
                f"Доступные колонки: {list(df.columns)}",
                field="file"
            )
        
        logger.info(f"📈 Numeric columns detected: {numeric_cols}")
        
        # Создаём ряды для каждой числовой колонки
        created_series = []
        base_name = name_prefix or file.filename.rsplit('.', 1)[0]
        
        for col in numeric_cols:
            values = df[col].dropna().tolist()
            
            if len(values) < 3:
                logger.warning(f"⚠️ Column '{col}' has only {len(values)} points, skipping")
                continue
            
            dates = None
            if date_column and date_column in df.columns:
                valid_mask = df[col].notna()
                if valid_mask.any():
                    dates = df[date_column][valid_mask].astype(str).tolist()
            
            clean_col_name = sanitize_column_name(col)
            series_name = f"{base_name}_{clean_col_name}"
            
            # ИЗМЕНЕНО: используем CRUD вместо in-memory
            series = await TimeSeriesCRUD.create(
                db=db,
                user_id=current_user.id,
                name=series_name,
                values=values,
                dates=dates,
                description=f"Автоматически загружен из колонки '{col}'"
            )
            
            created_series.append(SeriesUploadResponse(
                series_id=series.id,
                name=series.name,
                length=series.length,
                message=f"✅ Загружено {len(values)} точек из колонки '{col}'"
            ))
            
            logger.info(f"✅ Series created: {series.id} - {series.name} ({series.length} points)")
        
        if not created_series:
            raise ValidationError("Не удалось создать ни одного временного ряда", field="file")
        
        return created_series
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"❌ Multi-upload error: {str(e)}", exc_info=True)
        raise ValidationError(f"Ошибка при обработке файла: {str(e)}", field="file")


# ========== B04-01: Загрузка одного ряда ==========

@router.post("/upload", response_model=SeriesUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_series(
    file: UploadFile = File(...),
    name: Optional[str] = None,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    """Загрузка временного ряда из CSV или Excel файла (ТОЛЬКО ПЕРВАЯ ЧИСЛОВАЯ КОЛОНКА)"""
    logger.info(f"📤 Upload started: {file.filename} (user: {current_user.id})")
    
    if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
        raise ValidationError("Неподдерживаемый формат файла. Используйте CSV или Excel", field="file")
    
    try:
        content = await file.read()
        
        if len(content) == 0:
            raise ValidationError("Файл пуст", field="file")
        
        import pandas as pd
        
        if file.filename.endswith('.csv'):
            encoding = detect_encoding(content)
            try:
                decoded = content.decode(encoding)
            except UnicodeDecodeError:
                decoded = content.decode('utf-8', errors='replace')
            
            lines = [l.strip() for l in decoded.split('\n') if l.strip()]
            
            if len(lines) < 2:
                raise ValidationError("Файл должен содержать заголовок и хотя бы одну строку данных", field="file")
            
            separator = detect_separator(lines[0])
            df = pd.read_csv(io.BytesIO(content), sep=separator, encoding=encoding)
        else:
            df = pd.read_excel(io.BytesIO(content))
        
        logger.info(f"📊 DataFrame shape: {df.shape}")
        
        if df.empty:
            raise ValidationError("Файл не содержит данных", field="file")
        
        # Поиск числовой колонки
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if not numeric_cols:
            for col in df.columns:
                try:
                    converted = pd.to_numeric(df[col], errors='coerce')
                    if converted.notna().sum() > 0:
                        numeric_cols.append(col)
                        df[col] = converted
                except:
                    pass
        
        if not numeric_cols:
            raise ValidationError(
                f"Файл не содержит числовых данных. Доступные колонки: {list(df.columns)}",
                field="file"
            )
        
        value_column = numeric_cols[0]
        values = df[value_column].dropna().tolist()
        
        logger.info(f"📈 Values from column '{value_column}': {len(values)} points")
        
        if len(values) < 3:
            raise ValidationError(f"Ряд должен содержать минимум 3 точки. Найдено: {len(values)}", field="file")
        
        # Поиск колонки с датами
        date_column = None
        date_keywords = ['date', 'time', 'datetime', 'дата', 'день', 'dt']
        
        for col in df.columns:
            if any(keyword in col.lower() for keyword in date_keywords):
                date_column = col
                break
        
        dates = None
        if date_column:
            dates = df[date_column].astype(str).tolist()[:len(values)]
            logger.info(f"📅 Date column: '{date_column}'")
        
        series_name = name or file.filename.rsplit('.', 1)[0]
        
        # ИЗМЕНЕНО: используем CRUD вместо in-memory
        series = await TimeSeriesCRUD.create(
            db=db,
            user_id=current_user.id,
            name=series_name,
            values=values,
            dates=dates
        )
        
        logger.info(f"✅ Series created: {series.id} ({series.length} points)")
        
        return SeriesUploadResponse(
            series_id=series.id,
            name=series.name,
            length=series.length,
            message=f"Загружено {len(values)} точек из колонки '{value_column}'"
        )
        
    except ValidationError:
        raise
    except Exception as e:
        logger.error(f"❌ Upload error: {str(e)}", exc_info=True)
        raise ValidationError(f"Ошибка при обработке файла: {str(e)}", field="file")


# ========== B04-09: Получение структуры файла ==========

@router.post("/preview-structure")
async def preview_file_structure(
    file: UploadFile = File(...),
    current_user = Depends(get_current_user)
):
    """Предпросмотр структуры файла без сохранения в БД"""
    import pandas as pd
    
    content = await file.read()
    
    if file.filename.endswith('.csv'):
        encoding = detect_encoding(content)
        try:
            decoded = content.decode(encoding)
        except UnicodeDecodeError:
            decoded = content.decode('utf-8', errors='replace')
        
        lines = [l.strip() for l in decoded.split('\n') if l.strip()]
        separator = detect_separator(lines[0]) if len(lines) > 0 else ','
        df = pd.read_csv(io.BytesIO(content), sep=separator, encoding=encoding)
    else:
        df = pd.read_excel(io.BytesIO(content))
    
    columns_info = []
    date_candidates = []
    numeric_candidates = []
    
    for col in df.columns:
        col_info = {
            "name": col,
            "type": str(df[col].dtype),
            "is_numeric": pd.api.types.is_numeric_dtype(df[col]),
            "sample_values": df[col].head(5).dropna().tolist(),
            "null_count": int(df[col].isna().sum()),
            "unique_count": int(df[col].nunique())
        }
        columns_info.append(col_info)
        
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_candidates.append(col)
        
        if any(keyword in col.lower() for keyword in ['date', 'time', 'datetime', 'дата']):
            date_candidates.append(col)
    
    return {
        "filename": file.filename,
        "total_rows": len(df),
        "total_columns": len(df.columns),
        "columns": columns_info,
        "date_column_candidates": date_candidates,
        "numeric_column_candidates": numeric_candidates,
        "preview_data": df.head(10).to_dict(orient='records')
    }


# ========== B04-02: Список рядов пользователя ==========

@router.get("/", response_model=SeriesListResponse)
async def list_series(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: Optional[str] = None,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    # ИЗМЕНЕНО: используем CRUD вместо in-memory
    items, total = await TimeSeriesCRUD.get_by_user(
        db=db,
        user_id=current_user.id,
        skip=(page - 1) * page_size,
        limit=page_size,
        search=search
    )
    
    return SeriesListResponse(
        items=[SeriesResponse(**{
            "id": s.id,
            "user_id": s.user_id,
            "name": s.name,
            "description": s.description,
            "length": s.length,
            "min_value": s.min_value,
            "max_value": s.max_value,
            "avg_value": s.avg_value,
            "created_at": s.created_at,
            "updated_at": s.updated_at
        }) for s in items],
        total=total,
        page=page,
        page_size=page_size
    )


# ========== B04-03: Информация о ряде ==========

@router.get("/{series_id}", response_model=SeriesResponse)
async def get_series(
    series_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    # ИЗМЕНЕНО: используем CRUD вместо in-memory
    series = await TimeSeriesCRUD.get_by_id(db, series_id)
    
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    return SeriesResponse(
        id=series.id,
        user_id=series.user_id,
        name=series.name,
        description=series.description,
        length=series.length,
        min_value=series.min_value,
        max_value=series.max_value,
        avg_value=series.avg_value,
        created_at=series.created_at,
        updated_at=series.updated_at
    )


# ========== B04-04: Предпросмотр данных ==========

@router.get("/{series_id}/preview", response_model=SeriesPreviewResponse)
async def preview_series(
    series_id: str,
    rows: int = Query(20, ge=1, le=100),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    series = await TimeSeriesCRUD.get_by_id(db, series_id)
    
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    # Формируем предпросмотр из данных ряда
    preview_values = series.values[:rows]
    preview_dates = series.dates[:rows] if series.dates else [str(i) for i in range(rows)]
    
    return SeriesPreviewResponse(
        headers=["index", "date", "value"],
        data=[
            {"index": i, "date": preview_dates[i], "value": preview_values[i]}
            for i in range(len(preview_values))
        ]
    )


# ========== B04-05: Удаление ряда ==========

@router.delete("/{series_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_series_endpoint(
    series_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    series = await TimeSeriesCRUD.get_by_id(db, series_id)
    
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    await TimeSeriesCRUD.delete(db, series_id)
    logger.info(f"🗑️ Series deleted: {series_id}")


# ========== B04-06: Обновление метаданных ==========

@router.patch("/{series_id}", response_model=SeriesResponse)
async def update_series_endpoint(
    series_id: str,
    update_data: SeriesUpdate,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    series = await TimeSeriesCRUD.get_by_id(db, series_id)
    
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    update_dict = update_data.model_dump(exclude_unset=True)
    updated_series = await TimeSeriesCRUD.update(db, series_id, **update_dict)
    
    if not updated_series:
        raise NotFoundError("Временной ряд после обновления", series_id)
    
    return SeriesResponse(
        id=updated_series.id,
        user_id=updated_series.user_id,
        name=updated_series.name,
        description=updated_series.description,
        length=updated_series.length,
        min_value=updated_series.min_value,
        max_value=updated_series.max_value,
        avg_value=updated_series.avg_value,
        created_at=updated_series.created_at,
        updated_at=updated_series.updated_at
    )


# ========== B04-07: График ряда (base64) ==========

@router.get("/{series_id}/plot", response_model=SeriesPlotResponse)
async def get_series_plot(
    series_id: str,
    width: int = Query(800, ge=400, le=1200),
    height: int = Query(400, ge=200, le=800),
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_session)
):
    series = await TimeSeriesCRUD.get_by_id(db, series_id)
    
    if not series:
        raise NotFoundError("Временной ряд", series_id)
    
    if series.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Доступ запрещён")
    
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(width/100, height/100))
        
        ax.plot(series.values, linewidth=2, color='#667eea')
        ax.set_title(series.name, fontsize=14, fontweight='bold')
        ax.set_xlabel('Индекс')
        ax.set_ylabel('Значение')
        ax.grid(True, alpha=0.3)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plot_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return SeriesPlotResponse(plot=plot_base64, format="png")
        
    except ImportError:
        logger.warning("matplotlib не установлен")
        return SeriesPlotResponse(plot="", format="png")