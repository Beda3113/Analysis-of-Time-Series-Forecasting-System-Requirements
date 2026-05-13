"""
W04: Maintenance Tasks - Celery задачи для обслуживания системы
- W04-01: cleanup_temp_files
- W04-02: archive_old_forecasts
- W04-03: model_health_check
- W04-04: metrics_export
"""

import os
import json
import tempfile
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from celery import shared_task
from celery.schedules import crontab
from src.workers.tasks.base import BaseTask
from src.utils.logger import get_logger

logger = get_logger("maintenance_tasks")

# In-memory хранилище для метрик
metrics_store = {}


def update_metrics(metric_name: str, value: Any) -> None:
    """Обновление метрики"""
    if metric_name not in metrics_store:
        metrics_store[metric_name] = []
    metrics_store[metric_name].append({
        "value": value,
        "timestamp": datetime.utcnow().isoformat()
    })
    if len(metrics_store[metric_name]) > 1000:
        metrics_store[metric_name] = metrics_store[metric_name][-1000:]


# ========== W04-01: cleanup_temp_files ==========

@shared_task(bind=True, base=BaseTask, name="cleanup_temp_files", queue="maintenance")
def cleanup_temp_files_task(self, max_age_hours: int = 1):
    """Удаление временных файлов старше указанного возраста"""
    try:
        logger.info("Starting temporary files cleanup")
        
        temp_paths = [
            tempfile.gettempdir(),
            "/tmp/uploaded_files",
            "/tmp/model_cache",
        ]
        
        project_temp = os.path.join(os.path.dirname(__file__), "../../../temp")
        if os.path.exists(project_temp):
            temp_paths.append(project_temp)
        
        deleted_files = 0
        freed_space = 0
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        for temp_path in temp_paths:
            if not os.path.exists(temp_path):
                continue
            
            for root, dirs, files in os.walk(temp_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                        if mtime < cutoff_time:
                            file_size = os.path.getsize(file_path)
                            os.remove(file_path)
                            deleted_files += 1
                            freed_space += file_size
                    except (PermissionError, OSError):
                        pass
                    except Exception as e:
                        logger.debug(f"Failed to delete {file_path}: {e}")
        
        result = {
            "deleted_files": deleted_files,
            "freed_space_mb": round(freed_space / (1024 * 1024), 2),
            "max_age_hours": max_age_hours,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        update_metrics("cleanup_temp_files", result)
        logger.info(f"Cleanup completed: deleted {deleted_files} files")
        return result
        
    except Exception as e:
        logger.error(f"Cleanup failed: {str(e)}")
        raise self.retry_task(e, max_retries=2)


# ========== W04-02: archive_old_forecasts ==========

@shared_task(bind=True, base=BaseTask, name="archive_old_forecasts", queue="maintenance")
def archive_old_forecasts_task(self, max_age_days: int = 30):
    """Архивация старых прогнозов старше 30 дней"""
    try:
        logger.info(f"Starting forecasts archiving (max age: {max_age_days} days)")
        
        from src.models import models_db
        
        cutoff_date = datetime.utcnow() - timedelta(days=max_age_days)
        
        archived_count = 0
        archived_models = []
        
        for model_id, model in list(models_db.items()):
            if hasattr(model, 'created_at'):
                model_date = model.created_at
                if isinstance(model_date, str):
                    model_date = datetime.fromisoformat(model_date.replace('Z', '+00:00'))
                
                if model_date < cutoff_date:
                    archived_count += 1
                    archived_models.append({
                        "model_id": model.id,
                        "model_name": model.name,
                        "model_type": model.model_type,
                        "created_at": model_date.isoformat() if hasattr(model_date, 'isoformat') else str(model_date)
                    })
        
        archive_path = None
        if archived_models:
            archive_dir = "/tmp/archives"
            os.makedirs(archive_dir, exist_ok=True)
            archive_path = os.path.join(archive_dir, f"forecasts_archive_{datetime.utcnow().strftime('%Y%m%d')}.json")
            with open(archive_path, 'w') as f:
                json.dump(archived_models, f, indent=2, default=str)
        
        result = {
            "archived_count": archived_count,
            "max_age_days": max_age_days,
            "archive_path": archive_path,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        update_metrics("archive_old_forecasts", result)
        logger.info(f"Archiving completed: {archived_count} forecasts archived")
        return result
        
    except Exception as e:
        logger.error(f"Archiving failed: {str(e)}")
        raise self.retry_task(e, max_retries=2)


# ========== W04-03: model_health_check ==========

@shared_task(bind=True, base=BaseTask, name="model_health_check", queue="maintenance")
def model_health_check_task(self, threshold_mape: float = 20.0):
    """Проверка моделей на деградацию качества"""
    try:
        logger.info("Starting model health check")
        
        from src.models import models_db
        
        degraded_models = []
        healthy_models = []
        
        for model_id, model in models_db.items():
            mape = model.metrics.get('mape', 0) if hasattr(model, 'metrics') and model.metrics else 0
            
            status = {
                "model_id": model.id,
                "model_name": model.name,
                "model_type": model.model_type,
                "mape": mape,
                "is_active": model.is_active if hasattr(model, 'is_active') else False
            }
            
            if mape > threshold_mape:
                status["status"] = "degraded"
                status["recommendation"] = "Рекомендуется переобучение модели"
                degraded_models.append(status)
            else:
                status["status"] = "healthy"
                healthy_models.append(status)
        
        total_models = len(models_db)
        degraded_percentage = (len(degraded_models) / total_models * 100) if total_models > 0 else 0
        
        result = {
            "total_models": total_models,
            "healthy_models": len(healthy_models),
            "degraded_models": len(degraded_models),
            "degraded_percentage": round(degraded_percentage, 2),
            "degraded_list": degraded_models[:10],
            "threshold_mape": threshold_mape,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        update_metrics("model_health_check", result)
        
        if degraded_models:
            logger.warning(f"Found {len(degraded_models)} degraded models")
        
        logger.info(f"Health check completed")
        return result
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise self.retry_task(e, max_retries=2)


# ========== W04-04: metrics_export ==========

@shared_task(bind=True, base=BaseTask, name="metrics_export", queue="maintenance")
def metrics_export_task(self, export_format: str = "json"):
    """Экспорт метрик для мониторинга"""
    try:
        logger.info("Starting metrics export")
        
        aggregated_metrics = {
            "timestamp": datetime.utcnow().isoformat(),
            "metrics": {},
            "summary": {}
        }
        
        for metric_name, values in metrics_store.items():
            if values:
                aggregated_metrics["metrics"][metric_name] = values[-10:]
                aggregated_metrics["summary"][metric_name] = {
                    "count": len(values),
                    "last_value": values[-1]["value"] if values else None,
                    "last_update": values[-1]["timestamp"] if values else None
                }
        
        export_dir = "/tmp/metrics"
        os.makedirs(export_dir, exist_ok=True)
        export_path = os.path.join(export_dir, f"metrics_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.{export_format}")
        
        if export_format == "json":
            with open(export_path, 'w') as f:
                json.dump(aggregated_metrics, f, indent=2, default=str)
        
        # Prometheus format
        prometheus_lines = [
            "# HELP timeseries_forecasting_metrics Метрики системы",
            "# TYPE timeseries_forecasting_metrics gauge",
            f'timeseries_forecasting_info{{version="1.0.0"}} 1'
        ]
        
        for metric_name, values in metrics_store.items():
            if values:
                last_value = values[-1].get("value", {})
                if isinstance(last_value, dict):
                    for k, v in last_value.items():
                        if isinstance(v, (int, float)):
                            prometheus_lines.append(f'timeseries_{metric_name}_{k} {v}')
                elif isinstance(last_value, (int, float)):
                    prometheus_lines.append(f'timeseries_{metric_name} {last_value}')
        
        prom_path = os.path.join(export_dir, "prometheus_metrics.txt")
        with open(prom_path, 'w') as f:
            f.write("\n".join(prometheus_lines))
        
        result = {
            "export_format": export_format,
            "export_path": export_path,
            "prometheus_path": prom_path,
            "metrics_count": len(metrics_store),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        update_metrics("metrics_export", result)
        logger.info(f"Metrics exported to {export_path}")
        return result
        
    except Exception as e:
        logger.error(f"Metrics export failed: {str(e)}")
        raise self.retry_task(e, max_retries=2)


def get_beat_schedule():
    """Расписание периодических задач для Celery Beat"""
    return {
        'cleanup-temp-files-hourly': {
            'task': 'cleanup_temp_files',
            'schedule': crontab(minute=0),
            'args': (1,),
            'options': {'queue': 'maintenance'}
        },
        'metrics-export-hourly': {
            'task': 'metrics_export',
            'schedule': crontab(minute=30),
            'args': ('json',),
            'options': {'queue': 'maintenance'}
        },
        'model-health-check-6h': {
            'task': 'model_health_check',
            'schedule': crontab(minute=0, hour='*/6'),
            'args': (20.0,),
            'options': {'queue': 'maintenance'}
        },
        'archive-forecasts-daily': {
            'task': 'archive_old_forecasts',
            'schedule': crontab(minute=0, hour=1),
            'args': (30,),
            'options': {'queue': 'maintenance'}
        },
    }
