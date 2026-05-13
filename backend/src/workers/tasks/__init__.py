"""
Celery задачи для асинхронной обработки
"""

from src.workers.tasks.base import BaseTask, task_with_retry, get_retry_delay
from src.workers.tasks.training import (
    train_xgboost_task,
    train_lstm_task,
    train_prophet_task,
    train_sarima_task,
    train_ensemble_task,
    hyperparameter_tuning_task
)
from src.workers.tasks.interpretation import (
    compute_shap_task,
    explain_lags_task,
    generate_report_task,
    lime_explain_task
)

__all__ = [
    'BaseTask',
    'task_with_retry',
    'get_retry_delay',
    # Training tasks
    'train_xgboost_task',
    'train_lstm_task',
    'train_prophet_task',
    'train_sarima_task',
    'train_ensemble_task',
    'hyperparameter_tuning_task',
    # Interpretation tasks
    'compute_shap_task',
    'explain_lags_task',
    'generate_report_task',
    'lime_explain_task'
]

from src.workers.tasks.maintenance import (
    cleanup_temp_files_task,
    archive_old_forecasts_task,
    model_health_check_task,
    metrics_export_task,
    get_beat_schedule
)

__all__.extend([
    'cleanup_temp_files_task',
    'archive_old_forecasts_task',
    'model_health_check_task',
    'metrics_export_task',
    'get_beat_schedule'
])
