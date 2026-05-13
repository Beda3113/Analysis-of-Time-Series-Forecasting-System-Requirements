"""
W01-01: Celery app configuration - Broker (Redis), backend
"""

import os
from celery import Celery
from kombu import Exchange, Queue

# Настройки по умолчанию
REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
REDIS_PORT = os.environ.get('REDIS_PORT', '6379')
REDIS_DB = os.environ.get('REDIS_DB', '0')
REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD', '')

# URL для брокера и бэкенда
if REDIS_PASSWORD:
    BROKER_URL = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
    RESULT_BACKEND = f'redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{int(REDIS_DB) + 1}'
else:
    BROKER_URL = f'redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}'
    RESULT_BACKEND = f'redis://{REDIS_HOST}:{REDIS_PORT}/{int(REDIS_DB) + 1}'

# Создание Celery приложения
celery_app = Celery(
    'timeseries_forecasting',
    broker=BROKER_URL,
    backend=RESULT_BACKEND,
    include=[
        'src.workers.tasks.training',
        'src.workers.tasks.interpretation',
        'src.workers.tasks.maintenance'
    ]
)

# Конфигурация Celery
celery_app.conf.update(
    # W01-01: Основные настройки
    task_serializer='json',
    accept_content=['json'],
    result_serializer='json',
    timezone='UTC',
    enable_utc=True,
    
    # W01-02: Worker pool setup
    worker_concurrency=int(os.environ.get('CELERY_WORKER_CONCURRENCY', 4)),
    worker_max_tasks_per_child=int(os.environ.get('CELERY_MAX_TASKS_PER_CHILD', 100)),
    worker_prefetch_multiplier=1,
    
    # W01-05: Task retry policy (exponential backoff)
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_default_retry_delay=60,
    task_max_retries=3,
    
    # Очереди
    task_queues=(
        Queue('default', Exchange('default'), routing_key='default'),
        Queue('training', Exchange('training'), routing_key='training'),
        Queue('interpretation', Exchange('interpretation'), routing_key='interpretation'),
        Queue('maintenance', Exchange('maintenance'), routing_key='maintenance'),
        Queue('priority', Exchange('priority'), routing_key='priority'),
    ),
    task_default_queue='default',
    task_default_exchange='default',
    task_default_routing_key='default',
    
    # Маршрутизация задач по очередям
    task_routes={
        'src.workers.tasks.training.*': {'queue': 'training'},
        'src.workers.tasks.interpretation.*': {'queue': 'interpretation'},
        'src.workers.tasks.maintenance.*': {'queue': 'maintenance'},
        '*.priority_*': {'queue': 'priority'},
    },
    
    # W01-05: Exponential backoff для повторных попыток
    task_retry_backoff=True,
    task_retry_backoff_max=600,
    task_retry_jitter=True,
    
    # Мониторинг
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Таймауты
    task_time_limit=int(os.environ.get('CELERY_TASK_TIME_LIMIT', 3600)),
    task_soft_time_limit=int(os.environ.get('CELERY_TASK_SOFT_TIME_LIMIT', 3000)),
)

# W01-02: Настройки для разных типов воркеров
WORKER_SETTINGS = {
    'default': {
        'concurrency': 4,
        'max_tasks_per_child': 100,
        'queues': ['default']
    },
    'training': {
        'concurrency': 2,
        'max_tasks_per_child': 50,
        'queues': ['training'],
        'time_limit': 7200,  # 2 часа
        'soft_time_limit': 6000  # 100 минут
    },
    'interpretation': {
        'concurrency': 2,
        'max_tasks_per_child': 100,
        'queues': ['interpretation'],
        'time_limit': 600,  # 10 минут
        'soft_time_limit': 500  # 8 минут
    },
    'maintenance': {
        'concurrency': 1,
        'max_tasks_per_child': 10,
        'queues': ['maintenance'],
        'time_limit': 3600,  # 1 час
        'soft_time_limit': 3000
    }
}


def get_worker_settings(worker_type: str = 'default') -> dict:
    """Получение настроек для конкретного типа воркера"""
    return WORKER_SETTINGS.get(worker_type, WORKER_SETTINGS['default'])


# Экспорт приложения
app = celery_app
