"""
Workers модуль для асинхронной обработки задач
"""

from src.workers.celery_app import celery_app
from src.workers.config import get_worker_command, get_beat_command
from src.workers.signals import setup_signal_handlers

__all__ = [
    'celery_app',
    'get_worker_command',
    'get_beat_command',
    'setup_signal_handlers'
]
