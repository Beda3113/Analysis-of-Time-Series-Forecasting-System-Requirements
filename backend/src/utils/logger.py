"""
B01-05: Structured logging
"""

import logging
import json
import sys
from datetime import datetime
from typing import Any, Dict, Optional
from contextvars import ContextVar

from src.config import settings

request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


class JSONFormatter(logging.Formatter):
    """JSON форматтер"""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry: Dict[str, Any] = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        request_id = request_id_var.get()
        if request_id:
            log_entry["request_id"] = request_id
        
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_entry, ensure_ascii=False)


class TextFormatter(logging.Formatter):
    """Текстовый форматтер"""
    
    def format(self, record: logging.LogRecord) -> str:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        request_id = request_id_var.get()
        
        prefix = f"[{timestamp}] [{record.levelname}]"
        if request_id:
            prefix += f" [req:{request_id}]"
        
        message = record.getMessage()
        
        if record.exc_info:
            message += f"\n{self.formatException(record.exc_info)}"
        
        return f"{prefix} {message}"


def setup_logging() -> None:
    """Настройка логирования"""
    
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    console_handler = logging.StreamHandler(sys.stdout)
    
    if settings.log_format == "json":
        formatter = JSONFormatter()
    else:
        formatter = TextFormatter()
    
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Настройка сторонних логгеров
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    
    app_logger = logging.getLogger("app")
    app_logger.info(f"Логирование настроено. Уровень: {settings.log_level}")


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"app.{name}")
