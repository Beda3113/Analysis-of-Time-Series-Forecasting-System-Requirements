# Analysis-of-Time-Series-Forecasting-System-Requirements

# Time Series Forecasting Platform — Полное описание проекта

## №1. Общее описание проекта

**Time Series Forecasting Platform** — это полнофункциональное веб-приложение для анализа, прогнозирования и интерпретации временных рядов. Проект решает задачу не только получения точных прогнозов, но и объяснения результатов (XAI — Explainable AI) с помощью методов SHAP, LIME и больших языковых моделей (LLM Qwen).

### Ключевые особенности:

- **Мульти-модельность**: Поддержка классических статистических (SARIMA), бизнес-ориентированных (Prophet) и ML/DL моделей (XGBoost, LSTM).
- **Интерпретируемость**: Генерация понятных отчетов о том, какие лаги и факторы повлияли на прогноз.
- **Асинхронная архитектура**: Использование Celery для обработки тяжелых задач обучения без блокировки интерфейса.
- **Полный цикл данных**: Загрузка → Предобработка (аномалии, стационарность) → Обучение → Прогноз → Интерпретация.

---

## №2. Технологический стек

### Backend (Python 3.11)

| Компонент | Технологии |
|-----------|------------|
| **Framework** | FastAPI (асинхронный API) |
| **База данных** | PostgreSQL + SQLAlchemy + asyncpg |
| **Хранилище артефактов** | MinIO (S3-compatible) |
| **Кэш и брокер задач** | Redis |
| **Очереди задач** | Celery |
| **ML библиотеки** | XGBoost, Statsmodels (SARIMA), Prophet, TensorFlow/Keras (LSTM), Scikit-learn, SHAP, LIME |
| **LLM Интеграция** | Qwen-1.5B/7B (отдельный микросервис) |

### Frontend (React 18 + TypeScript)

| Компонент | Технологии |
|-----------|------------|
| **Сборщик** | Vite |
| **UI Kit** | Material UI (MUI) |
| **State Management** | React Query + Context API |
| **Визуализация** | Recharts |
| **HTTP Client** | Axios + JWT интерцепторы |

### Infrastructure & DevOps

- **Контейнеризация**: Docker & Docker Compose
- **Мониторинг**: Flower (Celery), Prometheus metrics
- **Безопасность**: JWT (Access/Refresh), bcrypt, Rate Limiting (Redis)

---

## №3. Архитектура системы

Проект построен по принципу **модульного монолита** с четким разделением ответственности.

### Структура Backend (`backend/src/`)

```
├── api/                      # API слой
│   ├── endpoints/            # Эндпоинты
│   │   ├── auth.py           # Регистрация, логин, refresh
│   │   ├── series.py         # CRUD временных рядов
│   │   ├── training.py       # Запуск обучения, статус
│   │   ├── forecast.py       # Прогнозы, экспорт, пакетный режим
│   │   ├── interpretation.py # SHAP, LIME, отчёты
│   │   ├── preprocessing.py  # Аномалии, стационарность
│   │   └── external_models.py# Загрузка пользовательских моделей
│   ├── schemas/              # Pydantic модели
│   │   ├── auth.py
│   │   ├── series.py
│   │   ├── training.py
│   │   ├── forecast.py
│   │   ├── interpretation.py
│   │   └── preprocessing.py
│   ├── dependencies.py       # get_current_user, get_session
│   └── router.py             # Регистрация всех роутеров
├── core/                     # Бизнес-логика ML
│   ├── forecasting/          # Модели прогнозирования
│   │   ├── base.py           # Абстрактный BaseForecaster
│   │   ├── xgboost_forecaster.py
│   │   ├── lstm_forecaster.py
│   │   ├── prophet_forecaster.py
│   │   ├── sarima_forecaster.py
│   │   ├── ensemble.py       # Ансамбль моделей
│   │   └── registry.py       # Фабрика моделей
│   ├── feature_engineering/  # Инженерия признаков
│   │   ├── lag_creator.py    # Лаги t-1, t-2, t-7...
│   │   ├── rolling_stats.py  # Скользящие средние, std
│   │   └── validator.py      # Проверка на lookahead
│   ├── interpretation/       # Интерпретация
│   │   ├── shap_explainer.py
│   │   ├── lime_explainer.py
│   │   ├── text_report.py    # Генерация текстовых отчётов
│   │   └── cached_explainer.py
│   └── preprocessing/        # Предобработка
│       ├── anomaly_detector.py   # Z-score, IQR, STL
│       ├── stationarity_tester.py# ADF тест
│       └── decomposer.py         # Декомпозиция
├── services/                 # Сервисный слой
│   ├── training_service.py
│   ├── forecast_service.py
│   ├── interpretation_service.py
│   └── external_model_service.py
├── storage/                  # Хранение данных
│   ├── postgres/
│   │   ├── models.py         # SQLAlchemy ORM
│   │   ├── crud.py           # Асинхронные CRUD
│   │   └── connection.py     # Engine и сессии
│   ├── redis/
│   │   ├── client.py
│   │   ├── cache.py          # Кэш прогнозов и объяснений
│   │   └── rate_limiter.py   # Счётчики для rate limiting
│   └── minio/
│       ├── client.py         # boto3 клиент
│       ├── model_storage.py  # Сохранение/загрузка моделей
│       ├── ttl_manager.py    # Автоудаление старых файлов
│       └── signed_urls.py    # Подписанные URL
├── workers/                  # Celery задачи
│   ├── celery_app.py         # Конфигурация Celery
│   ├── config.py             # Настройки воркеров
│   ├── signals.py            # Обработчики SIGTERM, SIGSEGV
│   └── tasks/
│       ├── training.py       # Обучение XGBoost, LSTM, Prophet, SARIMA
│       ├── interpretation.py # SHAP, LIME, отчёты
│       └── maintenance.py    # Очистка, архивация, health check
├── config.py                 # Pydantic settings
└── main.py                   # Точка входа FastAPI
```

Кодовая база фронтенда организована следующим образом:

```
Структура фронтенда:
├── api/                      # API-клиенты
│   ├── client.ts             # Axios с интерцепторами
│   ├── auth.ts               # Логин, регистрация, refresh
│   ├── series.ts             # Загрузка, список, удаление
│   ├── training.ts           # Запуск обучения, статус
│   └── forecast.ts           # Прогноз, экспорт
├── contexts/
│   └── AuthContext.tsx       # React Context для авторизации
├── pages/                    # Страницы
│   ├── Login.tsx
│   ├── Register.tsx
│   ├── Dashboard.tsx         # Список рядов
│   ├── Upload.tsx            # Загрузка CSV/Excel
│   ├── Training.tsx          # Обучение с polling
│   ├── Forecast.tsx          # График прогноза
│   └── Interpretation.tsx    # SHAP, LIME, Qwen
├── styles/
│   └── globals.css
├── App.tsx                   # Роутинг
└── main.tsx                  # Точка входа
```

---

## №4. Ключевые функциональные модули

### 1. Управление данными (Series API)

- Загрузка CSV/Excel файлов
- Автоматическое определение разделителей и кодировок
- Парсинг дат и числовых колонок
- Предпросмотр данных и базовая статистика

### 2. Предобработка (Preprocessing API)

| Функция | Методы |
|---------|--------|
| **Детекция аномалий** | Z-score, IQR, STL |
| **Обработка аномалий** | Медиана, интерполяция сплайном, удаление |
| **Стационарность** | ADF-тест + дифференцирование |
| **Декомпозиция** | STL (тренд + сезонность + остаток) |

### 3. Обучение моделей (Training API)

- Асинхронный запуск через Celery
- Поддержка гиперпараметров
- **Ensemble Forecaster**: объединение прогнозов (среднее, медиана, взвешенное)
- Сохранение модели в MinIO + метрик в PostgreSQL

### 4. Прогнозирование (Forecast API)

- Прогноз на заданный горизонт
- Доверительные интервалы (где применимо)
- Экспорт в CSV/JSON
- Пакетный прогноз для нескольких рядов

### 5. Интерпретация (Interpretation API) — **"Фишка" проекта**

| Метод | Описание |
|-------|----------|
| **SHAP** | Глобальная важность лагов + локальный вклад для XGBoost |
| **LIME** | Локальные объяснения для LSTM (аппроксимация линейной моделью) |
| **LLM Report (Qwen)** | Генерация текстового отчета на естественном языке |
| **Кэширование** | Результаты SHAP/LIME/LLM кэшируются в Redis на 24 часа |

---

## №5. Безопасность и Надежность

### Аутентификация и авторизация

- **JWT**: Access Token (короткоживущий) + Refresh Token (в БД)
- Автоматическое обновление токена на клиенте
- Все запросы фильтруются по `user_id`

### Rate Limiting

- Middleware на уровне FastAPI
- Sliding Window через Redis
- Например: 100 req/min

### Обработка ошибок

- Глобальный exception handler
- Стандартизированные JSON-ошибки

### Celery Reliability

| Механизм | Назначение |
|----------|------------|
| `task_acks_late=True` | Подтверждение только после выполнения |
| `retry_backoff` | Экспоненциальная задержка при ошибке |
| Signal Handlers | Корректная обработка SIGTERM/SIGSEGV |

---

## №6. Статус реализации и ограничения

### Реализовано ✅

- Полный CRUD для пользователей и временных рядов
- Обучение XGBoost, Prophet, SARIMA, LSTM (CPU fallback)
- Интеграция SHAP и LIME
- Интеграция с Qwen LLM сервисом
- Детекция и обработка аномалий
- Docker-compose окружение
- Frontend интерфейс для всех сценариев

### Известные ограничения ⚠️

| Ограничение | Описание |
|-------------|----------|
| **Тесты** | Unit/Integration тесты — только заготовки, покрытие низкое |
| **LSTM на CPU** | Медленно и нестабильно без GPU |
| **Qwen Service** | Требует GPU для скорости, без GPU — fallback шаблоны |
| **SARIMA параметры** | Нет полноценного auto-arima, параметры задаются вручную |
| **Восстановление Celery** | При падении воркера задача может потеряться |

---

## №7. Как запустить проект

### 1. Клонировать репозиторий

```
git clone https://github.com/Beda3113/Analysis-of-Time-Series-Forecasting-System-Requirements
cd time-series-platform
```
2. Настроить переменные окружения
```
cp .env.example .env
```
3. Запустить через Docker Compose
```
docker-compose up -d
```
4. (Опционально) Запустить Qwen Service
```
cd qwen/
docker build -t qwen-service .
docker run --gpus all -p 5001:5001 qwen-service
```
# Доступные адреса

| Сервис | URL | Логин/Пароль |
|--------|-----|--------------|
| Web App | http://localhost | — |
| API Docs (Swagger) | http://localhost/docs | — |
| Flower (Celery Monitor) | http://localhost:5555 | admin / admin |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin |

---

## №8. Заключение

Проект представляет собой **зрелое MVP** платформы для прогнозирования. Сильной стороной является глубокая проработка архитектуры (разделение очередей, кэширование, асинхронность) и фокус на интерпретируемости (XAI), что выделяет его среди стандартных учебных проектов.
