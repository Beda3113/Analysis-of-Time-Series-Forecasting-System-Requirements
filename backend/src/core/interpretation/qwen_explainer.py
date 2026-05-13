import aiohttp
import asyncio
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class QwenLagExplainer:
    """Реальный Qwen-7B LLM объяснитель через API"""
    
    def __init__(self, api_url: str = "http://localhost:8001", use_real_llm: bool = True):
        self.api_url = api_url
        self.use_real_llm = use_real_llm
        self._available = None
    
    async def _check_availability(self) -> bool:
        """Проверка доступности Qwen-7B сервиса"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.api_url}/health", timeout=aiohttp.ClientTimeout(total=5)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("model_loaded", False)
            return False
        except Exception as e:
            logger.warning(f"Qwen-7B service not available: {e}")
            return False
    
    async def explain_lags(
        self, 
        important_lags: List[int], 
        series_name: Optional[str] = None, 
        language: str = "ru"
    ) -> str:
        """
        Реальный вызов Qwen-7B для объяснения лагов
        """
        if self._available is None:
            self._available = await self._check_availability()
        
        if not self._available or not self.use_real_llm:
            logger.warning("Qwen-7B not available, using fallback explanation")
            return self._generate_fallback_explanation(important_lags, series_name, language)
        
        prompt = self._generate_prompt(important_lags, series_name, language)
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.api_url}/explain",
                    json={
                        "important_lags": important_lags,
                        "series_name": series_name,
                        "language": language
                    },
                    timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("explanation", self._generate_fallback_explanation(important_lags, series_name, language))
                    else:
                        logger.error(f"Qwen-7B API error: {resp.status}")
                        return self._generate_fallback_explanation(important_lags, series_name, language)
        except Exception as e:
            logger.error(f"Qwen-7B request failed: {e}")
            return self._generate_fallback_explanation(important_lags, series_name, language)
    
    def _generate_prompt(self, important_lags: List[int], series_name: Optional[str], language: str) -> str:
        """Генерация промпта для LLM (используется, если API не отвечает)"""
        if language == "ru":
            return f"""Ты эксперт по анализу временных рядов.
Ряд: {series_name if series_name else "Неизвестный ряд"}
Важные лаги: {', '.join([f"t-{lag}" for lag in important_lags])}

Объясни:
1. Что означают эти лаги
2. Как они влияют на прогноз
3. Дай практические рекомендации

Ответ должен быть на русском языке.
"""
        else:
            return f"""You are a time series analysis expert.
Series: {series_name if series_name else "Unknown series"}
Important lags: {', '.join([f"t-{lag}" for lag in important_lags])}

Explain:
1. What these lags mean
2. How they affect the forecast
3. Provide practical recommendations

Answer in English.
"""
    
    def _generate_fallback_explanation(self, important_lags: List[int], series_name: Optional[str], language: str) -> str:
        """Fallback объяснение когда Qwen-7B недоступен (но это уже не заглушка, а информационное сообщение)"""
        if language == "ru":
            explanation = f"""Анализ временного ряда "{series_name if series_name else 'Неизвестный ряд'}"

На основе анализа данных выявлены важные лаги: {', '.join(map(str, important_lags[:5]))}

Для получения AI-объяснения необходимо запустить Qwen-7B сервис.
Текущие выводы основаны на SHAP анализе модели.
"""
        else:
            explanation = f"""Time series analysis "{series_name if series_name else 'Unknown series'}"

Important lags identified: {', '.join(map(str, important_lags[:5]))}

To get AI explanation, please start Qwen-7B service.
Current conclusions are based on SHAP analysis.
"""
        return explanation