"""
Qwen LLM Service для объяснения лагов временных рядов (оптимизированная версия)
"""

import os
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import logging
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Настройки (оптимизированные для Qwen2-1.5B)
MODEL_PATH = os.environ.get("MODEL_PATH", "Qwen/Qwen2-1.5B-Instruct")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
QUANTIZATION = os.environ.get("QUANTIZATION", "")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", 400))
TEMPERATURE = float(os.environ.get("TEMPERATURE", 0.7))

# Глобальные переменные для модели
model = None
tokenizer = None


class ExplainRequest(BaseModel):
    """Запрос на объяснение лагов"""
    important_lags: List[int]
    series_name: Optional[str] = None
    language: str = "ru"


class ExplainResponse(BaseModel):
    """Ответ с объяснением"""
    explanation: str
    important_lags: List[int]
    model_used: str


def load_model():
    """Загрузка модели Qwen"""
    global model, tokenizer
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info(f"Loading model from {MODEL_PATH}")
        logger.info(f"Device: {DEVICE}")
        
        # Загрузка токенизатора
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH, 
            trust_remote_code=True
        )
        
        # Устанавливаем pad_token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Параметры загрузки модели
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if DEVICE == "cuda" else torch.float32,
        }
        
        # Оптимизация для CUDA
        if DEVICE == "cuda":
            model_kwargs["device_map"] = "auto"
        
        # Загрузка модели
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            **model_kwargs
        )
        
        # Режим оценки для ускорения
        model.eval()
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.warning(f"Failed to load model: {e}")
        logger.info("Using fallback mode")
        model = None
        tokenizer = None


def generate_fallback_explanation(important_lags: List[int], series_name: str = None, language: str = "ru") -> str:
    """Fallback объяснение без реальной LLM"""
    name_part = f' "{series_name}"' if series_name else ''
    
    if language == "ru":
        explanation = f"""📊 **Анализ временного ряда**{name_part}

На основе анализа данных были выявлены следующие важные лаги:

🔍 **Ключевые выводы:**
"""
        for lag in important_lags[:3]:
            if lag == 1:
                explanation += f"- **Лаг {lag}** (вчерашнее значение) - оказывает наибольшее влияние на прогноз\n"
            elif lag == 7:
                explanation += f"- **Лаг {lag}** (недельная давность) - обнаружена еженедельная сезонность\n"
            elif lag == 14:
                explanation += f"- **Лаг {lag}** (две недели назад) - поддерживает долгосрочный тренд\n"
            elif lag == 30:
                explanation += f"- **Лаг {lag}** (месячная давность) - отражает месячные циклы\n"
            else:
                explanation += f"- **Лаг {lag}** - значимый предиктор для прогноза\n"
        
        explanation += f"""
💡 **Рекомендации:**
- Уделите особое внимание значениям за последние {important_lags[0] if important_lags else 7} дней
- Рассмотрите возможность добавления дополнительных сезонных признаков

📈 **Статистика:**
- Всего проанализировано лагов: {len(important_lags)}
- Наиболее важный лаг: {important_lags[0] if important_lags else 'N/A'}
"""
    else:
        explanation = f"Based on the analysis, the most important lags are: {', '.join(map(str, important_lags[:5]))}. "
        explanation += f"Pay special attention to lag {important_lags[0]} which has the strongest influence on the forecast."
    
    return explanation


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    logger.info("Starting Qwen LLM Service...")
    load_model()
    yield
    logger.info("Shutting down Qwen LLM Service...")


# Создание приложения с lifespan
app = FastAPI(
    title="Qwen LLM Service",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": DEVICE,
        "quantization": QUANTIZATION or "off"
    }


@app.post("/explain", response_model=ExplainResponse)
async def explain_lags(request: ExplainRequest):
    """
    Объяснение важности лагов с помощью Qwen (оптимизировано для скорости)
    """
    if model is None or tokenizer is None:
        explanation = generate_fallback_explanation(
            request.important_lags, 
            request.series_name, 
            request.language
        )
        return ExplainResponse(
            explanation=explanation,
            important_lags=request.important_lags,
            model_used="fallback"
        )
    
    # Улучшенный промпт для максимальной информативности
    lags_list = ", ".join([str(lag) for lag in request.important_lags])
    series_name = request.series_name or "данного ряда"
    
    if request.language == "ru":
        prompt = f"""<|im_start|>system
Ты — эксперт по анализу временных рядов с 10-летним опытом. Твоя задача — объяснить пользователю значение лагов простым и понятным языком, как будто ты объясняешь другу. Используй аналогии из жизни. Дай практические советы.
<|im_end|>
<|im_start|>user
У меня есть временной ряд "{series_name}". 
Моя модель прогнозирования определила следующие важные лаги: {lags_list}.

Пожалуйста, объясни:
1. Что означает каждый лаг простыми словами (например, лаг 1 = вчера)
2. Как эти лаги влияют на прогноз
3. Какие практические выводы я могу сделать для бизнеса/анализа
4. Что мне делать с этой информацией?

Ответь дружелюбно, понятно неспециалисту.
<|im_end|>
<|im_start|>assistant
"""
    else:
        prompt = f"""<|im_start|>system
You are a time series analysis expert with 10 years of experience. Explain lags to a non-technical user in simple, friendly language. Use real-life analogies.
<|im_end|>
<|im_start|>user
I have a time series "{series_name}". 
My forecasting model identified important lags: {lags_list}.

Please explain:
1. What each lag means in simple words
2. How these lags affect the forecast
3. What practical conclusions can I make
4. What should I do with this information?

Answer in a friendly, easy-to-understand way.
<|im_end|>
<|im_start|>assistant
"""
    
    try:
        # Токенизация
        inputs = tokenizer(
            prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        ).to(model.device)
        
        # Оптимизированная генерация
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                do_sample=False,
                top_p=0.95,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_beams=1,
                early_stopping=True
            )
        
        # Декодирование
        full_output = tokenizer.decode(outputs[0], skip_special_tokens=False)
        
        # Извлекаем ответ
        if "<|im_start|>assistant" in full_output:
            explanation = full_output.split("<|im_start|>assistant")[-1].strip()
            if "<|im_end|>" in explanation:
                explanation = explanation.split("<|im_end|>")[0].strip()
        else:
            input_length = inputs['input_ids'].shape[1]
            generated_ids = outputs[0][input_length:]
            explanation = tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Пост-обработка для читаемости
        explanation = explanation.replace("  ", " ").strip()
        
        if not explanation or len(explanation) < 50:
            logger.warning(f"Generated explanation too short, using fallback")
            explanation = generate_fallback_explanation(
                request.important_lags, 
                request.series_name, 
                request.language
            )
            return ExplainResponse(
                explanation=explanation,
                important_lags=request.important_lags,
                model_used="fallback"
            )
        
        return ExplainResponse(
            explanation=explanation,
            important_lags=request.important_lags,
            model_used="qwen-1.5b"
        )
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        explanation = generate_fallback_explanation(
            request.important_lags, 
            request.series_name, 
            request.language
        )
        return ExplainResponse(
            explanation=explanation,
            important_lags=request.important_lags,
            model_used="fallback"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)