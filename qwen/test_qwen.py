#!/usr/bin/env python
"""
Тестовый скрипт для диагностики Qwen-7B
"""

import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Настройки
MODEL_PATH = "Qwen/Qwen-7B-Chat"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Loading model from {MODEL_PATH}")
print(f"Device: {DEVICE}")

# Загрузка токенизатора
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Загрузка модели
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

print("Model loaded successfully!")

# Тестовый промпт
prompt = """<|im_start|>system
Ты — эксперт по анализу временных рядов.
<|im_end|>
<|im_start|>user
Объясни, что означает лаг 1 для прогноза временного ряда.
Ответь коротко, 2-3 предложения.
<|im_end|>
<|im_start|>assistant
"""

print(f"\nPrompt length: {len(prompt)} chars")
print("-" * 50)

# Токенизация
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)

# Генерация
print("Generating...")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

# Декодирование
input_length = inputs['input_ids'].shape[1]
generated_ids = outputs[0][input_length:]
response = tokenizer.decode(generated_ids, skip_special_tokens=True)

print(f"\nResponse: {response}")
print(f"Response length: {len(response)} chars")

if len(response) < 20:
    print("\n WARNING: Response is too short!")
    print("Full output:")
    print(tokenizer.decode(outputs[0], skip_special_tokens=False))