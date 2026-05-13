#!/usr/bin/env python3
"""
Скачивание реальных данных с использованием yfinance (рабочая версия)
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def download_yahoo_finance():
    """Скачивание данных курса валют/акций с Yahoo Finance"""
    print(" Загрузка данных с Yahoo Finance...")
    
    try:
        import yfinance as yf
        
        # Скачиваем курс USD/RUB
        print("   Загрузка USD/RUB...")
        usd = yf.download("RUB=X", period="1y", interval="1d", progress=False)
        if len(usd) > 0:
            usd = usd[['Close']].rename(columns={'Close': 'value'})
            usd.index.name = 'date'
            usd.reset_index().to_csv("real_data/usd_rub.csv", index=False)
            print(f"    USD/RUB: {len(usd)} дней")
        
        # Скачиваем S&P 500
        print("   Загрузка S&P 500...")
        sp500 = yf.download("^GSPC", period="1y", interval="1d", progress=False)
        if len(sp500) > 0:
            sp500 = sp500[['Close']].rename(columns={'Close': 'value'})
            sp500.index.name = 'date'
            sp500.reset_index().to_csv("real_data/sp500.csv", index=False)
            print(f"    S&P 500: {len(sp500)} дней")
        
        # Скачиваем Bitcoin
        print("   Загрузка Bitcoin...")
        btc = yf.download("BTC-USD", period="6mo", interval="1d", progress=False)
        if len(btc) > 0:
            btc = btc[['Close']].rename(columns={'Close': 'value'})
            btc.index.name = 'date'
            btc.reset_index().to_csv("real_data/bitcoin.csv", index=False)
            print(f"    Bitcoin: {len(btc)} дней")
            
        return True
        
    except Exception as e:
        print(f"    Ошибка загрузки: {e}")
        return False

def generate_realistic_data():
    """Генерация реалистичных данных (если не удалось загрузить реальные)"""
    print("\n Генерация реалистичных данных...")
    
    # 1. Продажи интернет-магазина
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    n = len(dates)
    
    # Тренд + сезонность
    trend = np.linspace(100, 180, n)
    weekly = 30 * np.sin(2 * np.pi * np.arange(n) / 7)
    monthly = 15 * np.sin(2 * np.pi * np.arange(n) / 30)
    noise = np.random.normal(0, 10, n)
    
    sales = trend + weekly + monthly + noise
    sales = np.maximum(sales, 20)
    
    df_sales = pd.DataFrame({'date': dates, 'value': sales})
    df_sales.to_csv("real_data/sales_data.csv", index=False)
    print(f"    Продажи: {len(df_sales)} дней")
    
    # 2. Данные температуры
    day_of_year = np.arange(n)
    temp_base = 15 + 15 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)
    temp_noise = np.random.normal(0, 4, n)
    temperature = temp_base + temp_noise
    
    df_temp = pd.DataFrame({'date': dates, 'value': temperature})
    df_temp.to_csv("real_data/temperature.csv", index=False)
    print(f"    Температура: {len(df_temp)} дней")
    
    # 3. Данные загрузки CPU (почасовые)
    hours = pd.date_range(start='2024-01-01', periods=720, freq='h')
    hour_of_day = hours.hour
    
    cpu_load = 30 + 25 * np.sin(2 * np.pi * hour_of_day / 24 - np.pi/2)
    cpu_load += 10 * np.sin(2 * np.pi * np.arange(720) / 168)
    cpu_load += np.random.normal(0, 5, 720)
    cpu_load = np.clip(cpu_load, 5, 95)
    
    df_cpu = pd.DataFrame({'date': hours, 'value': cpu_load})
    df_cpu.to_csv("real_data/cpu_load.csv", index=False)
    print(f"    CPU Load: {len(df_cpu)} часов")
    
    return True

# Основной блок
if __name__ == "__main__":
    os.makedirs("real_data", exist_ok=True)
    
    print("="*60)
    print(" ЗАГРУЗКА РЕАЛЬНЫХ ДАННЫХ")
    print("="*60)
    
    # Пытаемся загрузить реальные данные
    success = download_yahoo_finance()
    
    # Генерируем дополнительные данные
    generate_realistic_data()
    
    print("\n" + "="*60)
    print(" ДАННЫЕ ГОТОВЫ")
    print("="*60)
    print(" Файлы в папке 'real_data':")
    for f in os.listdir("real_data"):
        size = os.path.getsize(f"real_data/{f}") / 1024
        print(f"   - {f} ({size:.1f} KB)")
