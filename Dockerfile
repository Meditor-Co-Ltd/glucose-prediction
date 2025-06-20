FROM python:3.9-slim

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY app.py .

# Скачиваем модель при сборке контейнера (опционально)
# RUN python -c "
# import requests
# import os
# if not os.path.exists('model.pkl'):
#     url = 'https://firebasestorage.googleapis.com/v0/b/innomax-40d4d.appspot.com/o/random_forest_model_0740_rmse_17.pkl?alt=media&token=35720e11-2a75-4e6e-82f2-50c635043837'
#     response = requests.get(url)
#     with open('model.pkl', 'wb') as f:
#         f.write(response.content)
# "

# Открываем порт
EXPOSE 8000

# Создаем пользователя для безопасности
RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

# Команда запуска
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "app:app"]