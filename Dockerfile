FROM python:3.9-slim

# Set environment variables to force CPU usage
ENV CUDA_VISIBLE_DEVICES=""
ENV TORCH_DEVICE="cpu"

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
COPY utils.py .
COPY probablistic_model_NORMAL_traced.pt .
COPY NORMAL_average.pkl .
COPY NORMAL_average0.pkl .
COPY NORMAL_average50.pkl .
COPY NORMAL_average100.pkl .
COPY probablistic_model_PREDIABETIC_traced.pt .
COPY PREDIABETIC_average.pkl .
COPY PREDIABETIC_average0.pkl .
COPY PREDIABETIC_average50.pkl .
COPY PREDIABETIC_average100.pkl .
COPY probablistic_model_DIABETIC_traced.pt .
COPY DIABETIC_average.pkl .
COPY DIABETIC_average0.pkl .
COPY DIABETIC_average50.pkl .
COPY DIABETIC_average100.pkl .
COPY classification_model_binwidth10.pt .

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
