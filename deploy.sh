#!/bin/bash

# Скрипт для развертывания ML API на Ubuntu/Debian VPS

set -e

echo "🚀 Начинаем развертывание Glucose Prediction API..."

# Обновляем систему
echo "📦 Обновляем систему..."
sudo apt update && sudo apt upgrade -y

# Устанавливаем Docker и Docker Compose
if ! command -v docker &> /dev/null; then
    echo "🐳 Устанавливаем Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sudo sh get-docker.sh
    sudo usermod -aG docker $USER
    rm get-docker.sh
fi

if ! command -v docker-compose &> /dev/null; then
    echo "🔧 Устанавливаем Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Создаем директорию для приложения
APP_DIR="/opt/glucose-api"
sudo mkdir -p $APP_DIR
sudo chown $USER:$USER $APP_DIR
cd $APP_DIR

# Создаем файлы конфигурации
echo "📝 Создаем конфигурационные файлы..."

# app.py уже должен быть скопирован вручную
# Или можно скачать из репозитория:
# wget -O app.py https://raw.githubusercontent.com/your-repo/app.py

# requirements.txt
cat > requirements.txt << EOF
flask==2.3.3
joblib==1.3.2
numpy==1.24.3
scikit-learn==1.3.0
requests==2.31.0
gunicorn==21.2.0
EOF

# Dockerfile
cat > Dockerfile << 'EOF'
FROM python:3.9-slim

RUN apt-get update && apt-get install -y \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

EXPOSE 8000

RUN adduser --disabled-password --gecos '' appuser && chown -R appuser /app
USER appuser

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "2", "--timeout", "120", "app:app"]
EOF

# docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  glucose-api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - FLASK_ENV=production
      - PORT=8000
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "curl -f http://localhost:8000/ || exit 1"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    depends_on:
      - glucose-api
    restart: unless-stopped
EOF

# nginx.conf (упрощенная версия)
cat > nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    upstream glucose_api {
        server glucose-api:8000;
    }

    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=10r/m;

    server {
        listen 80;
        
        location / {
            limit_req zone=api_limit burst=20 nodelay;
            
            proxy_pass http://glucose_api;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 60s;
            
            client_max_body_size 10M;
        }

        location /health {
            proxy_pass http://glucose_api/;
            access_log off;
        }
    }
}
EOF

# Устанавливаем ufw (firewall)
echo "🔒 Настраиваем firewall..."
sudo ufw --force reset
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw --force enable

# Создаем systemd service для автозапуска
echo "🔄 Создаем systemd service..."
sudo tee /etc/systemd/system/glucose-api.service > /dev/null << EOF
[Unit]
Description=Glucose Prediction API
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=true
WorkingDirectory=$APP_DIR
ExecStart=/usr/local/bin/docker-compose up -d
ExecStop=/usr/local/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable glucose-api.service

# Запускаем приложение
echo "🚀 Запускаем приложение..."
docker-compose build
docker-compose up -d

# Ждем, пока сервис поднимется
echo "⏳ Ждем запуска сервиса..."
sleep 30

# Проверяем статус
if curl -f http://localhost:80/ > /dev/null 2>&1; then
    echo "✅ API успешно развернут!"
    echo "🌐 Ваш API доступен по адресу: http://$(curl -s ipinfo.io/ip)"
    echo "📍 Тестовый endpoint: http://$(curl -s ipinfo.io/ip)/health"
else
    echo "❌ Что-то пошло не так. Проверьте логи:"
    docker-compose logs
fi

echo "📊 Для мониторинга используйте:"
echo "  - docker-compose logs -f"
echo "  - docker-compose ps"
echo "  - systemctl status glucose-api"

echo "🔄 Для перезапуска:"
echo "  - docker-compose restart"
echo "  - sudo systemctl restart glucose-api"

echo "🎉 Развертывание завершено!"