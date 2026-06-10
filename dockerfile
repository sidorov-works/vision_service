FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    git \ 
    wget \
    && rm -rf /var/lib/apt/lists/*

# Копирование зависимостей
COPY requirements.prod.txt .
RUN pip install --no-cache-dir -r requirements.prod.txt

# Копирование кода
COPY . .

# Создание директории для моделей
RUN mkdir -p /app/models

# Переменные окружения
ENV PYTHONPATH=/app
ENV MODELS_ROOT=/app/models
ENV OCR_BACKEND=transformers
ENV DEVICE=cuda
ENV DOCKER_ENV=true

# Порт сервера
EXPOSE 8080

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]