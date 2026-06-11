### Для продакшен: добавление моделей и сборка Docker образов

#### Быстрый старт

**1. Сборка всех образов:**
```bash
chmod +x build.sh
./build.sh
```

**2. Запуск сервисов:**
```bash
docker-compose up -d
```

**3. Проверка работы:**
```bash
curl http://localhost:8249/health  # LFM25-VL
curl http://localhost:8349/health  # Qwen2.5-VL
curl http://localhost:8399/health  # GLM-OCR
```

---

#### Архитектура сборки

Проект использует **многослойную сборку Docker**:

```
┌─────────────────────────────────────────────────────────────┐
│                    Dockerfile.base                          │
│  (PyTorch + CUDA + общие зависимости + общий код)           │
│  Собирается один раз, используется всеми моделями           │
└─────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│Dockerfile.lfm25│    │Dockerfile.qwen│    │Dockerfile.glm │
│ + специфичные  │    │ + специфичные  │    │ + специфичные  │
│ зависимости    │    │ зависимости    │    │ зависимости    │
└───────────────┘    └───────────────┘    └───────────────┘
```

**Преимущества:**
- Общий код и базовые зависимости не дублируются
- Каждая модель получает свои версии библиотек (нет конфликтов)
- Пересборка конкретной модели не требует пересборки всего

---

#### Добавление новой модели

##### Шаг 1: Добавить модель в реестр (`shared/model_configs.py`)

```python
"новая-модель": ModelConfig(
    transformers_id="author/model-name",      # ID на HuggingFace
    mlx_id="mlx-community/model-name",        # ID для MLX (если есть)
    strategy_key="новая-модель",              # Ключ стратегии
    default_prompt="Describe this image.",    # Промпт по умолчанию
    generation_params={
        "max_new_tokens": 512,
        "do_sample": False,
        "temperature": 0.1,
    }
),
```

##### Шаг 2: Создать стратегию инференса (`shared/strategies/transformers_strategies.py`)

```python
class NewModelStrategy(InferenceStrategy):
    def __init__(self):
        self.model = None
        self.processor = None

    async def load_model(self, model_id: str, cache_dir: Path, device: str):
        from transformers import AutoModelForCausalLM, AutoProcessor
        
        def _load():
            processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True, cache_dir=str(cache_dir)
            )
            torch_dtype, device_map = _get_dtype_and_device_map(device)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, trust_remote_code=True, 
                dtype=torch_dtype, device_map=device_map, cache_dir=str(cache_dir)
            )
            model.eval()
            return processor, model
        
        self.processor, self.model = await asyncio.to_thread(_load)

    async def infer(self, image_bytes: bytes, prompt: str, generation_params=None):
        # Реализация инференса
        pass

    def cleanup(self):
        if self.model:
            del self.model
            torch.cuda.empty_cache()

# Добавить в реестр стратегий
STRATEGY_REGISTRY["новая-модель"] = NewModelStrategy
```

##### Шаг 3: Создать Dockerfile для модели

```dockerfile
# Dockerfile.newmodel
FROM vision-base:latest

COPY requirements.newmodel.txt .
RUN pip install --no-cache-dir -r requirements.newmodel.txt

ENV BACKEND=transformers
ENV MODEL=новая-модель
ENV DEVICE=cuda

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
```

##### Шаг 4: Создать файл зависимостей

```txt
# requirements.newmodel.txt
transformers==4.57.0
accelerate==1.13.0
# другие специфичные зависимости
```

##### Шаг 5: Добавить сервис в `docker-compose.yml`

```yaml
newmodel:
  build:
    context: .
    dockerfile: Dockerfile.newmodel
  image: vision-newmodel:latest
  container_name: vision-newmodel
  ports:
    - "8499:8080"
  environment:
    - MODEL=новая-модель
    - HF_TOKEN=${HF_TOKEN}
    - MODELS_ROOT=/app/models
  volumes:
    - ./models:/app/models
    - ./logs:/app/logs
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
  restart: unless-stopped
```

##### Шаг 6: Добавить сборку в `build.sh`

```bash
echo -e "${GREEN}[5/5] Building NewModel image...${NC}"
docker build -f Dockerfile.newmodel -t vision-newmodel:latest .
```

---

#### Важные замечания по зависимостям

**Разные модели могут требовать разные версии transformers:**

| Модель | Версия transformers | Особенности |
|--------|---------------------|-------------|
| LFM2.5-VL | 4.57.0 (PyPI) | Стандартная установка |
| Qwen2.5-VL | 4.57.0 + qwen-vl-utils | Дополнительная библиотека |
| GLM-OCR | >=5.1.0 (из GitHub) | Установка из исходников |

**Почему нельзя один requirements.txt:**
- GLM-OCR требует `git+https://github.com/huggingface/transformers.git`
- Qwen требует `qwen-vl-utils==0.0.10`
- LFM работает со стандартным пакетом

Изоляция через разные Dockerfile решает эту проблему.

---

#### Рабочий процесс разработки

**Локальная разработка (Mac с MLX):**
```bash
# Не нужны Dockerfile.base и сложная сборка
BACKEND=mlx MODEL=smolvlm uvicorn main:app --reload
```

**Продакшен (CUDA сервер):**
```bash
# Сборка и запуск через Docker
./build.sh && docker-compose up -d
```

---

#### Команды управления

```bash
# Сборка всех образов
./build.sh

# Сборка только конкретной модели
docker build -f Dockerfile.qwen -t vision-qwen:latest .

# Запуск всех сервисов
docker-compose up -d

# Запуск конкретного сервиса
docker-compose up -d qwen25-vl

# Просмотр логов
docker-compose logs -f glm-ocr

# Перезапуск сервиса после изменений
docker-compose restart lfm25-vl

# Остановка всех сервисов
docker-compose down

# Очистка неиспользуемых образов
docker image prune -a
```

---

#### Диагностика проблем

**Проблема: образ vision-base не найден**
```bash
# Решение: сначала собрать базовый образ
docker build -f Dockerfile.base -t vision-base:latest .
```

**Проблема: не хватает VRAM**
```yaml
# В docker-compose.yml добавьте ограничения
deploy:
  resources:
    limits:
      memory: 16G
```

**Проблема: модель долго скачивается**
```yaml
# Первый запуск всегда долгий. Используйте volumes для кэша
volumes:
  - ./models:/app/models  # Модели сохранятся локально
```

**Проблема: GLM-OCR не загружается**
```bash
# Проверьте HF_TOKEN и что модель не gated
# GLM-OCR требует подтверждения доступа на HuggingFace
```

---

#### Структура файлов проекта

```
project/
├── Dockerfile.base           # Базовый слой (PyTorch + общий код)
├── Dockerfile.lfm25          # Специфичный для LFM2.5-VL
├── Dockerfile.qwen           # Специфичный для Qwen2.5-VL
├── Dockerfile.glm            # Специфичный для GLM-OCR
├── docker-compose.yml        # Оркестрация всех сервисов
├── requirements.base.txt     # Общие зависимости
├── requirements.lfm25.txt    # Зависимости LFM2.5-VL
├── requirements.qwen.txt     # Зависимости Qwen2.5-VL
├── requirements.glm.txt      # Зависимости GLM-OCR
├── .env                      # Переменные окружения
├── build.sh                  # Скрипт сборки образов
├── main.py                   # FastAPI приложение
├── shared/                   # Общий код
│   ├── config.py
│   ├── model_configs.py      # Реестр моделей ← добавлять сюда
│   ├── schemas.py
│   ├── prepare_image.py
│   ├── backend/
│   │   ├── base.py
│   │   └── transformers.py
│   └── strategies/
│       └── transformers_strategies.py ← добавлять стратегии сюда
└── workers/
    └── vision_worker.py
```