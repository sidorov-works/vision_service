## Vision Service

Универсальный сервис для работы с изображениями на основе Vision-Language моделей (VLM). Поддерживает:

- **OCR** — извлечение текста из изображений
- **Описание** — получение текстового описания содержимого изображения
- **Батчевую обработку** — до 20 изображений за запрос

Сервис написан на Python с использованием FastAPI и поддерживает два бэкенда:
- **Transformers** — для продакшена на GPU (CUDA)
- **MLX** — для разработки на Apple Silicon

### Архитектура

```
HTTP-клиент
    │
    ▼
┌─────────────────────────────────────────────────────────┐
│                    FastAPI приложение                   │
│                                                         │
│  Эндпоинты:                                             │
│  POST /ocr          — распознавание текста              │
│  POST /ocr/batch    — батчевое распознавание            │
│  POST /describe     — описание изображения              │
│  POST /describe/batch — батчевое описание               │
│  GET  /health       — проверка здоровья                 │
│  GET  /info         — метаданные сервиса                │
│                                                         │
│  Middleware:                                            │
│  - Rate limiter (slowapi)                               │
│  - CORS                                                 │
│  - Аутентификация по заголовку                          │
└───────────┬─────────────────────────────────────────────┘
            │
            │ asyncio.Queue (input_queue)
            ▼
┌─────────────────────────────────────────────────────────┐
│                      VisionWorker                       │
│                                                         │
│  Единственный владелец модели.                          │
│  Последовательно обрабатывает задачи из очереди.        │
│  Поддерживает бэкенды: Transformers, MLX.               │
└─────────────────────────────────────────────────────────┘
```

**Почему один воркер:** VLM-модели занимают несколько гигабайт GPU-памяти. Загрузить несколько копий невозможно. Очередь работает как семафор — все запросы выстраиваются и обрабатываются последовательно.

### Поддерживаемые модели

| Ключ | Модель | Размер | OCR | Описание | Русский |
|---|---|---|---|---|---|
| `paddle-ocr` | PaddleOCR-VL | ~0.9B | ✅ | ❌ | ❌ |
| `glm-ocr` | GLM-OCR | ~1B | ✅ | ❌ | ❌ |
| `lighton` | LightOnOCR-2-1B | ~1B | ✅ | ❌ | ❌ |
| `smolvlm` | SmolVLM-256M-Instruct | ~1.5GB (6bit) | ✅ | ✅ | ❌ |
| `qwen25-vl` | Qwen2.5-VL-3B-Instruct | ~4GB (8bit) | ✅ | ✅ | ✅ |
| `lfm25-vl` | LFM2.5-VL-450M | ~0.5GB | ✅ | ✅ | ✅ |

Модели, помеченные как "Описание: ✅", могут использоваться в эндпоинтах `/describe`. OCR-специализированные модели работают только с `/ocr`.

#### Добавление новой модели

1. Добавить запись в `MODEL_REGISTRY` в `shared/model_configs.py`
2. Указать `transformers_id`, `mlx_id`, `strategy_key`
3. Указать `default_prompt` и `generation_params`
4. Создать стратегии инференса в `shared/strategies/transformers_strategies.py` и `shared/strategies/mlx_strategies.py`
5. Зарегистрировать стратегии в `STRATEGY_REGISTRY` и `MLX_STRATEGY_REGISTRY`

### Конфигурация

Все настройки задаются через переменные окружения или `.env` файл.

#### Выбор бэкенда и модели

| Переменная | По умолчанию | Описание |
|---|---|---|
| `BACKEND` | `transformers` | Бэкенд: `transformers`, `mlx` |
| `MODEL` | — | Ключ модели из реестра (обязательно) |

#### Transformers (продакшн на GPU)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `DEVICE` | `cuda` или `cpu` | Устройство: `cuda`, `cpu`, `mps` |
| `MODELS_ROOT` | `/app/models` | Папка для кэширования моделей |
| `USE_FLASH_ATTENTION` | `false` | Включить flash attention |

#### MLX (разработка на Mac)

| Переменная | По умолчанию | Описание |
|---|---|---|
| `MLX_MODEL_ID` | из реестра | ID модели в MLX формате (определяется автоматически) |

#### Ограничения

| Переменная | По умолчанию | Описание |
|---|---|---|
| `MAX_IMAGE_SIZE_MB` | `10` | Макс. размер одного изображения |
| `MAX_IMAGE_DIMENSION` | `2000` | Макс. ширина/высота после ресайза |
| `MAX_BATCH_SIZE` | `20` | Макс. количество изображений в батче |
| `MAX_TOTAL_BATCH_MB` | `50` | Макс. общий размер батча |
| `INPUT_QUEUE_MAXSIZE` | `100` | Размер очереди задач |
| `OCR_TIMEOUT` | `60.0` | Таймаут обработки (сек) |

#### Безопасность

| Переменная | По умолчанию | Описание |
|---|---|---|
| `REQUIRE_AUTH` | `true` | Требовать аутентификацию |
| `INTERNAL_API_SECRET` | — | Секретный ключ для заголовка |

#### Rate limiting

| Переменная | По умолчанию | Описание |
|---|---|---|
| `RATE_LIMIT_OCR` | `30/minute` | Лимит для `/ocr`, `/ocr/batch`, `/describe`, `/describe/batch` |
| `RATE_LIMIT_HEALTH` | `100/minute` | Лимит для `/health` и `/info` |

#### Логирование

| Переменная | По умолчанию | Описание |
|---|---|---|
| `LOG_PATH` | `logs` | Папка с логами |
| `LOGGING_LEVEL` | `INFO` | Уровень: `DEBUG`, `INFO`, `WARNING`, `ERROR` |
| `DOCKER_ENV` | `false` | Режим логирования для Docker |

### Быстрый старт

#### Установка зависимостей

```bash
pip install -r requirements.txt
```

#### Локально на Mac (MLX)

```bash
# SmolVLM (лёгкая, ~1.5GB)
BACKEND=mlx MODEL=smolvlm uvicorn main:app --port 8299

# LFM2.5-VL (мультиязычная, ~0.5GB)
BACKEND=mlx MODEL=lfm25-vl uvicorn main:app --port 8299
```

#### Локально на GPU (Transformers)

```bash
# LFM2.5-VL (рекомендуется для русского языка)
BACKEND=transformers MODEL=lfm25-vl DEVICE=cuda uvicorn main:app --port 8080

# Qwen2.5-VL (требует ~8GB VRAM)
BACKEND=transformers MODEL=qwen25-vl DEVICE=cuda uvicorn main:app --port 8080
```

#### Docker (production)

```bash
# Сборка образа
docker build -t vision-service .

# Запуск с GPU
docker run -d \
  --gpus all \
  -p 8080:8080 \
  -e BACKEND=transformers \
  -e MODEL=lfm25-vl \
  -e DEVICE=cuda \
  -e MODELS_ROOT=/app/models \
  -v ./models:/app/models \
  -v ./logs:/app/logs \
  --name vision-service \
  vision-service
```

### API

#### POST /ocr

Распознавание текста из одного изображения.

**Запрос:**

```json
{
  "image_base64": "data:image/png;base64,iVBORw0KGgo...",
  "prompt": "Text Recognition:"
}
```

Поле `prompt` опционально. Если не указано, используется дефолтный промпт модели.

**Ответ:**

```json
{
  "success": true,
  "task_id": "550e8400-e29b-41d4-a716-446655440000",
  "text": "Распознанный текст",
  "processing_time_ms": 1234.5,
  "error": null
}
```

#### POST /ocr/batch

Пакетное распознавание нескольких изображений.

**Запрос:**

```json
{
  "images": ["base64...", "base64..."],
  "prompts": ["prompt1", "prompt2"],
  "common_prompt": "Text Recognition:"
}
```

**Ответ:**

```json
{
  "success": true,
  "task_id": "batch-uuid",
  "results": [
    {
      "success": true,
      "task_id": "batch_0_uuid",
      "text": "Текст первого изображения",
      "processing_time_ms": 800.0,
      "error": null
    }
  ],
  "processing_time_ms": 2456.7,
  "error": null
}
```

#### POST /describe

Описание изображения.

**Запрос:**

```json
{
  "image_base64": "data:image/png;base64,...",
  "prompt": "Describe this image in detail."
}
```

**Ответ:**

```json
{
  "success": true,
  "task_id": "550e8400...",
  "description": "На изображении изображён...",
  "processing_time_ms": 1234.5,
  "error": null
}
```

#### POST /describe/batch

Пакетное описание нескольких изображений. Аналогичен `/ocr/batch`, но с полем `description` вместо `text`.

#### GET /health

Проверка здоровья сервиса.

```json
{
  "status": "healthy",
  "backend": "transformers",
  "model": "lfm25-vl",
  "tasks_processed": 42,
  "queue_size": 3
}
```

#### GET /info

Метаданные сервиса.

```json
{
  "service": "Vision Service",
  "version": "2.0.0",
  "backend": "transformers",
  "model": "lfm25-vl",
  "allowed_mime_types": ["image/jpeg", "image/png", ...],
  "max_image_size_mb": 10,
  "max_queue_size": 100,
  "max_batch_size": 20
}
```

### HTTP клиент для взаимодействия с сервисом

Для работы с Vision Service используется `RetryableHTTPClient` из пакета `http_utils`, который поддерживает автоматические ретраи и аутентификацию.

#### Установка

```python
from http_utils.http_client import RetryableHTTPClient
from http_utils import create_signed_client, AuthType
```

#### Параметры клиента

| Параметр | Значение | Для Vision Service |
|---|---|---|
| `base_timeout` | Таймаут одного запроса | `60.0` (сек) |
| `max_retries` | Кол-во повторных попыток | `2` |
| `base_delay` | Начальная задержка | `1.0` (сек) |
| `total_timeout` | Общий таймаут всех попыток | `120.0` (сек) |

#### Аутентификация

Сервис требует заголовок `X-Internal-Secret`. Для автоматической подписи используйте `create_signed_client`:

```python
client = create_signed_client(
    RetryableHTTPClient(base_timeout=60.0, max_retries=2),
    secret="my_secret_key",
    auth_type=AuthType.SECRET_HEADER_AUTH
)
```

#### Пример: описание изображения

```python
import base64

async with create_signed_client(
    RetryableHTTPClient(base_timeout=60.0, max_retries=2),
    secret="my_secret_key",
    auth_type=AuthType.SECRET_HEADER_AUTH
) as client:
    
    with open("image.png", "rb") as f:
        image_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    response = await client.post_with_retry(
        "http://localhost:8299/describe",
        json={"image_base64": image_base64}
    )
    
    result = response.json()
    if result["success"]:
        description = result["description"]
        print(f"Описание: {description}")
    else:
        print(f"Ошибка: {result['error']}")
```

### Структура проекта

```
vision-service/
├── main.py                          # FastAPI приложение
├── workers/
│   └── vision_worker.py             # Воркер (VisionWorker)
├── shared/
│   ├── config.py                    # Конфигурация
│   ├── model_configs.py             # Реестр моделей
│   ├── schemas.py                   # Pydantic модели
│   ├── prepare_image.py             # Ресайз изображений
│   ├── auth_service.py              # Аутентификация
│   ├── strategies/
│   │   ├── transformers_strategies.py   # Стратегии для Transformers
│   │   └── mlx_strategies.py            # Стратегии для MLX
│   └── backend/
│       ├── base.py                  # Базовый класс VisionBackend
│       ├── transformers.py          # Transformers бэкенд
│       └── mlx.py                   # MLX бэкенд
├── requirements.txt                 # Зависимости
├── Dockerfile                       # Docker образ
└── docker-compose.yml               # Docker Compose
```

### Лицензия

MIT