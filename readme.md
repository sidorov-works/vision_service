## Сборка Docker образов - подробная инструкция

### 1. Подготовка

Убедитесь, что структура проекта правильная:

```
project/
├── build.sh                 # Скрипт сборки
├── docker-compose.yml       # Запуск сервисов
├── .env                     # Переменные окружения (HF_TOKEN обязательно)
├── dockerfiles/
│   ├── Dockerfile.base
│   ├── Dockerfile.lfm25
│   ├── Dockerfile.qwen
│   ├── Dockerfile.glm
│   └── Dockerfile.lighton
├── requirements/
│   ├── base.txt
│   ├── lfm25.txt
│   ├── qwen.txt
│   ├── glm.txt
│   └── lighton.txt
└── (остальной код: main.py, shared/, workers/)
```

### 2. Дать права на выполнение скрипта (только первый раз)

```bash
chmod +x build.sh
```

Без этого команда `./build.sh` выдаст ошибку "Permission denied".

---

### 3. Сборка образов

#### Собрать все модели (обычный запуск)

```bash
./build.sh
```

Что произойдёт:
- Соберётся базовый образ `vision-base:latest`
- Последовательно соберутся все модели: lfm25, qwen, glm, lighton
- Каждая модель получит свой образ: `vision-lfm25:latest`, `vision-qwen:latest` и т.д.

**Время сборки:** 10-30 минут (скачиваются зависимости), повторно 1-3 минуты (используется кэш).

#### Собрать только одну модель

```bash
./build.sh qwen
```

Когда использовать:
- Вы добавили новую стратегию только для Qwen
- У вас мало места на диске
- Вы тестируете одну модель и не хотите ждать сборку остальных

После этой команды нужно запускать только соответствующий сервис:
```bash
docker-compose up -d qwen25-vl
```

**Важно:** Если вы меняли `Dockerfile.base` или общие файлы (shared/, workers/), нужно сначала пересобрать base:
```bash
./build.sh base      # пересобрать базовый слой
./build.sh qwen      # потом модель
```

#### Сборка без кэша (--no-cache)

```bash
./build.sh --no-cache
```

**Когда это нужно:**

| Проблема | Решение |
|----------|---------|
| `pip install` выдаёт ошибки про старые версии | Кэш pip мог испортиться |
| Docker выдаёт `no matching manifest` | Кэш образа повреждён |
| После изменения `requirements/*.txt` образ не обновляется | Docker не видит изменений в файлах |
| Странные ошибки импорта в Python | Кэш содержит битые слои |
| Хотите гарантированно чистую сборку | На проде перед релизом |

`--no-cache` заставляет Docker пересобрать всё с нуля, игнорируя сохранённые слои. **Минус:** сборка дольше в 2-3 раза.

---

### 4. Запуск сервисов

#### Запустить все модели

```bash
docker-compose up -d
```

#### Запустить только конкретную модель

```bash
docker-compose up -d qwen25-vl
```

Порты:
- LFM2.5-VL → 8249
- Qwen2.5-VL → 8349
- GLM-OCR → 8399
- LightOnOCR → 8499

#### Проверка статуса

```bash
docker-compose ps
```

Вывод должен показывать `Up` для всех запущенных сервисов.

#### Проверка health endpoint

```bash
curl http://localhost:8349/health
```

Ответ при готовности:
```json
{"status":"healthy","backend":"transformers","model":"qwen25-vl","tasks_processed":0,"queue_size":0}
```

Если модель ещё загружается (первые минуты):
```json
{"status":"loading","backend":"transformers","model":"qwen25-vl","message":"Model is still loading"}
```

---

### 5. Типичный рабочий процесс

#### Первый запуск
```bash
chmod +x build.sh
./build.sh                # 10-30 минут, зависит от скорости интернета
docker-compose up -d
docker-compose logs -f qwen25-vl   # смотреть логи, ждать загрузки модели
```

#### После изменения кода модели (например, qwen_strategy.py)
```bash
./build.sh qwen           # пересобрать только Qwen (1-2 минуты)
docker-compose up -d --force-recreate qwen25-vl   # перезапустить с новым образом
```

#### После изменения общего кода (shared/config.py)
```bash
./build.sh base           # пересобрать базовый слой
./build.sh                # пересобрать все модели
docker-compose up -d --force-recreate
```

#### Проблемы с зависимостями (ошибки pip)
```bash
./build.sh --no-cache qwen   # пересобрать Qwen без кэша
```

#### Очистка и пересборка всего с нуля
```bash
docker-compose down
docker system prune -a    # ОСТОРОЖНО: удалит все неиспользуемые образы
./build.sh --no-cache
docker-compose up -d
```

---

### 6. Просмотр логов при проблемах

```bash
# Логи конкретного сервиса
docker-compose logs -f qwen25-vl

# Только ошибки
docker-compose logs qwen25-vl 2>&1 | grep -i error

# Последние 100 строк
docker-compose logs --tail=100 qwen25-vl
```

---

### 7. Остановка

```bash
# Остановить всё
docker-compose down

# Остановить конкретный сервис
docker-compose stop qwen25-vl
```

### 8. Полная очистка Docker

#### 1. Остановить и удалить контейнеры

```bash
# Остановить все запущенные контейнеры
docker-compose down

# Остановить и удалить volumes (модели, логи - всё!)
docker-compose down -v
```

#### 2. Удалить образы Vision Service

```bash
# Удалить все образы vision-*
docker rmi vision-base:latest vision-lfm25:latest vision-qwen:latest vision-glm:latest vision-lighton:latest

# Или одной командой
docker images | grep "vision-" | awk '{print $3}' | xargs docker rmi -f
```

#### 3. Полная очистка Docker (осторожно!)

```bash
# Удалить все неиспользуемые образы, контейнеры, сети
docker system prune -a

# Добавить удаление volumes (ВНИМАНИЕ: удалит всё!)
docker system prune -a --volumes
```

#### 4. Очистка кэша моделей (если нужно)

```bash
# Удалить локально скачанные модели HuggingFace
rm -rf ./models/*

# Или если использовали HF кэш по умолчанию
rm -rf ~/.cache/huggingface/
```

#### 5. Полный сброс (всё перечисленное вместе)

```bash
# Скрипт для полной очистки
docker-compose down -v
docker images | grep "vision-" | awk '{print $3}' | xargs docker rmi -f
docker system prune -f
rm -rf ./models/*
```

После этого система чиста, можно начинать сборку заново.

---

### Краткая шпаргалка

| Что нужно | Команда |
|-----------|---------|
| Первая сборка | `./build.sh && docker-compose up -d` |
| Пересобрать всё | `./build.sh && docker-compose up -d --force-recreate` |
| Пересобрать одну модель | `./build.sh qwen && docker-compose up -d --force-recreate qwen25-vl` |
| Сборка без кэша | `./build.sh --no-cache` |
| Посмотреть логи | `docker-compose logs -f qwen25-vl` |