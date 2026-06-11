#!/bin/bash # запуск через Bash

# ===================================================================
# build.sh - Скрипт сборки Docker образов для Vision Service
# ===================================================================
#
# Использование:
#   ./build.sh              # Собрать все образы
#   ./build.sh --no-cache   # Собрать все образы без кэша
#   ./build.sh lfm25        # Собрать только указанную модель
#
# Доступные модели: base, lfm25, qwen, glm, lighton
# ===================================================================

set -e  # Останавливаем скрипт при любой ошибке

# Цветной вывод для лучшей читаемости
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Параметры сборки
NO_CACHE=""
BUILD_SPECIFIC=""

# Разбор аргументов командной строки
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        base|lfm25|qwen|glm|lighton)
            BUILD_SPECIFIC="$1"
            shift
            ;;
        *)
            echo -e "${RED}Неизвестный аргумент: $1${NC}"
            echo "Использование: ./build.sh [--no-cache] [base|lfm25|qwen|glm|lighton]"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Vision Service Docker Images Builder  ${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Проверяем наличие Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Ошибка: Docker не установлен${NC}"
    exit 1
fi

# Проверяем наличие .env файла
if [ ! -f .env ]; then
    echo -e "${YELLOW}Предупреждение: Файл .env не найден${NC}"
    echo "Создайте файл .env с необходимыми переменными (HF_TOKEN и др.)"
    echo "Можно скопировать из .env.example"
    echo ""
    read -p "Продолжить без .env? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Проверяем наличие NVIDIA Docker runtime (если нужно)
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA CUDA обнаружена${NC}"
else
    echo -e "${YELLOW}⚠ NVIDIA CUDA не обнаружена (сборка продолжится, но GPU может не работать)${NC}"
fi
echo ""

# Функция для сборки образа
build_image() {
    local dockerfile="dockerfiles/$1"
    local image_name=$2
    local description=$3
    
    echo -e "${GREEN}▶ Сборка: ${description}${NC}"
    echo "  Файл: $dockerfile"
    echo "  Тег: $image_name"
    
    if [ -n "$NO_CACHE" ]; then
        echo "  Режим: без кэша"
    fi
    
    docker build $NO_CACHE -f "$dockerfile" -t "$image_name" .
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}  ✓ Готово${NC}"
    else
        echo -e "${RED}  ✗ Ошибка сборки $description${NC}"
        exit 1
    fi
    echo ""
}

# ===================================================================
# Сборка образов
# ===================================================================

# Базовый образ (общий для всех моделей)
if [ -z "$BUILD_SPECIFIC" ] || [ "$BUILD_SPECIFIC" == "base" ]; then
    build_image "Dockerfile.base" "vision-base:latest" "Базовый образ (vision-base)"
fi

# LFM2.5-VL
if [ -z "$BUILD_SPECIFIC" ] || [ "$BUILD_SPECIFIC" == "lfm25" ]; then
    build_image "Dockerfile.lfm25" "vision-lfm25:latest" "LFM2.5-VL (450M параметров)"
fi

# Qwen2.5-VL
if [ -z "$BUILD_SPECIFIC" ] || [ "$BUILD_SPECIFIC" == "qwen" ]; then
    build_image "Dockerfile.qwen" "vision-qwen:latest" "Qwen2.5-VL (3B параметров)"
fi

# GLM-OCR
if [ -z "$BUILD_SPECIFIC" ] || [ "$BUILD_SPECIFIC" == "glm" ]; then
    build_image "Dockerfile.glm" "vision-glm:latest" "GLM-OCR (специализированная OCR)"
fi

# LightOnOCR
if [ -z "$BUILD_SPECIFIC" ] || [ "$BUILD_SPECIFIC" == "lighton" ]; then
    build_image "Dockerfile.lighton" "vision-lighton:latest" "LightOnOCR-2-1B (специализированная OCR)"
fi

# ===================================================================
# Вывод результатов
# ===================================================================

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✓ Все образы успешно собраны!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Список собранных образов:"
docker images | grep "vision-" | head -10
echo ""
echo "Запуск сервисов:"
echo "  docker-compose up -d"
echo ""
echo "Проверка статуса:"
echo "  docker-compose ps"
echo ""
echo "Просмотр логов:"
echo "  docker-compose logs -f"
echo ""
echo "Остановка сервисов:"
echo "  docker-compose down"