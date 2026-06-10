"""
Конфигурация сервиса из переменных окружения.
"""

import os
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv

load_dotenv(override=True)


class Config:
    """Конфигурация сервиса."""

    # ----------------------------------------------------------------------
    # Безопасность
    # ----------------------------------------------------------------------
    INTERNAL_API_SECRET = os.getenv("INTERNAL_API_SECRET")
    REQUIRE_AUTH = os.getenv("REQUIRE_AUTH", "true").lower() == "true"

    # ----------------------------------------------------------------------
    # Выбор бэкенда и модели
    # ----------------------------------------------------------------------
    BACKEND = os.getenv("BACKEND", "transformers")  # transformers, mlx
    MODEL = os.getenv("MODEL")  # ключ из MODEL_REGISTRY

    # ----------------------------------------------------------------------
    # Transformers бэкенд
    # ----------------------------------------------------------------------
    DEVICE = os.getenv("DEVICE", "cuda" if os.getenv("CUDA_AVAILABLE") else "cpu")
    USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "false").lower() == "true"
    MODELS_ROOT = Path(os.getenv("MODELS_ROOT", "/app/models"))

    # ----------------------------------------------------------------------
    # MLX бэкенд
    # ----------------------------------------------------------------------
    @property
    def MLX_MODEL_ID(self) -> str:
        """ID модели в MLX формате."""
        from shared.model_configs import MODEL_REGISTRY
        return MODEL_REGISTRY[self.MODEL].mlx_id

    # ----------------------------------------------------------------------
    # Общие ограничения
    # ----------------------------------------------------------------------
    MAX_BATCH_SIZE = int(os.getenv("MAX_BATCH_SIZE", "20"))
    MAX_TOTAL_BATCH_MB = int(os.getenv("MAX_TOTAL_BATCH_MB", "50"))
    BASE64_OVERHEAD = 1.33

    MAX_IMAGE_SIZE_MB = int(os.getenv("MAX_IMAGE_SIZE_MB", "10"))
    MAX_IMAGE_DIMENSION = int(os.getenv("MAX_IMAGE_DIMENSION", "2000"))

    ALLOWED_MIME_TYPES = {
        "image/jpeg", "image/jpg", "image/png", "image/tiff",
        "image/bmp", "image/webp", "image/heic"
    }

    INPUT_QUEUE_MAXSIZE = int(os.getenv("INPUT_QUEUE_MAXSIZE", "100"))
    PROCESS_TIMEOUT = float(os.getenv("PROCESS_TIMEOUT", "60.0"))

    # Максимальное время ожидания инициализации
    WAIT_FOR_BACKEND_TIMEOUT = float(os.getenv("WAIT_FOR_BACKEND_TIMEOUT", "60.0"))

    # ----------------------------------------------------------------------
    # Rate limiting
    # ----------------------------------------------------------------------
    RATE_LIMIT_OCR = os.getenv("RATE_LIMIT_OCR", "30/minute")
    RATE_LIMIT_HEALTH = os.getenv("RATE_LIMIT_HEALTH", "100/minute")

    # ----------------------------------------------------------------------
    # Логирование
    # ----------------------------------------------------------------------
    LOG_PATH = Path(os.getenv("LOG_PATH", "logs"))
    LOGGING_LEVEL = os.getenv("LOGGING_LEVEL", "INFO")
    DOCKER_ENV = os.getenv("DOCKER_ENV", "false").lower() == "true"


config = Config()