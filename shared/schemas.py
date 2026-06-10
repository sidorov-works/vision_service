"""
Pydantic модели для Vision Service (OCR + описание).
"""

import re
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from shared.config import config


# ======================================================================
# Базовые классы с общей валидацией
# ======================================================================

class BaseImageRequest(BaseModel):
    """
    Запрос с одним изображением.
    """
    image_base64: str = Field(..., description="Base64-строка изображения")

    @field_validator('image_base64')
    @classmethod
    def validate_image(cls, v: str) -> str:
        """Проверка: не пусто, размер не превышает MAX_IMAGE_SIZE_MB."""
        if not v:
            raise ValueError('image_base64 cannot be empty')

        # Извлекаем чистый base64 (отрезаем data:image/png;base64, если есть)
        mime_match = re.match(r"data:([\w/]+);base64,", v)
        clean_b64 = v[mime_match.end():] if mime_match else v

        max_bytes = config.MAX_IMAGE_SIZE_MB * 1024 * 1024
        max_base64 = int(max_bytes * config.BASE64_OVERHEAD)

        if len(clean_b64) > max_base64:
            estimated_mb = len(clean_b64) * 3 // 4 // (1024 * 1024)
            raise ValueError(
                f'Image size ~{estimated_mb} MB exceeds maximum {config.MAX_IMAGE_SIZE_MB} MB'
            )
        return v


class BaseBatchRequest(BaseModel):
    """
    Батчевый запрос (список изображений).
    """
    images: List[str] = Field(..., description="Список base64-строк изображений")
    prompts: Optional[List[str]] = Field(None, description="Промпты для каждого изображения")
    common_prompt: Optional[str] = Field(None, description="Промпт по умолчанию")

    @field_validator('images')
    @classmethod
    def validate_images_batch(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError('Images list cannot be empty')

        if len(v) > config.MAX_BATCH_SIZE:
            raise ValueError(f'Batch size {len(v)} exceeds maximum {config.MAX_BATCH_SIZE}')

        # Проверка общего размера батча
        total_base64_len = sum(len(img) for img in v)
        max_total_bytes = config.MAX_TOTAL_BATCH_MB * 1024 * 1024
        max_total_base64 = int(max_total_bytes * config.BASE64_OVERHEAD)

        if total_base64_len > max_total_base64:
            raise ValueError(f'Total batch size exceeds {config.MAX_TOTAL_BATCH_MB} MB')

        # Проверка каждого изображения
        max_bytes = config.MAX_IMAGE_SIZE_MB * 1024 * 1024
        max_base64 = int(max_bytes * config.BASE64_OVERHEAD)

        for i, img_b64 in enumerate(v):
            mime_match = re.match(r"data:([\w/]+);base64,", img_b64)
            clean_b64 = img_b64[mime_match.end():] if mime_match else img_b64
            if len(clean_b64) > max_base64:
                estimated_mb = len(clean_b64) * 3 // 4 // (1024 * 1024)
                raise ValueError(f'Image {i} size ~{estimated_mb} MB exceeds max')

        return v

    @field_validator('prompts')
    @classmethod
    def validate_prompts(cls, v: Optional[List[str]], info) -> Optional[List[str]]:
        if v is None:
            return v

        images = info.data.get('images', [])
        if len(v) != len(images):
            raise ValueError(f'Prompts count ({len(v)}) != images count ({len(images)})')

        for i, p in enumerate(v):
            if p is not None and len(p.strip()) == 0:
                raise ValueError(f'Prompt at index {i} cannot be empty')
        return v


# ======================================================================
# OCR эндпоинты
# ======================================================================

class OCRSingleRequest(BaseImageRequest):
    """OCR запрос одного изображения."""
    prompt: Optional[str] = Field(None, description="Промпт для OCR")

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v.strip()) == 0:
            raise ValueError('Prompt cannot be empty')
        return v


class OCRBatchRequest(BaseBatchRequest):
    """OCR батчевый запрос."""
    pass


class OCRResponse(BaseModel):
    """OCR ответ — только текст, без markdown."""
    success: bool
    task_id: str
    text: str = ""
    processing_time_ms: float = 0
    error: Optional[str] = None


class OCRBatchResponse(BaseModel):
    """OCR батчевый ответ."""
    success: bool
    task_id: str
    results: List[OCRResponse]
    processing_time_ms: float = 0
    error: Optional[str] = None


# ======================================================================
# Describe эндпоинты (описание изображений)
# ======================================================================

class DescribeSingleRequest(BaseImageRequest):
    """Describe запрос одного изображения."""
    prompt: Optional[str] = Field(None, description="Промпт для описания")

    @field_validator('prompt')
    @classmethod
    def validate_prompt(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and len(v.strip()) == 0:
            raise ValueError('Prompt cannot be empty')
        return v


class DescribeBatchRequest(BaseBatchRequest):
    """Describe батчевый запрос."""
    pass


class DescribeResponse(BaseModel):
    """Describe ответ."""
    success: bool
    task_id: str
    description: str = ""
    processing_time_ms: float = 0
    error: Optional[str] = None


class DescribeBatchResponse(BaseModel):
    """Describe батчевый ответ."""
    success: bool
    task_id: str
    results: List[DescribeResponse]
    processing_time_ms: float = 0
    error: Optional[str] = None


# ======================================================================
# Health check
# ======================================================================

class HealthResponse(BaseModel):
    """Ответ health check."""
    status: str
    backend: str
    model: str
    tasks_processed: int = 0
    queue_size: int = 0