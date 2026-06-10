# shared/backend/base.py

"""
Абстрактный базовый класс для всех бэкендов.
"""

from abc import ABC, abstractmethod


class VisionBackend(ABC):
    """
    Бэкенд для работы с vision-моделями (OCR + описание).
    Единый метод process принимает изображение и промпт,
    возвращает текст.
    """

    @abstractmethod
    async def initialize(self):
        """Загрузка модели."""
        pass

    @abstractmethod
    async def process(self, image_bytes: bytes, prompt: str) -> str:
        """
        Универсальный метод.

        Args:
            image_bytes: изображение в байтах (PNG/JPEG)
            prompt: текстовый промпт ("OCR:", "Describe...", и т.д.)

        Returns:
            Текст ответа модели
        """
        pass

    @abstractmethod
    def is_healthy(self) -> bool:
        """Готов ли бэкенд к работе."""
        pass

    async def close(self):
        """Освобождение ресурсов (опционально)."""
        pass