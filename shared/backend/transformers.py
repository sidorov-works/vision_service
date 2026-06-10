# shared/backend/transformers.py

"""
Transformers бэкенд — загрузка модели через HuggingFace, инференс через стратегию.
"""

import logging
from typing import Dict, Any

from shared.backend.base import VisionBackend
from shared.config import config
from shared.prepare_image import prepare_image

logger = logging.getLogger(__name__)


class TransformersBackend(VisionBackend):
    """Бэкенд на основе transformers."""

    def __init__(self):
        self.strategy = None
        self.generation_params: Dict[str, Any] = {}
        self._healthy = False

    # async def initialize(self):
    #     """
    #     Загрузка модели через стратегию.
    #     Параметры генерации берутся из ModelConfig.generation_params
    #     и передаются в стратегию.
    #     """
    #     from shared.model_configs import MODEL_REGISTRY

    #     model_config = MODEL_REGISTRY[config.MODEL]
    #     self.strategy = model_config.create_transformers_strategy()
    #     self.generation_params = model_config.generation_params.copy()

    #     await self.strategy.load_model(
    #         model_id=model_config.transformers_id,
    #         cache_dir=config.MODELS_ROOT,  # ← папка для кэширования моделей (для прода)
    #         device=config.DEVICE,
    #     )

    #     self._healthy = True
    #     logger.info(f"Transformers бэкенд готов. Модель: {config.MODEL}, "
    #                f"устройство: {config.DEVICE}, параметры: {self.generation_params}")

    async def initialize(self):
        logger.info("=== TransformersBackend.initialize() START ===")
        from shared.model_configs import MODEL_REGISTRY

        logger.info(f"Loading model config for MODEL={config.MODEL}")
        model_config = MODEL_REGISTRY[config.MODEL]
        logger.info(f"Model config loaded: {model_config.transformers_id}")
        
        logger.info("Creating strategy...")
        self.strategy = model_config.create_transformers_strategy()
        self.generation_params = model_config.generation_params.copy()

        logger.info(f"Loading model from {model_config.transformers_id}...")
        await self.strategy.load_model(
            model_id=model_config.transformers_id,
            cache_dir=config.MODELS_ROOT,
            device=config.DEVICE,
        )

        self._healthy = True
        logger.info(f"Transformers бэкенд готов. Модель: {config.MODEL}, устройство: {config.DEVICE}")

    async def process(self, image_bytes: bytes, prompt: str) -> str:
        """
        Распознавание/описание изображения.
        
        Параметры генерации передаются в стратегию.
        """
        resized_bytes = prepare_image(image_bytes, config.MAX_IMAGE_DIMENSION)
        
        # Передаём параметры генерации в стратегию
        response = await self.strategy.infer(
            resized_bytes, 
            prompt,
            generation_params=self.generation_params
        )

        # Некоторые модели возвращают промпт в начале ответа — убираем
        if response and prompt and response.startswith(prompt):
            response = response[len(prompt):].lstrip()

        return response or ""

    def is_healthy(self) -> bool:
        return self._healthy

    async def close(self):
        """Освобождение ресурсов."""
        if self.strategy:
            self.strategy.cleanup()
        self._healthy = False