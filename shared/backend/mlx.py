# shared/backend/mlx.py

"""
MLX бэкенд — для Apple Silicon.
"""

import asyncio
import io
import logging
from typing import Dict, Any
from PIL import Image

from mlx_vlm import generate, load
from mlx_vlm.utils import load_config

from shared.backend.base import VisionBackend
from shared.config import config
from shared.prepare_image import prepare_image

logger = logging.getLogger(__name__)


class MLXBackend(VisionBackend):
    """Бэкенд на основе MLX."""

    def __init__(self):
        self.model = None
        self.processor = None
        self.model_config = None
        self.strategy = None
        self.generation_params: Dict[str, Any] = {}
        self._healthy = False

    async def initialize(self):
        """
        Загрузка модели и получение параметров генерации из конфига.
        
        MLX использует стандартный кэш HuggingFace (не MODELS_ROOT),
        так как это бэкенд для разработки и прототипирования.
        """
        from shared.model_configs import MODEL_REGISTRY

        model_config = MODEL_REGISTRY[config.MODEL]
        self.strategy = model_config.create_mlx_strategy()
        self.generation_params = model_config.generation_params.copy()
        model_id = model_config.mlx_id

        if model_id is None:
            raise ValueError(f"Model '{config.MODEL}' has no mlx_id. Cannot use MLX backend.")

        logger.info(f"Загрузка MLX модели: {model_id}")

        def _load():
            # MLX качает в стандартный HF кэш (по умолчанию ~/.cache/huggingface)
            model, processor = load(model_id, trust_remote_code=True)
            cfg = load_config(model_id)
            return model, processor, cfg

        self.model, self.processor, self.model_config = await asyncio.to_thread(_load)
        self._healthy = True
        logger.info(f"MLX бэкенд готов. Модель: {model_id}, параметры: {self.generation_params}")

    async def process(self, image_bytes: bytes, prompt: str) -> str:
        """
        Распознавание/описание через MLX.
        
        Параметры генерации (max_tokens, temp, min_p, repetition_penalty и т.д.)
        берутся из ModelConfig.generation_params.
        """
        resized_bytes = prepare_image(image_bytes, config.MAX_IMAGE_DIMENSION)
        image = Image.open(io.BytesIO(resized_bytes)).convert("RGB")

        formatted_prompt = self.strategy.format_prompt(
            self.processor, self.model_config, prompt, image
        )

        # Подготовка параметров генерации
        # MLX generate ожидает параметры: max_tokens, temp, min_p, repetition_penalty, verbose
        gen_params = self.generation_params.copy()
        
        # Преобразуем max_new_tokens -> max_tokens (если нужно)
        # Transformers использует max_new_tokens, MLX — max_tokens
        if "max_new_tokens" in gen_params and "max_tokens" not in gen_params:
            gen_params["max_tokens"] = gen_params.pop("max_new_tokens")
        
        # Устанавливаем verbose=False по умолчанию (подавляем вывод прогресса)
        if "verbose" not in gen_params:
            gen_params["verbose"] = False
        
        # Убираем параметры, которые не поддерживаются MLX generate
        # do_sample в MLX не используется — вместо него temp=0 даёт детерминированный вывод
        unsupported = ["do_sample"]
        for key in unsupported:
            gen_params.pop(key, None)

        logger.debug(f"MLX generation params: {gen_params}")

        def _infer():
            output = generate(
                self.model,
                self.processor,
                prompt=formatted_prompt,
                image=image,
                **gen_params
            )
            return output.text.strip()

        return await asyncio.to_thread(_infer)

    def is_healthy(self) -> bool:
        return self._healthy