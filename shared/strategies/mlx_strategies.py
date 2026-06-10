# shared/strategies/mlx_strategies.py

"""
Стратегии инференса для MLX бэкенда.

Каждая стратегия определяет, как форматировать промпт для конкретной модели.
Параметры генерации (max_tokens, temp и т.д.) задаются в ModelConfig.generation_params
и передаются напрямую в mlx_vlm.generate() из бэкенда.
"""

from abc import ABC, abstractmethod
from typing import Any
from PIL import Image
from mlx_vlm.prompt_utils import apply_chat_template


class MLXInferenceStrategy(ABC):
    """
    Абстрактная стратегия для MLX.
    
    Единственная обязанность — правильно отформатировать промпт
    для конкретной модели через apply_chat_template.
    """

    @abstractmethod
    def format_prompt(self, processor: Any, config: Any, prompt: str, image: Image.Image) -> str:
        """
        Форматирует промпт для конкретной модели.
        
        Args:
            processor: процессор модели (содержит tokenizer)
            config: конфиг модели (из mlx_vlm.utils.load_config)
            prompt: исходный текстовый промпт
            image: изображение (PIL)
        
        Returns:
            Отформатированная строка промпта, готовая для передачи в generate()
        """
        pass


class SmolVLMMLXStrategy(MLXInferenceStrategy):
    """Стратегия для SmolVLM на MLX."""
    
    def format_prompt(self, processor: Any, config: Any, prompt: str, image: Image.Image) -> str:
        return apply_chat_template(
            processor, 
            config, 
            prompt, 
            add_generation_prompt=True, 
            num_images=1
        )


class PaddleOCRMLXStrategy(MLXInferenceStrategy):
    """Стратегия для PaddleOCR-VL на MLX."""
    
    def format_prompt(self, processor: Any, config: Any, prompt: str, image: Image.Image) -> str:
        # PaddleOCR-VL использует короткий промпт "OCR:" по умолчанию
        if not prompt:
            prompt = "OCR:"
        return apply_chat_template(
            processor, 
            config, 
            prompt, 
            add_generation_prompt=True, 
            num_images=1
        )


class GLMOCRMLXStrategy(MLXInferenceStrategy):
    """Стратегия для GLM-OCR на MLX."""
    
    def format_prompt(self, processor: Any, config: Any, prompt: str, image: Image.Image) -> str:
        return apply_chat_template(
            processor, 
            config, 
            prompt, 
            add_generation_prompt=True, 
            num_images=1
        )


class LightOnOCRMLXStrategy(MLXInferenceStrategy):
    """Стратегия для LightOnOCR на MLX."""
    
    def format_prompt(self, processor: Any, config: Any, prompt: str, image: Image.Image) -> str:
        return apply_chat_template(
            processor, 
            config, 
            prompt, 
            add_generation_prompt=True, 
            num_images=1
        )


class LFM25VLMLXStrategy(MLXInferenceStrategy):
    """
    Стратегия для LFM2.5-VL-450M на MLX.
    
    Модель требует обязательного применения chat template.
    Параметры генерации задаются в ModelConfig.generation_params.
    """
    
    def format_prompt(self, processor: Any, config: Any, prompt: str, image: Image.Image) -> str:
        return apply_chat_template(
            processor, 
            config, 
            prompt, 
            add_generation_prompt=True, 
            num_images=1
        )


# Реестр стратегий для MLX бэкенда
# Ключ должен совпадать со значением strategy_key в ModelConfig
MLX_STRATEGY_REGISTRY = {
    "paddle-ocr": PaddleOCRMLXStrategy,
    "glm-ocr": GLMOCRMLXStrategy,
    "lighton": LightOnOCRMLXStrategy,
    "smolvlm": SmolVLMMLXStrategy,
    "lfm25-vl": LFM25VLMLXStrategy,
}