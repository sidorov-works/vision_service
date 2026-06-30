# shared/model_configs.py

"""
Реестр моделей: для каждой модели указаны ID в разных бэкендах,
ключ стратегии инференса и параметры генерации.
"""

from typing import Dict, Any, Optional


class ModelConfig:
    """Конфигурация одной модели."""

    def __init__(
        self,
        transformers_id: str,                      # ID на HuggingFace для Transformers
        mlx_id: str,                               # ID для MLX (из mlx-community)
        strategy_key: str,                         # ключ в STRATEGY_REGISTRY
        default_prompt: Optional[str] = None,      # дефолтный промпт для модели
        generation_params: Optional[Dict[str, Any]] = None,  # параметры генерации (max_tokens, temp, и т.д.)
        trust_remote_code: bool = False
    ):
        self.transformers_id = transformers_id
        self.mlx_id = mlx_id
        self.strategy_key = strategy_key
        self.default_prompt = default_prompt
        self.generation_params = generation_params or {}
        self.trust_remote_code = trust_remote_code

    def create_transformers_strategy(self):
        """Создаёт стратегию для Transformers бэкенда."""
        from shared.strategies.transformers_strategies import STRATEGY_REGISTRY
        strategy_class = STRATEGY_REGISTRY.get(self.strategy_key)
        if not strategy_class:
            raise ValueError(f"Strategy '{self.strategy_key}' not found")
        return strategy_class()

    def create_mlx_strategy(self):
        """Создаёт стратегию для MLX бэкенда."""
        from shared.strategies.mlx_strategies import MLX_STRATEGY_REGISTRY
        strategy_class = MLX_STRATEGY_REGISTRY.get(self.strategy_key)
        if not strategy_class:
            raise ValueError(f"MLX strategy '{self.strategy_key}' not found")
        return strategy_class()


# Реестр моделей
MODEL_REGISTRY: Dict[str, ModelConfig] = {
    # ======================================================================
    # OCR-специализированные 
    # ======================================================================
    "paddle-ocr": ModelConfig(
        transformers_id="PaddlePaddle/PaddleOCR-VL",
        mlx_id="mlx-community/PaddleOCR-VL-4bit",
        strategy_key="paddle-ocr",
        default_prompt="OCR:",
        generation_params={
            "max_new_tokens": 1024,
            "do_sample": False,
            "temperature": 0.1,
        }
    ),
    "glm-ocr": ModelConfig(
        transformers_id="zai-org/GLM-OCR",
        mlx_id="mlx-community/GLM-OCR-bf16",
        strategy_key="glm-ocr",
        default_prompt="Text Recognition:",
        generation_params={
            "max_new_tokens": 8192,
            "do_sample": False,
            "temperature": 0.1,
        }
    ),
    "lighton": ModelConfig(
        transformers_id="lightonai/LightOnOCR-2-1B", # в проде требует transformers>=5.0.0
        mlx_id="mlx-community/LightOnOCR-2-1B-8bit",
        strategy_key="lighton",
        default_prompt="Extract text from this image.",
        generation_params={
            "max_new_tokens": 4096,
            "do_sample": False,
            "temperature": 0.1,
        }
    ),

    # ======================================================================
    # VLM-модели (будут использоваться для описания изображений)
    # ======================================================================
    "smolvlm": ModelConfig(
        transformers_id="HuggingFaceTB/SmolVLM-256M-Instruct",
        mlx_id="mlx-community/SmolVLM-256M-Instruct-6bit",
        strategy_key="smolvlm",
        default_prompt="Describe this image in detail.",
        generation_params={
            "max_new_tokens": 512,
            "do_sample": False,
            "temperature": 0.1,
        }
    ),
    "qwen25-vl": ModelConfig(
        transformers_id="Qwen/Qwen2.5-VL-3B-Instruct",
        mlx_id=None,  # не используем (qwen плохо работает на mlx)
        strategy_key="qwen25-vl",
        default_prompt="Describe this image in detail.",
        generation_params={
            "max_new_tokens": 1024,
            "do_sample": False,
            "temperature": 0.1,
        }
    ),
    "lfm25-vl": ModelConfig(
        transformers_id="LiquidAI/LFM2.5-VL-450M",
        mlx_id="LiquidAI/LFM2.5-VL-450M-MLX-8bit",
        strategy_key="lfm25-vl",
        default_prompt="Describe this image in detail.",
        generation_params={
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.1,
            "min_p": 0.15,
            "repetition_penalty": 1.05,
        },
        trust_remote_code=True 
    )
}