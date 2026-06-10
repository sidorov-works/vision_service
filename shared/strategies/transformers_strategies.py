# shared/strategies/transformers_strategies.py

"""
Стратегии инференса для разных моделей в Transformers бэкенде.
Каждая стратегия знает, как загрузить конкретную модель и выполнить инференс.
Параметры генерации (max_new_tokens, temperature и т.д.) передаются из бэкенда.
"""

import asyncio
import io
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from PIL import Image

logger = logging.getLogger(__name__)


class InferenceStrategy:
    """
    Абстрактная стратегия инференса для Transformers бэкенда.
    
    Каждая конкретная стратегия знает:
    1. Как загрузить свою модель и процессор
    2. Как выполнить инференс (распознавание текста или описание)
    3. Как освободить ресурсы
    """

    async def load_model(self, model_id: str, cache_dir: Path, device: str):
        """Загружает модель и процессор. Должен быть переопределён в наследниках."""
        raise NotImplementedError

    async def infer(self, image_bytes: bytes, prompt: str, generation_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Выполняет инференс.
        
        Args:
            image_bytes: изображение в байтах (уже отресайзенное)
            prompt: текстовый промпт
            generation_params: параметры генерации (max_new_tokens, temperature, do_sample, и т.д.)
        
        Returns:
            Текст ответа модели
        """
        raise NotImplementedError

    def cleanup(self):
        """Освобождает ресурсы GPU. Переопределяется при необходимости."""
        pass


# ======================================================================
# PaddleOCR-VL
# ======================================================================

class PaddleOCRStrategy(InferenceStrategy):
    """
    Стратегия инференса для PaddleOCR-VL (PaddlePaddle/PaddleOCR-VL).
    
    Особенности:
    - Использует AutoModelForCausalLM + AutoProcessor
    - Промпты зависят от задачи: "OCR:", "Table Recognition:", "Formula Recognition:", "Chart Recognition:"
    - Поддерживает flash-attention для ускорения
    """

    def __init__(self):
        self.model = None
        self.processor = None

    async def load_model(self, model_id: str, cache_dir: Path, device: str):
        from transformers import AutoModelForCausalLM, AutoProcessor

        logger.info(f"Загрузка PaddleOCR-VL: {model_id} на {device}")

        def _load():
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=str(cache_dir)
            )
            
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                cache_dir=str(cache_dir)
            )
            model.eval()
            return processor, model

        self.processor, self.model = await asyncio.to_thread(_load)
        logger.info(f"PaddleOCR-VL загружена")

    async def infer(self, image_bytes: bytes, prompt: str, generation_params: Optional[Dict[str, Any]] = None) -> str:
        """Распознавание текста через PaddleOCR-VL."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Параметры генерации по умолчанию
        params = generation_params or {}
        max_new_tokens = params.get("max_new_tokens", 1024)
        do_sample = params.get("do_sample", False)
        temperature = params.get("temperature", 0.1)

        def _infer():
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
            }]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature
                )

            result = self.processor.batch_decode(
                output_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )[0]
            return result.strip()

        return await asyncio.to_thread(_infer)


# ======================================================================
# SmolVLM
# ======================================================================

class SmolVLMStrategy(InferenceStrategy):
    """
    Стратегия инференса для SmolVLM-256M-Instruct.
    
    Особенности:
    - Лёгкая VLM-модель от HuggingFace
    - Хорошо работает на CPU и Mac
    - Поддерживает любые промпты (описание, вопросы, OCR)
    """

    def __init__(self):
        self.model = None
        self.processor = None

    async def load_model(self, model_id: str, cache_dir: Path, device: str):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        logger.info(f"Загрузка SmolVLM: {model_id} на {device}")

        def _load():
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=str(cache_dir)
            )
            
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            
            model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                trust_remote_code=True,
                dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                cache_dir=str(cache_dir)
            )
            model.eval()
            return processor, model

        self.processor, self.model = await asyncio.to_thread(_load)
        logger.info(f"SmolVLM загружена")

    async def infer(self, image_bytes: bytes, prompt: str, generation_params: Optional[Dict[str, Any]] = None) -> str:
        """Универсальный инференс для любых промптов."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        params = generation_params or {}
        max_new_tokens = params.get("max_new_tokens", 512)
        do_sample = params.get("do_sample", False)
        temperature = params.get("temperature", 0.1)

        def _infer():
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
            }]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature
                )

            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()

        return await asyncio.to_thread(_infer)

    def cleanup(self):
        """Освобождение GPU памяти."""
        if self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.processor = None


# ======================================================================
# Qwen2.5-VL
# ======================================================================

class Qwen25VLStrategy(InferenceStrategy):
    """
    Стратегия инференса для Qwen2.5-VL-3B-Instruct.
    
    Особенности:
    - Мощная мультиязычная VLM-модель
    - Поддерживает русский язык
    - Требует много VRAM (~8GB в fp16)
    - Использует специализированный Qwen2VLForConditionalGeneration
    """

    def __init__(self):
        self.model = None
        self.processor = None

    async def load_model(self, model_id: str, cache_dir: Path, device: str):
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

        logger.info(f"Загрузка Qwen2.5-VL: {model_id} на {device}")

        def _load():
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=str(cache_dir)
            )
            
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                cache_dir=str(cache_dir)
            )
            model.eval()
            return processor, model

        self.processor, self.model = await asyncio.to_thread(_load)
        logger.info(f"Qwen2.5-VL загружена")

    async def infer(self, image_bytes: bytes, prompt: str, generation_params: Optional[Dict[str, Any]] = None) -> str:
        """Универсальный инференс для любых промптов."""
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        params = generation_params or {}
        max_new_tokens = params.get("max_new_tokens", 1024)
        do_sample = params.get("do_sample", False)
        temperature = params.get("temperature", 0.1)

        def _infer():
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
            }]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature
                )

            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()

        return await asyncio.to_thread(_infer)

    def cleanup(self):
        """Освобождение GPU памяти."""
        if self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.processor = None


# ======================================================================
# LFM2.5-VL-450M
# ======================================================================

class LFM25VLStrategy(InferenceStrategy):
    """
    Стратегия инференса для LFM2.5-VL-450M от Liquid AI.
    
    Особенности:
    - Лёгкая VLM-модель (450M параметров)
    - Поддерживает мультиязычность (включая русский)
    - Работает на обоих бэкендах (Transformers и MLX)
    - Параметры генерации задаются в ModelConfig.generation_params
    """

    def __init__(self):
        self.model = None
        self.processor = None

    async def load_model(self, model_id: str, cache_dir: Path, device: str):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        logger.info(f"Загрузка LFM2.5-VL: {model_id} на {device}")

        def _load():
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=str(cache_dir)
            )
            
            torch_dtype = torch.float16 if device == "cuda" else torch.float32
            
            model = AutoModelForVision2Seq.from_pretrained(
                model_id,
                trust_remote_code=True,
                dtype=torch_dtype,
                device_map="auto" if device == "cuda" else None,
                cache_dir=str(cache_dir)
            )
            model.eval()
            return processor, model

        self.processor, self.model = await asyncio.to_thread(_load)
        logger.info(f"LFM2.5-VL загружена")

    async def infer(self, image_bytes: bytes, prompt: str, generation_params: Optional[Dict[str, Any]] = None) -> str:
        """
        Универсальный инференс для LFM2.5-VL.
        
        Параметры генерации (max_new_tokens, temperature, min_p, repetition_penalty и т.д.)
        полностью берутся из generation_params, переданных из бэкенда.
        """
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Параметры по умолчанию — только на случай, если generation_params не передан
        params = {
            "max_new_tokens": 512,
            "do_sample": True,
            "temperature": 0.1,
            "min_p": 0.15,
            "repetition_penalty": 1.05,
        }
        if generation_params:
            params.update(generation_params)

        def _infer():
            messages = [{
                "role": "user",
                "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
            }]
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True,
                tokenize=True, return_dict=True, return_tensors="pt"
            ).to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=params["max_new_tokens"],
                    do_sample=params.get("do_sample", False),
                    temperature=params.get("temperature", 0.1),
                    min_p=params.get("min_p"),
                    repetition_penalty=params.get("repetition_penalty", 1.0),
                )

            response = self.processor.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            return response.strip()

        return await asyncio.to_thread(_infer)

    def cleanup(self):
        """Освобождение GPU памяти."""
        if self.model:
            del self.model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.model = None
            self.processor = None


# ======================================================================
# Реестр стратегий
# ======================================================================

STRATEGY_REGISTRY = {
    "paddle-ocr": PaddleOCRStrategy,
    "smolvlm": SmolVLMStrategy,
    "qwen25-vl": Qwen25VLStrategy,
    "lfm25-vl": LFM25VLStrategy,
}