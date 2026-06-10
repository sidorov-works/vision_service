# shared/prepare_image.py

"""
Подготовка изображения: ресайз и конвертация в PNG.
"""

import io
from PIL import Image
import logging

logger = logging.getLogger(__name__)


def prepare_image(image_bytes: bytes, max_dimension: int) -> bytes:
    """
    Ресайзит изображение до max_dimension по длинной стороне,
    конвертирует в RGB и сохраняет как PNG.

    Args:
        image_bytes: исходное изображение в байтах
        max_dimension: максимальная ширина или высота

    Returns:
        PNG изображение в байтах
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    if image.width > max_dimension or image.height > max_dimension:
        ratio = min(max_dimension / image.width, max_dimension / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        logger.debug(f"Resized: {image.width}x{image.height} -> {new_size[0]}x{new_size[1]}")

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()