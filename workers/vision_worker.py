# workers/vision_worker.py

"""
Воркер — единственный владелец модели.
Забирает задачи из очереди, вызывает бэкенд, возвращает результат.
"""

import asyncio
import time
import logging
from dataclasses import dataclass
from typing import Optional

from shared.config import config
from shared.backend.base import VisionBackend

logger = logging.getLogger(__name__)


@dataclass
class VisionTask:
    """Задача на обработку."""
    task_id: str
    image_bytes: bytes
    prompt: str
    created_at: float


@dataclass
class VisionResult:
    """Результат обработки."""
    task_id: str
    success: bool
    text: str = ""
    error: str = None
    processing_time_ms: float = 0


class VisionWorker:
    """Воркер, обрабатывающий задачи последовательно."""

    def __init__(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = True
        self.backend: Optional[VisionBackend] = None
        self.tasks_processed = 0
        self._initialized = asyncio.Event()
        
        # Флаг, указывающий на то, что инициализация бэкенда окончательно провалилась.
        # Нужен для того, чтобы воркер не ждал вечно self.backend, если при загрузке
        # модели произошла ошибка (например, нет прав на файл, не хватило памяти,
        # модель несовместима с бэкендом, истёк таймаут скачивания и т.д.).
        # Без этого флага при провале инициализации воркер продолжает работать,
        # но self.backend навсегда остаётся None, и все задачи будут висеть
        # до таймаута WAIT_FOR_BACKEND_TIMEOUT (60 секунд), после чего падать с ошибкой.
        # С этим флагом задачи сразу получают ошибку, не заставляя клиента ждать.
        self._init_failed = False

    async def initialize(self):
        """Создаёт и инициализирует бэкенд."""
        # Ленивые импорты — только для того бэкенда, который реально используется
        if config.BACKEND == "transformers":
            from shared.backend.transformers import TransformersBackend
            self.backend = TransformersBackend()
        elif config.BACKEND == "mlx":
            from shared.backend.mlx import MLXBackend
            self.backend = MLXBackend()
        else:
            raise ValueError(f"Unknown backend: {config.BACKEND}")

        await self.backend.initialize()
        logger.info(f"Worker инициализирован. Бэкенд: {config.BACKEND}")
        self._initialized.set()

    async def wait_until_ready(self):
        """Ожидает готовности воркера."""
        await self._initialized.wait()

    async def start(self):
        """Запускает воркер. Инициализация модели происходит в фоне."""
        # Запускаем инициализацию в фоне, не блокируем основной цикл
        asyncio.create_task(self._init_backend())
        
        # Основной цикл обработки задач
        while self.running:
            try:
                task = await asyncio.wait_for(self.input_queue.get(), timeout=1.0)
                
                # Если инициализация бэкенда уже провалилась, не ждём и не пытаемся
                # обработать задачу — сразу возвращаем ошибку клиенту.
                # Это позволяет клиенту получить ответ мгновенно, а не ждать
                # WAIT_FOR_BACKEND_TIMEOUT секунд до таймаута.
                if self._init_failed:
                    result = VisionResult(
                        task_id=task.task_id,
                        success=False,
                        error="Backend initialization failed - service misconfigured"
                    )
                    await self.output_queue.put(result)
                    self.input_queue.task_done()
                    continue
                
                # Ждём бэкенд с таймаутом
                timeout = config.WAIT_FOR_BACKEND_TIMEOUT  # Максимальное время ожидания инициализации
                start_wait = time.time()
                while self.backend is None and self.running:
                    if time.time() - start_wait > timeout:
                        result = VisionResult(
                            task_id=task.task_id,
                            success=False,
                            error="Backend initialization timeout"
                        )
                        await self.output_queue.put(result)
                        break
                    await asyncio.sleep(0.1)
                
                if self.backend and self.running:
                    result = await self._process_task(task)
                    await self.output_queue.put(result)
                    
                self.input_queue.task_done()
                self.tasks_processed += 1
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Worker error: {e}")

    async def _init_backend(self):
        """Фоновая инициализация бэкенда."""
        try:
            await self.initialize()
            logger.info("Backend initialized successfully")
        except Exception as e:
            # Инициализация провалилась. Устанавливаем флаг, чтобы воркер знал
            # об этом и не пытался обрабатывать задачи.
            # Само исключение не перевыбрасываем, так как оно уже залогировано,
            # и воркер должен продолжить работу в режиме "возвращать ошибки на все запросы",
            # а не падать полностью.
            self._init_failed = True
            logger.error(f"Failed to initialize backend: {e}")

    async def _process_task(self, task: VisionTask) -> VisionResult:
        """Обрабатывает одну задачу."""
        start_time = time.time()

        try:
            text = await self.backend.process(task.image_bytes, task.prompt)

            logger.debug(f"Task {task.task_id}: {len(text)} chars, { (time.time() - start_time) * 1000:.0f}ms")

            return VisionResult(
                task_id=task.task_id,
                success=True,
                text=text,
                processing_time_ms=(time.time() - start_time) * 1000
            )

        except Exception as e:
            logger.error(f"Task {task.task_id} failed: {e}")
            return VisionResult(
                task_id=task.task_id,
                success=False,
                error=str(e),
                processing_time_ms=(time.time() - start_time) * 1000
            )

    def is_healthy(self) -> bool:
        return self.backend is not None and self.backend.is_healthy()

    async def stop(self):
        """Остановка и освобождение ресурсов."""
        self.running = False
        if self.backend:
            await self.backend.close()
        logger.info(f"Worker stopped. Processed: {self.tasks_processed}")