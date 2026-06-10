"""
Vision Service — универсальный сервис для работы с изображениями.
Поддерживает: OCR (извлечение текста) и описание изображений.
"""

import asyncio
import uuid
import time
import base64
import re
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import slowapi
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

import setproctitle
setproctitle.setproctitle("vision_service")

from shared.config import config
from shared.model_configs import MODEL_REGISTRY
from workers.vision_worker import VisionWorker, VisionTask, VisionResult
from shared.schemas import (
    OCRSingleRequest, OCRBatchRequest, OCRResponse, OCRBatchResponse,
    DescribeSingleRequest, DescribeBatchRequest, DescribeResponse, DescribeBatchResponse,
    HealthResponse
)
from shared.auth_service import require_header_secret
from logger_utils import get_logger

logger = get_logger(
    "VISION_SERVICE",
    level=config.LOGGING_LEVEL,
    log_file=str(config.LOG_PATH / "vision.log"),
    docker_mode=config.DOCKER_ENV
)

require_auth = require_header_secret if config.REQUIRE_AUTH else lambda: None

# ======================================================================
# Глобальные объекты
# ======================================================================

input_queue = asyncio.Queue(maxsize=config.INPUT_QUEUE_MAXSIZE)
output_queue = asyncio.Queue(maxsize=config.INPUT_QUEUE_MAXSIZE)
worker = VisionWorker(input_queue, output_queue)
dispatcher: dict[str, asyncio.Future] = {}


# ======================================================================
# Lifespan
# ======================================================================

# main.py — фрагмент с lifespan

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Управление жизненным циклом приложения.
    
    Проблема: Загрузка модели может занимать несколько минут.
    Если ждать её в lifespan, Uvicorn не запустит сервер до завершения загрузки.
    Решение: Запускаем загрузку модели в фоновой задаче и не ждём её.
    Сервер стартует сразу, а модель грузится параллельно.
    """
    
    # Проверяем, что модель существует в реестре
    if config.MODEL not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        logger.critical(f"Неизвестная модель: {config.MODEL}. Доступны: {available}")
        raise RuntimeError(f"Unknown model: {config.MODEL}")

    logger.info(f"Запуск Vision Service. Модель: {config.MODEL}, бэкенд: {config.BACKEND}")

    # Запускаем воркер и диспетчер в фоновых задачах
    # Важно: НЕ используем await, чтобы не блокировать запуск сервера
    worker_task = asyncio.create_task(worker.start())
    dispatcher_task = asyncio.create_task(result_dispatcher())
    
    # НЕ ждём worker.wait_until_ready()
    # Сервер стартует сразу, даже если модель ещё не загружена
    logger.info("Сервер запускается, модель загружается в фоне. /health вернёт 503 до готовности модели")
    
    # Сохраняем задачи в контекст, чтобы потом корректно завершить
    app.state.worker_task = worker_task
    app.state.dispatcher_task = dispatcher_task
    
    yield  # Здесь FastAPI начинает принимать запросы
    
    # Graceful shutdown
    logger.info("Остановка Vision Service...")
    worker.running = False
    worker_task.cancel()
    dispatcher_task.cancel()
    
    try:
        await asyncio.gather(worker_task, dispatcher_task, return_exceptions=True)
    except asyncio.CancelledError:
        pass
    
    await worker.stop()
    logger.info("Vision Service остановлен")


async def result_dispatcher():
    """Отправляет результаты обратно ожидающим клиентам."""
    while True:
        try:
            result: VisionResult = await output_queue.get()
            future = dispatcher.pop(result.task_id, None)
            if future and not future.done():
                future.set_result(result)
            output_queue.task_done()
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Dispatcher error: {e}")


# ======================================================================
# FastAPI приложение
# ======================================================================

app = FastAPI(
    lifespan=lifespan,
    title="Vision Service",
    description="OCR и описание изображений через VLM",
    version="2.0.0"
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

limiter = slowapi.Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, slowapi._rate_limit_exceeded_handler)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [f"{' -> '.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()]
    logger.warning(f"Validation error: {errors}")
    return JSONResponse(status_code=400, content={"error": "Validation error", "detail": "; ".join(errors)})


# ======================================================================
# Вспомогательные функции
# ======================================================================

def validate_mime_type(mime_type: str) -> None:
    if mime_type not in config.ALLOWED_MIME_TYPES:
        raise HTTPException(415, f"Unsupported MIME type. Allowed: {', '.join(config.ALLOWED_MIME_TYPES)}")


def extract_mime_and_data(image_b64: str) -> tuple[str, str]:
    mime_match = re.match(r"data:([\w/]+);base64,", image_b64)
    if mime_match:
        return mime_match.group(1), image_b64[mime_match.end():]
    return "image/png", image_b64


async def submit_task(image_bytes: bytes, prompt: str, task_prefix: str = "") -> VisionResult:
    """Отправляет задачу в очередь и ждёт результат."""
    if input_queue.qsize() >= config.INPUT_QUEUE_MAXSIZE * 0.9:
        raise HTTPException(503, "Service busy")

    task_id = f"{task_prefix}_{uuid.uuid4()}" if task_prefix else str(uuid.uuid4())
    future = asyncio.get_event_loop().create_future()
    dispatcher[task_id] = future

    task = VisionTask(task_id=task_id, image_bytes=image_bytes, prompt=prompt, created_at=time.time())

    try:
        await input_queue.put(task)
        return await asyncio.wait_for(future, timeout=config.PROCESS_TIMEOUT)
    except asyncio.TimeoutError:
        dispatcher.pop(task_id, None)
        future.cancel()
        raise HTTPException(504, f"Timeout after {config.PROCESS_TIMEOUT}s")
    except Exception:
        dispatcher.pop(task_id, None)
        raise


# ======================================================================
# OCR эндпоинты
# ======================================================================

@app.post("/ocr", response_model=OCRResponse)
@limiter.limit(config.RATE_LIMIT_OCR)
async def ocr_single(request: Request, ocr_request: OCRSingleRequest, _: None = Depends(require_auth)):
    """Распознавание текста из изображения."""
    start_time = time.time()
    try:
        mime_type, clean_b64 = extract_mime_and_data(ocr_request.image_base64)
        validate_mime_type(mime_type)
        image_bytes = base64.b64decode(clean_b64)

        model_config = MODEL_REGISTRY[config.MODEL]
        prompt = ocr_request.prompt or model_config.default_prompt or "Text Recognition:"

        result = await submit_task(image_bytes, prompt)

        return OCRResponse(
            success=result.success,
            task_id=result.task_id,
            text=result.text,
            processing_time_ms=result.processing_time_ms,
            error=result.error
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"OCR error: {e}")
        return OCRResponse(
            success=False,
            task_id=str(uuid.uuid4()),
            text="",
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )


@app.post("/ocr/batch", response_model=OCRBatchResponse)
@limiter.limit(config.RATE_LIMIT_OCR)
async def ocr_batch(request: Request, batch_request: OCRBatchRequest, _: None = Depends(require_auth)):
    """Пакетное распознавание текста."""
    start_time = time.time()
    images = batch_request.images

    model_config = MODEL_REGISTRY[config.MODEL]
    default_prompt = model_config.default_prompt or "Text Recognition:"

    # Подготовка промптов
    prompts = [batch_request.common_prompt or default_prompt] * len(images)
    if batch_request.prompts:
        if len(batch_request.prompts) == len(images):
            for i, p in enumerate(batch_request.prompts):
                if p:
                    prompts[i] = p

    tasks = []
    for i, img in enumerate(images):
        mime_type, clean_b64 = extract_mime_and_data(img)
        validate_mime_type(mime_type)
        image_bytes = base64.b64decode(clean_b64)
        tasks.append(submit_task(image_bytes, prompts[i], f"batch_{i}"))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    ocr_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            ocr_results.append(OCRResponse(success=False, task_id=f"batch_{i}", text="", error=str(r)))
        else:
            ocr_results.append(OCRResponse(
                success=r.success, task_id=r.task_id, text=r.text,
                processing_time_ms=r.processing_time_ms, error=r.error
            ))

    return OCRBatchResponse(
        success=all(r.success for r in ocr_results),
        task_id=f"batch_{uuid.uuid4()}",
        results=ocr_results,
        processing_time_ms=(time.time() - start_time) * 1000
    )


# ======================================================================
# Describe эндпоинты
# ======================================================================

@app.post("/describe", response_model=DescribeResponse)
@limiter.limit(config.RATE_LIMIT_OCR)
async def describe_single(request: Request, describe_request: DescribeSingleRequest, _: None = Depends(require_auth)):
    """Описание изображения."""
    start_time = time.time()
    try:
        mime_type, clean_b64 = extract_mime_and_data(describe_request.image_base64)
        validate_mime_type(mime_type)
        image_bytes = base64.b64decode(clean_b64)

        model_config = MODEL_REGISTRY[config.MODEL]
        prompt = describe_request.prompt or model_config.default_prompt or "Describe this image in detail."

        result = await submit_task(image_bytes, prompt)

        return DescribeResponse(
            success=result.success,
            task_id=result.task_id,
            description=result.text,
            processing_time_ms=result.processing_time_ms,
            error=result.error
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Describe error: {e}")
        return DescribeResponse(
            success=False,
            task_id=str(uuid.uuid4()),
            description="",
            processing_time_ms=(time.time() - start_time) * 1000,
            error=str(e)
        )


@app.post("/describe/batch", response_model=DescribeBatchResponse)
@limiter.limit(config.RATE_LIMIT_OCR)
async def describe_batch(request: Request, batch_request: DescribeBatchRequest, _: None = Depends(require_auth)):
    """Пакетное описание изображений."""
    start_time = time.time()
    images = batch_request.images

    model_config = MODEL_REGISTRY[config.MODEL]
    default_prompt = model_config.default_prompt or "Describe this image in detail."

    prompts = [batch_request.common_prompt or default_prompt] * len(images)
    if batch_request.prompts:
        if len(batch_request.prompts) == len(images):
            for i, p in enumerate(batch_request.prompts):
                if p:
                    prompts[i] = p

    tasks = []
    for i, img in enumerate(images):
        mime_type, clean_b64 = extract_mime_and_data(img)
        validate_mime_type(mime_type)
        image_bytes = base64.b64decode(clean_b64)
        tasks.append(submit_task(image_bytes, prompts[i], f"batch_{i}"))

    results = await asyncio.gather(*tasks, return_exceptions=True)

    describe_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            describe_results.append(DescribeResponse(success=False, task_id=f"batch_{i}", description="", error=str(r)))
        else:
            describe_results.append(DescribeResponse(
                success=r.success, task_id=r.task_id, description=r.text,
                processing_time_ms=r.processing_time_ms, error=r.error
            ))

    return DescribeBatchResponse(
        success=all(r.success for r in describe_results),
        task_id=f"batch_{uuid.uuid4()}",
        results=describe_results,
        processing_time_ms=(time.time() - start_time) * 1000
    )


# ======================================================================
# Health & Info
# ======================================================================

@app.get("/health", response_model=HealthResponse)
@limiter.limit(config.RATE_LIMIT_HEALTH)
async def health_check(request: Request):
    if not worker.is_healthy():
        status = "loading" if worker.backend is None else "unhealthy"
        return JSONResponse(
            status_code=503,
            content={
                "status": status,
                "backend": config.BACKEND,
                "model": config.MODEL,
                "message": "Model is still loading" if worker.backend is None else "Backend initialization failed"
            }
        )
    return HealthResponse(
        status="healthy",
        backend=config.BACKEND,
        model=config.MODEL,
        tasks_processed=worker.tasks_processed,
        queue_size=input_queue.qsize()
    )


@app.get("/info")
@limiter.limit(config.RATE_LIMIT_HEALTH)
async def get_info(request: Request):
    return {
        "service": "Vision Service",
        "version": "2.0.0",
        "backend": config.BACKEND,
        "model": config.MODEL,
        "allowed_mime_types": list(config.ALLOWED_MIME_TYPES),
        "max_image_size_mb": config.MAX_IMAGE_SIZE_MB,
        "max_queue_size": config.INPUT_QUEUE_MAXSIZE,
        "max_batch_size": config.MAX_BATCH_SIZE,
    }