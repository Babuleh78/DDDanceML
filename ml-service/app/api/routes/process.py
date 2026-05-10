import logging
import json
import tempfile
import random
from pathlib import Path
from fastapi import APIRouter, HTTPException, status
from app.schemas.process import ProcessRequest
from app.worker.tasks import process_video_task
from app.schemas.process import ProcessUrlRequest
from app.worker.tasks import process_video_url_task
from app.worker.celery_app import celery_app
from app.core import s3 as s3_client
import httpx
import os
from app.schemas.compare import DanceCompareRequest, DanceCompareResponse
from app.worker.tasks import compare_dance_task

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["processing"])

GPU_SERVER_URL = os.getenv("GPU_SERVER_URL", "http://model-server:9001")
HEALTH_TIMEOUT = float(os.getenv("HEALTH_TIMEOUT", "3.0"))
INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "300.0"))

DESCRIPTION_SYSTEM_PROMPT = (
    "Ты спортивный комментатор танца. Твоя задача — ОТРАЖАТЬ ХАРАКТЕР ДВИЖЕНИЙ БЕЗ ДОДУМЫВАНИЙ. "
    "Никогда не используй слова 'плавный', 'гармоничный', 'мягкий', 'спокойный' для резких или быстрых движений. "
    "Если движения резкие — пиши 'резко', 'рвано', 'взрывно', 'остро'. "
    "Если движения быстрые — 'быстро', 'стремительно', 'динамично'. "
    "Если движения жёсткие — 'жёстко', 'энергично', 'силовые акценты'. "
    "Не перечисляй части тела. Говори о характере и энергии. "
    "Два предложения максимум. Без цифр и технических терминов. "
    "ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. Без английских слов."
)

def _contains_non_cyrillic_text(text: str) -> bool:
    import re
    non_cyrillic_pattern = re.compile(r'[^а-яА-ЯёЁ\s\.\,\!\?\;\:\-\(\)\"\'\`]')
    return bool(non_cyrillic_pattern.search(text))

async def _check_gpu_server() -> bool:
    """Проверяет доступность GPU-сервера и готовность модели."""
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{GPU_SERVER_URL}/health",
                timeout=HEALTH_TIMEOUT,
            )
            logger.debug(f"Health response: status={resp.status_code}, body={resp.text}")
            
            if resp.status_code == 200:
                data = resp.json()
                model_loaded = data.get("model_loaded", False)
                logger.info(f"Health check: model_loaded={model_loaded}, full response={data}")
                return model_loaded
    except httpx.ConnectError as e:
        logger.warning(f"Health check connection error: {e}")
    except httpx.TimeoutException as e:
        logger.warning(f"Health check timeout ({HEALTH_TIMEOUT}s): {e}")
    except json.JSONDecodeError as e:
        logger.warning(f"Health check invalid JSON: {e}, raw={resp.text[:200]}")
    except Exception as e:
        logger.warning(f"Health check unexpected error: {type(e).__name__}: {e}")
    
    return False

async def _describe_segment(features: str, segment_idx: int, dance_id: str) -> str:
    gpu_available = await _check_gpu_server()

    if not gpu_available:
        logger.warning(f"GPU-сервер недоступен ({GPU_SERVER_URL}), возвращаем заглушку")
        return f"GPU сервер недоступен! Описание сегмента {segment_idx+1}"

    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        temperature = round(random.uniform(0.2, 0.6), 2)
        logger.info(f"Попытка {attempt}/{max_attempts}, temperature={temperature}")

        payload = {
            "model": "local",
            "stream": False,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": 120,
            "messages": [
                {"role": "system", "content": DESCRIPTION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Характеристики сегмента:\n{features}\n\nКоротко — что за движение?"},
            ],
        }

        try:
            async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT) as client:
                response = await client.post(f"{GPU_SERVER_URL}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["message"]["content"].strip()

                if _contains_non_cyrillic_text(content):
                    logger.warning(f"Попытка {attempt}: обнаружены нерусские символы, перегенерируем. Ответ: {content[:100]}")
                    continue

                return content

        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            logger.error(f"Попытка {attempt}: ошибка запроса: {e}")
            if attempt == max_attempts:
                raise HTTPException(status_code=503, detail=str(e))
            
            logger.error("Все попытки вернули нерусский текст, возвращаем заглушку")
            return f"Не удалось получить описание сегмента {segment_idx+1}"

        except httpx.ConnectError:
            logger.error(f"Не удалось подключиться к GPU-серверу: {GPU_SERVER_URL}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"GPU-сервер недоступен ({GPU_SERVER_URL})",
            )
        
        except httpx.TimeoutException:
            logger.error(f"Таймаут GPU-сервера после {INFERENCE_TIMEOUT}с")
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail=f"GPU-сервер не ответил за {INFERENCE_TIMEOUT} секунд",
            )
        
        except httpx.RemoteProtocolError as e:
            logger.error(f"Протокольная ошибка (сервер закрыл соединение): {e}")
            logger.error(f"Request payload preview: {json.dumps(payload, ensure_ascii=False)[:200]}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Сервер закрыл соединение во время передачи ответа",
            )
        
        except httpx.ReadError as e:
            logger.error(f"Ошибка чтения ответа от сервера: {e}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Не удалось прочитать ответ от GPU-сервера",
            )
        
        except httpx.DecodingError as e:
            logger.error(f"Ошибка декодирования ответа (возможно, кодировка): {e}")
            logger.error(f"Raw bytes preview: {response.content[:100]}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Не удалось декодировать ответ от GPU-сервера",
            )
        
        except httpx.HTTPStatusError as e:
            logger.error(f"Ошибка GPU-сервера {e.response.status_code}: {e.response.text[:300]}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"Ошибка GPU-сервера: {e.response.text[:300]}",
            )
        
        except KeyError as e:
            logger.error(f"Неожиданный формат ответа GPU-сервера (KeyError: {e}): {json.dumps(data, ensure_ascii=False)[:300] if 'data' in locals() else 'N/A'}")
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Неожиданный формат ответа от GPU-сервера",
            )
        
        except Exception as e:
            logger.error(f"Непредвиденная ошибка при запросе к GPU-серверу: {type(e).__name__}: {e}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Внутренняя ошибка при обращении к GPU-серверу: {str(e)}",
            )


@router.post("/process")
async def process(req: ProcessRequest):
    task = process_video_task.delay(
        video_key=req.video_key,
        dance_id=req.dance_id,
        enable_labeling=req.enable_labeling,
    )
    return {"task_id": task.id, "dance_id": req.dance_id, "status": "queued"}


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    task = celery_app.AsyncResult(task_id)
    if task.state == "SUCCESS":
        return {"status": "done", "result": task.result}
    if task.state == "FAILURE":
        return {"status": "failed", "error": str(task.info)}
    return {"status": task.state.lower()}


@router.post("/process-url/")
async def process_url(req: ProcessUrlRequest):
    logger.info(f"Enqueue URL: {req.url}, dance_id={req.dance_id}")
    task = process_video_url_task.delay(
        url=req.url,
        dance_id=req.dance_id,
        enable_labeling=req.enable_labeling,
    )
    return {"task_id": task.id, "dance_id": req.dance_id, "status": "queued"}


@router.get("/segment_description/{dance_id}/{segment_idx}")
async def get_segment_description(dance_id: str, segment_idx: int):
    try:
        segments_key = f"results/{dance_id}/segments.json"

        with tempfile.TemporaryDirectory() as tmpdir:
            local_path = str(Path(tmpdir) / "segments.json")
            s3_client.download_file(segments_key, local_path)

            with open(local_path, "r", encoding="utf-8") as f:
                segments_data = json.load(f)

            segments = segments_data.get("segments", [])
            if segment_idx < 0 or segment_idx >= len(segments):
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Segment {segment_idx} not found. Available: 0-{len(segments)-1}",
                )

            segment = segments[segment_idx]
            
            # === Проверка кэша: если llm_description уже есть - вернуть его ===
            cached_description = segment.get("llm_description")
            if cached_description:  # Если не None и не пусто
                logger.info(f"Cache hit for segment {segment_idx}: returning cached llm_description")
                return {
                    "dance_id": dance_id,
                    "segment_idx": segment_idx,
                    "description": cached_description,
                    "from_cache": True,
                }

            # === Кэш не попал: запросить у GPU-сервера ===
            text_features = segment.get("text_dance_features")
            if not text_features:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Segment {segment_idx} has no 'text_dance_features' field",
                )

            logger.info(f"Cache miss for segment {segment_idx}: requesting from GPU server...")
            description = await _describe_segment(
                features=text_features,
                segment_idx=segment_idx,
                dance_id=dance_id,
            )

            # === Сохранить описание в segments.json (кэширование) ===
            segment["llm_description"] = description
            
            # Обновить segments.json в S3
            segments_path = Path(local_path)
            with open(segments_path, "w", encoding="utf-8") as f:
                json.dump(segments_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Uploading updated segments.json with cached description for segment {segment_idx}")
            s3_client.upload_file(str(segments_path), segments_key)

            return {
                "dance_id": dance_id,
                "segment_idx": segment_idx,
                "description": description,
                "from_cache": False,
            }

    except HTTPException:
        raise
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Segments data not found for dance_id: {dance_id}",
        )
    except Exception as e:
        logger.error(f"Error getting segment description: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error retrieving segment: {str(e)}",
        )
    

@router.post("/dance_compare", response_model=DanceCompareResponse)
async def dance_compare(req: DanceCompareRequest):
    task = compare_dance_task.delay(
        original_video_s3_path=req.original_video_s3_path,
        user_video_s3_path=req.user_video_s3_path,
        user_id=req.user_id,
        dance_id=req.dance_id,
    )
    return {
        "task_id": task.id,
        "dance_id": req.dance_id,
        "user_id": req.user_id,
        "status": "queued",
    }


@router.get("/health")
async def health_check():
    return {"status": "ok"}