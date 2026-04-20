import logging
import json
import tempfile
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
    "Ты — хореограф, который объясняет движения своим ученикам простым языком. "
    "Тебе дают технический анализ сегмента танца. Твоя задача — описать его так, "
    "как будто ты объясняешь ученику что происходит с его телом. "
    "Строгие правила: "
    "НИКАКИХ цифр, градусов, метров, BPM, Гц в ответе. "
    "НИКАКИХ технических терминов (амплитуда, диапазон, фаза). "
    "Используй только понятные танцору слова: широко, резко, плавно, высоко, быстро. "
    "3-5 предложений. Отвечай на русском языке. "
    "Примеры: вместо градусов пиши 'руки широко раскрыты', вместо BPM пиши 'быстрый темп'."
)


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
        return f"GPU сервер недоступен! Описание сегмента {segment_idx}"

    payload = {
        "model": "local",
        "stream": False,
        "temperature": 0.4,
        "top_p": 0.9,
        "max_tokens": 256,
        "messages": [
            {"role": "system", "content": DESCRIPTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Характеристики сегмента:\n{features}\n\nОпиши это движение БЕЗ каких-либо чисел и технических терминов. Отвечай на русском языке."},
        ],
    }

    logger.info(f"Запрос к GPU-серверу ({GPU_SERVER_URL}), dance_id={dance_id}, segment={segment_idx}")

    try:
        async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT) as client:
            logger.debug(f"→ Отправляю POST {GPU_SERVER_URL}/api/chat")
            response = await client.post(f"{GPU_SERVER_URL}/api/chat", json=payload)
            
            logger.debug(f"← Response status: {response.status_code}")
            logger.debug(f"← Response headers: {dict(response.headers)}")
            logger.debug(f"← Response raw (first 300 chars): {response.text[:300]}")
            
            response.raise_for_status()
            
            try:
                data = response.json()
            except json.JSONDecodeError as e:
                logger.error(f"Не удалось распарсить JSON: {e}")
                logger.error(f"Raw response text: {response.text[:500]}")
            
                try:
                    data = json.loads(response.content.decode('utf-8'))
                except Exception as e2:
                    raise HTTPException(
                        status_code=status.HTTP_502_BAD_GATEWAY,
                        detail=f"Некорректный ответ от GPU-сервера: {e2}",
                    )
            
            logger.debug(f" Parsed JSON keys: {list(data.keys()) if isinstance(data, dict) else 'N/A'}")
            
            content = None
            
            if isinstance(data, dict) and "message" in data and isinstance(data["message"], dict):
                content = data["message"].get("content")
            
            elif isinstance(data, dict) and "choices" in data and isinstance(data["choices"], list) and len(data["choices"]) > 0:
                content = data["choices"][0].get("message", {}).get("content")
            
            elif isinstance(data, dict) and "content" in data:
                content = data["content"]
            
            if content is None:
                logger.error(f"Не удалось извлечь content из ответа: {data}")
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Неизвестный формат ответа от GPU-сервера: {list(data.keys()) if isinstance(data, dict) else type(data)}",
                )
            
            return str(content).strip()

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

            features = segments[segment_idx].get("features")
            if not features:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Segment {segment_idx} has no 'features' field",
                )

        description = await _describe_segment(
            features=features,
            segment_idx=segment_idx,
            dance_id=dance_id,
        )

        return {
            "dance_id": dance_id,
            "segment_idx": segment_idx,
            "description": description,
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
    

 

 
 
@router.post("/dance_compare", response_model=None)
async def dance_compare(req: DanceCompareRequest):
    task = compare_dance_task.delay(
        video_key=req.video_key,
        dance_id=req.dance_id,
        segment_idx=req.segment_idx,
    )
    return {
        "task_id":     task.id,
        "dance_id":    req.dance_id,
        "segment_idx": req.segment_idx,
        "status":      "queued",
    }