import logging
import json
import tempfile
import random
from pathlib import Path
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from app.api.deps import verify_internal_token
from app.schemas.process import ProcessRequest
from app.worker.tasks import process_video_task
from app.schemas.process import ProcessUrlRequest
from app.worker.tasks import process_video_url_task
from app.worker.celery_app import celery_app
from app.core import s3 as s3_client
import httpx
import os
from app.schemas.compare import (
    DanceCompareRequest,
    DanceCompareResponse,
    CompareTipsRequest,
    CompareTipsResponse,
    CompareTip,
)
from app.worker.tasks import compare_dance_task

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ml",
    tags=["processing"],
    dependencies=[Depends(verify_internal_token)],
)

GPU_SERVER_URL = os.getenv("GPU_SERVER_URL", "http://model-server:9001")
HEALTH_TIMEOUT = float(os.getenv("HEALTH_TIMEOUT", "3.0"))
INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "300.0"))

DESCRIPTION_SYSTEM_PROMPT = (
    "Ты спортивный комментатор танца. Твоя задача — ОТРАЖАТЬ ХАРАКТЕР ДВИЖЕНИЙ БЕЗ ДОДУМЫВАНИЙ. "
    "Никогда не используй слова 'плавный', 'гармоничный', 'мягкий', 'спокойный' для резких или быстрых движений. "
    "Если движения резкие — пиши 'резко', 'рвано', 'взрывно', 'остро'. "
    "Если движения быстрые — 'быстро', 'стремительно', 'динамично'. "
    "Если движения жёсткие — 'жёстко', 'энергично', 'силовые акценты'. "

    "ЗАПРЕЩЕНО постоянно начинать с головы/торса и идти сверху вниз. "
    "ЗАПРЕЩЕНО повторять из раза в раз одну и ту же структуру описания. "
    "ЗАПРЕЩЕНО шаблон 'голова → торс → руки → ноги' или любая его вариация. "
    
    "МЕНЯЙ порядок упоминания частей тела от сегмента к сегменту. "
    "Иногда начинай с ног, иногда с рук, иногда с общего образа. "
    "Иногда вообще не перечисляй части тела, а опиши только общую энергию. "
    "Иногда вплетай части тела в метафоры: 'взмахи раскачивают пространство', 'ступни бьют ритм в пол'. "
    "Иногда используй синтаксическое слияние: 'голова и кисти движутся в унисон' вместо отдельных предложений. "
    
    "Примеры вариативности: "
    "'Ноги задают мощный ритм, а руки разрезают пространство острыми акцентами.' (начало с низу) "
    "'Взрывной порыв энергии сквозь всё тело — от макушки до пяток.' (общий образ) "
    "'Кисти стремительно чертят воздух, подхваченные динамикой плеч и корпуса.' (начало с рук, слияние) "
    "'Умеренный темп с силовыми акцентами в руках и мягким откликом в корпусе.' (фокус на темпе, не на перечислении) "
    
    "Два предложения максимум. Без цифр и технических терминов. "
    "ОТВЕЧАЙ ТОЛЬКО НА РУССКОМ ЯЗЫКЕ. Без английских слов."
)
def _contains_non_cyrillic_text(text: str) -> bool:
    import re
    non_cyrillic_pattern = re.compile(r'[^а-яА-ЯёЁ\s\.\,\!\?\;\:\-\(\)\"\'\`]')
    return bool(non_cyrillic_pattern.search(text))

async def _check_gpu_server() -> bool:
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
        return f"GPU сервер недоступен! Мы постараемся все починить, чтобы вы увидели описание сегмента {segment_idx+1}"

    max_attempts = 5

    for attempt in range(1, max_attempts + 1):
        temperature = round(random.uniform(0.2, 0.6), 2)
        logger.info(f"Попытка {attempt}/{max_attempts}, temperature={temperature}")

        payload = {
            "model": "local",
            "stream": False,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": 200,
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

        except (httpx.HTTPError, KeyError, json.JSONDecodeError) as e:
            logger.error(
                f"Попытка {attempt}/{max_attempts}: ошибка запроса к GPU: "
                f"{type(e).__name__}: {e}"
            )
            continue

    logger.error(f"Все {max_attempts} попыток описания сегмента не удались")
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Не удалось получить описание сегмента от GPU-сервера",
    )


@router.post("/process")
async def process(req: ProcessRequest):
    task = process_video_task.delay(
        video_key=req.video_key,
        dance_id=req.dance_id,
        enable_labeling=req.enable_labeling,
        uploader_user_id=req.uploader_user_id,
    )
    return {"task_id": task.id, "dance_id": req.dance_id, "status": "queued"}


_STAGE_LABELS = {
    "queued":           "Ожидание в очереди",
    "codec_check":      "Проверка видео",
    "moderation":       "Проверка контента",
    "pose_extraction":  "Извлечение движений",
    "segmentation":     "Анализ сегментов",
    "segment_analysis": "Анализ сегментов",
    "comparing":        "Сравнение движений",
    "animation_render": "Рендер 3D-анимации",
    "saving":           "Сохранение результатов",
    "done":             "Готово",
    "failed":           "Ошибка",
}


@router.get("/status/{task_id}")
async def get_status(task_id: str):
    task = celery_app.AsyncResult(task_id)

    if task.state == "SUCCESS":
        return {
            "status": "done",
            "stage": "done",
            "stage_label": "Готово",
            "progress": 100,
            "result": task.result,
        }

    if task.state == "FAILURE":
        logger.error(f"task {task_id} failed: {task.info}")
        return {
            "status": "failed",
            "stage": "failed",
            "stage_label": "Ошибка",
            "progress": 0,
            "error": "Обработка не удалась",
        }

    if task.state == "PROGRESS":
        meta = task.info or {}
        stage = meta.get("stage", "processing")
        progress = int(meta.get("progress", 50))
        label = meta.get("stage_label", _STAGE_LABELS.get(stage, "Обработка"))
        return {
            "status": "processing",
            "stage": stage,
            "stage_label": label,
            "progress": progress,
        }

    return {
        "status": "queued",
        "stage": "queued",
        "stage_label": "Ожидание в очереди",
        "progress": 0,
    }


@router.post("/process-url/")
async def process_url(req: ProcessUrlRequest):
    logger.info(f"Enqueue URL: {req.url}, dance_id={req.dance_id}")
    task = process_video_url_task.delay(
        url=req.url,
        dance_id=req.dance_id,
        enable_labeling=req.enable_labeling,
        uploader_user_id=req.uploader_user_id,
    )
    return {"task_id": task.id, "dance_id": req.dance_id, "status": "queued"}


@router.get("/segment_description/{dance_id}/{segment_idx}")
async def get_segment_description(dance_id: str, segment_idx: int):
    try:
        segments_key = f"results/{dance_id}/segments.json"
        desc_key = f"results/{dance_id}/segment_descriptions/{segment_idx}.txt"

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
            
            cached_description = segment.get("llm_description")
            if cached_description: 
                logger.info(f"Cache hit for segment {segment_idx}: returning cached llm_description")
                return {
                    "dance_id": dance_id,
                    "segment_idx": segment_idx,
                    "description": cached_description,
                    "from_cache": True,
                }

            if s3_client.file_exists(desc_key):
                desc_path = str(Path(tmpdir) / "desc.txt")
                s3_client.download_file(desc_key, desc_path)
                with open(desc_path, "r", encoding="utf-8") as cached_f:
                    cached_text = cached_f.read().strip()
                if cached_text:
                    logger.info(f"Cache hit (desc object) for segment {segment_idx}")
                    return {
                        "dance_id": dance_id,
                        "segment_idx": segment_idx,
                        "description": cached_text,
                        "from_cache": True,
                    }

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

            desc_path = str(Path(tmpdir) / "desc.txt")
            with open(desc_path, "w", encoding="utf-8") as out_f:
                out_f.write(description)
            logger.info(f"Caching description for segment {segment_idx} -> {desc_key}")
            s3_client.upload_file(desc_path, desc_key)

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
        attempt_id=req.attempt_id,
    )
    return {
        "task_id": task.id,
        "dance_id": req.dance_id,
        "user_id": req.user_id,
        "status": "queued",
    }


TIPS_SYSTEM_PROMPT = ( # До лучших времен
    "Ты тренер по танцам. На вход — общий результат попытки (0-100) и слабые места по сегментам "
    "с тегами feedback: on_time / early / late / low_amplitude. "
    "Твоя задача — дать 2-3 коротких совета (каждый одна фраза, 6-14 слов, на русском). "
    "Каждый совет должен быть конкретным и действенным — что именно подправить в следующей попытке. "
    "Никаких общих слов вроде 'тренируйся больше' или 'будь увереннее'. "
    "Никаких цифр и технических терминов (DTW, амплитуда в радианах и т.п.). "
    "Тип совета: 'warn' — там, где есть явная проблема (низкий score < 50 или плохой feedback). "
    "Тип 'info' — где скорее похвала с лёгкой подсказкой (score 50-75). "
    "Если общий результат >= 80 — хотя бы один совет должен быть info-похвалой. "
    "ОТВЕЧАЙ СТРОГО В ФОРМАТЕ JSON: "
    '{"tips": [{"type": "warn", "text": "..."}, {"type": "info", "text": "..."}]}. '
    "Никакого markdown, никаких комментариев вне JSON, никаких символов до или после JSON."
)

FEEDBACK_RU = {
    "on_time": "в ритм",
    "early": "поторопился",
    "late": "опоздал",
    "low_amplitude": "малая амплитуда",
}


def _build_tips_user_prompt(req: CompareTipsRequest) -> str:
    weak = [
        s for s in req.segments
        if s.score < 60 or (s.feedback and s.feedback != "on_time")
    ]
    weak.sort(key=lambda s: s.score)
    weak = weak[:4]

    lines = [f"Общий результат: {round(req.attempt_score)}/100"]
    if not weak:
        lines.append("Явных слабых мест по сегментам нет — попытка ровная.")
    else:
        lines.append("Слабые сегменты:")
        for s in weak:
            label = s.label or f"Сегмент {s.segment_id + 1}"
            fb = FEEDBACK_RU.get(s.feedback or "", s.feedback or "")
            fb_part = f", {fb}" if fb else ""
            lines.append(
                f"- {label}: общий {round(s.score)}, тайминг {round(s.timing)}, "
                f"амплитуда {round(s.amplitude)}, точность {round(s.pose_accuracy)}{fb_part}"
            )

    return "\n".join(lines) + "\n\nВыдай советы строго в JSON."


def _parse_tips_json(content: str) -> List[CompareTip]:
    text = content.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"No JSON object found in LLM response: {content[:200]}")
    payload = json.loads(text[start:end + 1])
    raw_tips = payload.get("tips", [])
    result: List[CompareTip] = []
    for raw in raw_tips:
        if not isinstance(raw, dict):
            continue
        t = raw.get("type")
        txt = raw.get("text")
        if t not in ("warn", "info") or not isinstance(txt, str) or not txt.strip():
            continue
        result.append(CompareTip(type=t, text=txt.strip()))
        if len(result) >= 3:
            break
    return result


async def _generate_tips(req: CompareTipsRequest) -> List[CompareTip]:
    gpu_available = await _check_gpu_server()
    if not gpu_available:
        logger.warning(f"GPU-сервер недоступен ({GPU_SERVER_URL}), tips пропускаем")
        return []

    user_prompt = _build_tips_user_prompt(req)

    max_attempts = 3
    last_error: Optional[str] = None
    for attempt in range(1, max_attempts + 1):
        temperature = round(random.uniform(0.3, 0.7), 2)
        payload = {
            "model": "local",
            "stream": False,
            "temperature": temperature,
            "top_p": 0.9,
            "max_tokens": 300,
            "messages": [
                {"role": "system", "content": TIPS_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=INFERENCE_TIMEOUT) as client:
                response = await client.post(f"{GPU_SERVER_URL}/api/chat", json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["message"]["content"]
            tips = _parse_tips_json(content)
            if not tips:
                last_error = f"empty tips parsed, raw={content[:200]}"
                logger.warning(f"Попытка {attempt}: {last_error}")
                continue
            return tips
        except (json.JSONDecodeError, ValueError) as e:
            last_error = str(e)
            logger.warning(f"Попытка {attempt}: не распарсили JSON: {e}")
            continue
        except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
            last_error = str(e)
            logger.error(f"Попытка {attempt}: ошибка запроса к GPU: {e}")
            continue

    logger.error(f"Все попытки tips не удались: {last_error}")
    return []


@router.post("/compare_tips", response_model=CompareTipsResponse)
async def compare_tips(req: CompareTipsRequest):
    tips = await _generate_tips(req)
    return CompareTipsResponse(tips=tips)


@router.get("/health")
async def health_check():
    return {"status": "ok"}