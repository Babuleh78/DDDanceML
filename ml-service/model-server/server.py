import os
import logging
import torch
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODEL_PATH     = os.getenv("MODEL_PATH", "/models/adapter")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
HF_TOKEN       = os.getenv("HF_TOKEN", None)
USE_STUB       = os.getenv("USE_MODEL_STUB", "false").lower() == "true"

BASE_MODEL_GPU = os.getenv("BASE_MODEL_GPU", "Qwen/Qwen2.5-7B-Instruct")
BASE_MODEL_CPU = os.getenv("BASE_MODEL_CPU", "Qwen/Qwen2.5-1.5B-Instruct")

model     = None
tokenizer = None
device    = None


def detect_device() -> tuple[str, str, dict]:
    """Возвращает (device, base_model_name, load_kwargs)"""
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        vram_gb = gpu.total_memory / 1e9
        logger.info(f"✅ GPU: {gpu.name} | VRAM: {vram_gb:.1f} GB")

        if vram_gb < 12:
            logger.warning("VRAM < 12GB, используем 3B модель на GPU")
            base_model = "Qwen/Qwen2.5-3B-Instruct"
        else:
            base_model = BASE_MODEL_GPU

        return "cuda", base_model, {
            "device_map": "auto",
            "torch_dtype": torch.float16,
        }
    else:
        cpu_count = os.cpu_count()
        ram_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        logger.warning(
            f"⚠️  GPU не найден. CPU режим ({cpu_count} cores, {ram_gb:.1f} GB RAM). "
            f"Модель: {BASE_MODEL_CPU}"
        )
        return "cpu", BASE_MODEL_CPU, {
            "device_map": "cpu",
            "torch_dtype": torch.float32,
            "low_cpu_mem_usage": True,
        }


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer, device

    if USE_STUB:
        logger.warning("⚠️  STUB режим активен — реальная модель не загружается")
        device = "cpu"
        yield
        return

    device, base_model_name, load_kwargs = detect_device()

    logger.info(f"📂 Загружаю токенайзер для {base_model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        token=HF_TOKEN,
        use_fast=False,
    )

    # Пытаемся прочитать конфиг адаптера для информации
    adapter_base = None
    try:
        peft_config = PeftConfig.from_pretrained(MODEL_PATH)
        adapter_base = peft_config.base_model_name_or_path
        logger.info(f"📋 Найден адаптер для базовой модели: {adapter_base}")
    except Exception as e:
        logger.warning(f"Не удалось прочитать adapter_config: {e}")

    logger.info(f"🔄 Загружаю базовую модель {base_model_name} [{device}]...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        token=HF_TOKEN,
        **load_kwargs,
    )

    # Применяем адаптер только если есть GPU и адаптер существует
    if device == "cuda" and adapter_base is not None:
        if adapter_base != base_model_name:
            logger.warning(
                f"⚠️ Адаптер обучен на {adapter_base}, а используется {base_model_name}. "
                "Возможна несовместимость размеров."
            )
        logger.info(f"🔧 Применяю LoRA адаптер из {MODEL_PATH}...")
        model = PeftModel.from_pretrained(base, MODEL_PATH)
    else:
        if device == "cpu":
            logger.info("ℹ️ CPU режим: работаем без адаптера (стандартная модель).")
        else:
            logger.info("ℹ️ Адаптер не найден или не используется.")
        model = base

    model.eval()
    logger.info(f"✅ Модель готова на {device.upper()}!")
    yield

    del model, tokenizer
    if device == "cuda":
        torch.cuda.empty_cache()
    logger.info("Model unloaded.")


app = FastAPI(lifespan=lifespan)


class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: Optional[str] = "local"
    messages: list[Message]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = MAX_NEW_TOKENS
    stream: bool = False

class ChatResponse(BaseModel):
    message: dict


@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if USE_STUB:
        return {"message": {
            "role": "assistant",
            "content": (
                "[STUB] Сегмент обработан. Движение плавное, симметричное, "
                "задействованы руки и корпус. Темп умеренный, характер лирический."
            ),
        }}

    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Модель ещё загружается")

    try:
        text = tokenizer.apply_chat_template(
            [m.model_dump() for m in req.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                repetition_penalty=1.1,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        new_tokens = output_ids[0][prompt_len:]
        result = tokenizer.decode(new_tokens, skip_special_tokens=True)

        return {"message": {"role": "assistant", "content": result.strip()}}

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(status_code=507, detail="GPU OOM — уменьшите MAX_NEW_TOKENS")
    except Exception as e:
        logger.error(f"Inference error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    cuda_info = None
    if device == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        cuda_info = {
            "name": props.name,
            "vram_total_gb": round(props.total_memory / 1e9, 1),
            "vram_free_gb": round(
                (props.total_memory - torch.cuda.memory_allocated()) / 1e9, 1
            ),
        }
    return {
        "status": "ok" if (model is not None or USE_STUB) else "loading",
        "device": device,
        "stub": USE_STUB,
        "model_loaded": model is not None,
        "gpu": cuda_info,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")