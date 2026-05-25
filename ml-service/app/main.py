# app/main.py
import asyncio
import logging

from fastapi import FastAPI

from app.api.routes.moderate import router as moderate_router
from app.api.routes.process import router as process_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Dance ML Service",
    description="Video to skeleton + movement segments",
    version="0.1.0",
)

app.include_router(process_router)
app.include_router(moderate_router)


@app.on_event("startup")
async def _startup() -> None:
    from app.telegram_bot.bot import start_bot
    asyncio.create_task(start_bot())


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def root():
    return {
        "service": "DDDance ML Service",
        "version": "0.1.0",
        "status": "running",
    }
