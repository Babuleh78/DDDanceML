# app/main.py
import logging
from fastapi import FastAPI

from app.api.routes.process import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(
    title="Dance ML Service",
    description="Video to skeleton + movement segments",
    version="0.1.0",
)

app.include_router(router)

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.get("/")
async def root():
    return {
        "service": "DDDance ML Service",
        "version": "0.1.0",
        "status": "running"
    }