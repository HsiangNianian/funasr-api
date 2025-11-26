from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI
from fastapi.logger import logger

status = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global status
    logger.info("Loading QwenAudio model...")
    status = True
    yield
    status = False
    logger.info("QwenAudio model unloaded.")
    

router = APIRouter(prefix="/qwenaudio", tags=["qwen-audio"], lifespan=lifespan)

@router.get("/status")
async def get_qwen_audio_status():
    return {"model": "QwenAudio", "status": status}