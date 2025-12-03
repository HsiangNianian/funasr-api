

from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI
from fastapi.logger import logger
from src.app import FunasrApp

status = False
model = None
funasr_app = None  # type: ignore

@asynccontextmanager
async def lifespan(app: FastAPI):
    global status, model, funasr_app
    logger.info("Loading SenseVoice model...")
    if funasr_app is None:
        funasr_app = FunasrApp()
        await funasr_app.init()
    try:
        model = await funasr_app.get_model("sensevoice")
        status = True if model else False
    except Exception as e:
        logger.error(f"SenseVoice model load failed: {e}")
        status = False
    yield
    status = False
    model = None
    logger.info("SenseVoice model unloaded.")
    

router = APIRouter(prefix="/sensevoice", tags=["sensevoice"], lifespan=lifespan)

@router.get("/status")
async def get_sense_voice_status():
    return {"model": "SenseVoice", "status": status}