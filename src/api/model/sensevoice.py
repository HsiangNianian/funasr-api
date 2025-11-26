from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI
from fastapi.logger import logger

status = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global status
    logger.info("Loading SenseVoice model...")
    status = True
    yield
    status = False
    logger.info("SenseVoice model unloaded.")
    

router = APIRouter(prefix="/sensevoice", tags=["sensevoice"], lifespan=lifespan)

@router.get("/status")
async def get_sense_voice_status():
    return {"model": "SenseVoice", "status": status}