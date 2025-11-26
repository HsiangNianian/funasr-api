from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI
from fastapi.logger import logger

status = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global status
    logger.info("Loading Whisper model...")
    status = True
    yield
    status = False
    logger.info("Whisper model unloaded.")
    

router = APIRouter(prefix="/whisper", tags=["whisper"], lifespan=lifespan)

@router.get("/status")
async def get_whisper_status():
    return {"model": "Whisper", "status": status}