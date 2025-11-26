from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI
from fastapi.logger import logger

status = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global status
    logger.info("Loading FSMNVAD model...")
    status = True
    yield
    status = False
    logger.info("FSMNVAD model unloaded.")
    

router = APIRouter(prefix="/fsmnvad", tags=["fsmn-vad"], lifespan=lifespan)

@router.get("/status")
async def get_fsmn_vad_tatus():
    return {"model": "FSMNVAD", "status": status}