from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI
from fastapi.logger import logger

status = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global status
    logger.info("Loading Conformer model...")
    status = True
    yield
    status = False
    logger.info("Conformer model unloaded.")
    
router = APIRouter(prefix="/conformer", tags=["conformer"], lifespan=lifespan)

@router.get("/status")
async def get_conformer_status():
    return {"model": "Conformer", "status": status}
