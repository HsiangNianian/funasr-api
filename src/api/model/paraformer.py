from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI
from fastapi.logger import logger

status = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global status
    logger.info("Loading Paraformer model...")
    status = True
    yield
    status = False
    logger.info("Paraformer model unloaded.")
    
router = APIRouter(prefix="/paraformer", tags=["paraformer"], lifespan=lifespan)

@router.get("/status")
async def get_paraformer_status():
    return {"model": "Paraformer", "status": status}
