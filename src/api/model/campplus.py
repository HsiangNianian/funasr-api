from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI
from fastapi.logger import logger

status = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global status
    logger.info("Loading CAMPPlus model...")
    status = True
    yield
    status = False
    logger.info("CAMPPlus model unloaded.")
    
router = APIRouter(prefix="/campplus", tags=["campplus"], lifespan=lifespan)

@router.get("/status")
async def get_campplus_status():
    return {"model": "CAMPPlus", "status": status}