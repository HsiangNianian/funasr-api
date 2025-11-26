from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI
from fastapi.logger import logger

status = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global status
    logger.info("Loading FAZH model...")
    status = True
    yield
    status = False
    logger.info("FAZH model unloaded.")
    

router = APIRouter(prefix="/fazh", tags=["fa-zh"], lifespan=lifespan)

@router.get("/status")
async def get_fazh_status():
    return {"model": "FAZH", "status": status}