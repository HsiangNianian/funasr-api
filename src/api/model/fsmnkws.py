from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI
from fastapi.logger import logger

status = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global status
    logger.info("Loading FSMNKWS model...")
    status = True
    yield
    status = False
    logger.info("FSMNKWS model unloaded.")
    

router = APIRouter(prefix="/fsmnkws", tags=["fsmn-kws"], lifespan=lifespan)

@router.get("/status")
async def get_fsmn_kws_status():
    return {"model": "FSMNKWS", "status": status}