from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI
from fastapi.logger import logger

status = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global status
    logger.info("Loading CTPunc model...")
    status = True
    yield
    status = False
    logger.info("CTPunc model unloaded.")
    

router = APIRouter(prefix="/ctpunc", tags=["ct-punc"], lifespan=lifespan)

@router.get("/status")
async def get_ctpunc_status():
    return {"model": "CTPunc", "status": status}