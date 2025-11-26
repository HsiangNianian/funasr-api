from contextlib import asynccontextmanager
from fastapi import APIRouter, FastAPI
from fastapi.logger import logger

status = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    global status
    logger.info("Loading Emotion2Vec model...")
    status = True
    yield
    status = False
    logger.info("Emotion2Vec model unloaded.")
    

router = APIRouter(prefix="/emotion2vec", tags=["emotion2vec"], lifespan=lifespan)

@router.get("/status")
async def get_emotion2vec_status():
    return {"model": "Emotion2Vec", "status": status}