
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from src.api.model import (
    campplus,
    ctpunc,
    emotion2vec,
    fazh,
    fsmnkws,
    fsmnvad,
    paraformer,
    qwenaudio,
    sensevoice,
    whisper,
)
from src.app import FunasrApp

funasr_app = FunasrApp()

@asynccontextmanager
async def lifespan(app: FastAPI):
    await funasr_app.init()
    app.state.funasr_app = funasr_app
    yield

app = FastAPI(description="FunASR API", title="FunASR", lifespan=lifespan)

app.include_router(campplus.router)
app.include_router(ctpunc.router)
app.include_router(emotion2vec.router)
app.include_router(fazh.router)
app.include_router(fsmnkws.router)
app.include_router(fsmnvad.router)
app.include_router(paraformer.router)
app.include_router(qwenaudio.router)
app.include_router(sensevoice.router)
app.include_router(whisper.router)

@app.get("/")
async def _():
    return RedirectResponse(url="/docs")

if __name__ == "__main__":
    uvicorn.run(
        "main:app", host="0.0.0.0", port=6959, reload=True, log_level="info", workers=1
    )
