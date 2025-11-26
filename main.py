import asyncio
import sys
import uvicorn
from contextlib import asynccontextmanager
from typing import Union
from pydantic import FilePath
from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from src.log import logger
from src.config import MainConfig
from src.api.model import (
    campplus,
    conformer,
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

if sys.version_info >= (3, 11):  # pragma: no cover
    import tomllib
else:  # pragma: no cover
    import tomli as tomllib
    


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield


app = FastAPI(description="FunASR API", title="FunASR", lifespan=lifespan)

app.include_router(campplus.router)
app.include_router(conformer.router)
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


async def read_config(config_path: Union[FilePath, str]) -> MainConfig:
    """Read configuration from a TOML file."""
    if isinstance(config_path, str):
        from pathlib import Path
        config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r") as f:
        try:
            config = tomllib.loads(f.read())
        except tomllib.TOMLDecodeError as e:
            logger.error(f"Failed to parse config file: {e}")
            return MainConfig()
    return MainConfig.model_validate(config)
    

if __name__ == "__main__":
    logger.info("Starting FunASR API server...")
    logger.debug(asyncio.run(read_config("config.toml")))
    uvicorn.run(
        "main:app", host="0.0.0.0", port=8000, reload=True, log_level="info", workers=1
    )
