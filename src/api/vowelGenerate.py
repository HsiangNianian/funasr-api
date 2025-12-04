"""Vowel API"""

from contextlib import asynccontextmanager
import os
from datetime import datetime
from enum import Enum
import sys
from typing import Dict, Optional, Union
import uuid
import aiohttp
import re
import uvicorn
from pathlib import Path
import asyncio
import argparse
from fastapi import BackgroundTasks, FastAPI, UploadFile
from fastapi.responses import RedirectResponse  # noqa: F401
from loguru import logger
from pydantic import BaseModel
from src.vowel import VowelExtractor
from src.apis.yunji_oss import put_file

MSST_API_URL = "http://localhost:25569"  # 183.147.142.111:63351

extractor = VowelExtractor()
UPLOAD_DIR = Path.cwd() / "uploads"
OUTPUT_DIR = Path.cwd() / "outputs"

argv = argparse.ArgumentParser()
argv.add_argument("--host", type=str, default="0.0.0.0")
argv.add_argument("--port", type=int, default=25568)
argv.add_argument("--up-load-dir", type=str, default=UPLOAD_DIR)
argv.add_argument("--out-put-dir", type=str, default=OUTPUT_DIR)
argv.add_argument("--msst-api-url", type=str, default=MSST_API_URL)
argv.add_argument("--audio-path", type=str, default=None)
args = argv.parse_args()

print(args)


@asynccontextmanager
async def _load_model(fastapi: FastAPI):
    logger.info("loading emotion2vec model...")
    extractor.load_models()
    logger.info("emotion2vec model loaded.")
    yield


app = FastAPI(lifespan=_load_model, description="Vowel API")


class TaskStatus(str, Enum):
    """Task Status"""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class ProcessingResponse(BaseModel):
    """Process Response Result"""

    taskId: str
    status: TaskStatus = TaskStatus.PENDING
    message: str


class TaskInfo(BaseModel):
    """Task Info Information"""

    taskId: str
    status: Optional[TaskStatus] = TaskStatus.PENDING
    url: Optional[str] = ""
    text: Optional[str] = ""
    emotion: Optional[str] = ""

class TaskResult(BaseModel):
    """Task Result Information"""

    taskId: str
    status: Optional[TaskStatus] = TaskStatus.PENDING
    url: Optional[str] = ""
    text: Optional[str] = ""
    emotion: Optional[str] = ""


class RequestBody(BaseModel):
    """Request Body Shape"""

    url: str
    """Audio Path or Url Address"""
    order_no: str
    """Order Register Number"""


tasks: Dict[str, TaskInfo] = {}


def is_valid_url(url: str) -> bool:
    # 简单的url正则校验
    regex = re.compile(r'^(https?|ftp)://[\w.-]+(?:\.[\w\.-]+)+[/\w\.-]*$')
    return re.match(regex, url) is not None

async def download_audio_from_url(url: str, save_path: str | Path) -> bool:
    if not is_valid_url(url):
        raise ValueError(f"URL不合法: {url}")
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    print(f"Downlaod failed: HTTP {response.status} — {url}")
                    return False

                with open(save_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        f.write(chunk)

        print(f"Download finished: {save_path}")
        return True

    except Exception as e:
        print(f"Download failed! {e} — URL: {url}")
        return False


@app.get("/")
async def _():
    return RedirectResponse(url="/docs")


async def separate_msst_audio_file(audio_file, target="vocals"):
    if not os.path.exists(audio_file):
        raise FileNotFoundError(f"文件不存在: {audio_file}")

    data = aiohttp.FormData()
    data.add_field(
        "file",
        open(audio_file, "rb"),
        filename=os.path.basename(audio_file),
        content_type="audio/wav",
    )
    data.add_field("model", "mel-band-roformer")
    data.add_field("output_format", "wav")
    data.add_field("target", target)

    timeout = aiohttp.ClientTimeout(total=300)
    async with aiohttp.ClientSession().post(
        f"{args.msst_api_url}/v1/audio/separations", data=data, timeout=timeout
    ) as response:
        response.raise_for_status()
        result = await response.json()

    print(f"ID: {result['id']}")

    # download
    async with aiohttp.ClientSession().get(
        f"{args.msst_api_url}{result['data']['files']['vocals']['url']}"
    ) as resp:
        resp.raise_for_status()
        with open(
            os.path.join(
                args.out_put_dir, result["data"]["files"]["vocals"]["filename"]
            ),
            "wb",
        ) as f:
            async for chunk in resp.content.iter_chunked(8192):
                f.write(chunk)
    return result["data"]["files"]["vocals"]["filename"]


from fastapi import HTTPException

@app.post("/api/v1/process")
async def process_audio(
    background_task: BackgroundTasks, url: str, order_no: Optional[str] = None
) -> TaskInfo:
    # 参数校验
    if not url or not isinstance(url, str):
        raise HTTPException(status_code=400, detail="参数url不能为空且必须为字符串类型")
    if not is_valid_url(url):
        raise HTTPException(status_code=400, detail=f"参数url不合法: {url}")

    taskId = str(uuid.uuid4())
    task_upload_dir = Path(args.up_load_dir) / taskId
    task_upload_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(url).stem
    filesuffix = Path(url).suffix
    audio_path = Path(task_upload_dir / (filename + filesuffix))
    print("Downloading audio from URL...")
    try:
        success = await download_audio_from_url(url=url, save_path=audio_path)
        if not success:
            raise HTTPException(status_code=400, detail="音频下载失败，请检查url是否有效或资源是否可访问")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"音频下载异常: {str(e)}")

    print("Audio downloaded:", audio_path)
    print("Separating vocals using MSST...")
    try:
        vocals_name_after_msst = await separate_msst_audio_file(audio_file=audio_path)
    except FileNotFoundError as fe:
        raise HTTPException(status_code=400, detail=str(fe))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"音频分离异常: {str(e)}")

    vocals_audio_after_msst_path = Path(args.out_put_dir) / vocals_name_after_msst
    print("Vocals separated:", vocals_audio_after_msst_path)

    task_info = TaskInfo(taskId=taskId)
    tasks[taskId] = task_info
    background_task.add_task(
        process_audio_task, taskId=taskId, audio_path=vocals_audio_after_msst_path
    )
    return TaskInfo(taskId=taskId)


@app.get("/api/v1/download/{taskId}")
@app.post("/api/v1/download/{taskId}")
async def download_file(taskId: str):
    task = tasks[taskId]
    if task.status != TaskStatus.COMPLETED:
        return {"error": "Task not completed yet."}
    task_result = task  # type: ignore
    return TaskResult(
        taskId=task_result.taskId,
        status=task_result.status,
        url=task_result.url,
        text=task_result.text,
        emotion=task_result.emotion,
    )

@app.post("/api/v1/query")
@app.get("/api/v1/query")
async def query_task_status(taskId: str):
    task = tasks.get(taskId)
    if not task:
        return {"error": "Task ID not found."}
    return TaskInfo(taskId=task.taskId, status=task.status)

async def process_audio_task(taskId: str, audio_path: Path):
    task = tasks[taskId]
    try:
        task.status = TaskStatus.PROCESSING
        print("Processing...")
        output_dir = Path(args.out_put_dir) / taskId
        output_dir.mkdir(parents=True, exist_ok=True)
        loop = asyncio.get_event_loop()
        emotion_segments = await loop.run_in_executor(
            None, extractor, audio_path, output_dir
        )

        # print(emotion_segments)
        sort_emotion_segments = sorted(
            emotion_segments, key=lambda x: x.quality_score, reverse=True
        )
        best_segment = sort_emotion_segments[0]
        res = await put_file(best_segment.segment_path)
        if res is False:
            raise Exception("上传文件失败")
        print(best_segment)
        task.status = TaskStatus.COMPLETED
        task_result = TaskResult(
            taskId=taskId,
            status=TaskStatus.COMPLETED,
            url=res[1], # type: ignore
            text=best_segment.text,
            emotion=best_segment.emotion,
        )
        tasks[taskId] = task_result  #   type:  ignore
        return task_result
    except Exception as e:
        task.status = TaskStatus.FAILED
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host=args.host, port=args.port)
